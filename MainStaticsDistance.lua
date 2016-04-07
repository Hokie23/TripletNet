require ('mobdebug').start()
require "csvigo"
require 'cutorch'
require 'cunn'
require 'xlua'
require 'trepl'
require 'loadutils'
require 'cudnn'
require 'SiameseDistanceNet'
require 'AttentionLSTM'

----------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Testing a Siamese Distance network on Fashion')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
--cmd:option('-modelsFolder',       './Results/WedMar2314:56:452016/',            'Models Folder') -- 128 The best model
--cmd:option('-network',            'Embedding.t715',            'embedding network file - must return valid network.')
cmd:option('-modelsFolder',       './Results/WedApr614:32:082016/',            'Models Folder') -- 128 attention The best model
cmd:option('-network',            'Embedding.t713',            'embedding network file - must return valid network.')

--cmd:option('-modelsFolder',       './Results/TueMar2914:54:572016/',            'Models Folder') -- 128(relu) The best model
--cmd:option('-network',            'Embedding.t711',            'embedding network file - must return valid network.')
--
--cmd:option('-modelsFolder',       './Results/MonMar2116:45:482016/',            'Models Folder') -- 512 model
--cmd:option('-network',            'Embedding.t76',            'embedding network file - must return valid network.')
--cmd:option('-network',            'Embedding.t758',            'embedding network file - must return valid network.')

--cmd:option('-modelsFolder',       './Results/TueMar2220:13:462016/',            'Models Folder') -- 512 model
--cmd:option('-network',            'Embedding.t78',            'embedding network file - must return valid network.')

--cmd:option('-modelsFolder',       './Results/WedMar2319:55:362016/',            'Models Folder') -- 1024(relu) model
--cmd:option('-network',            'Embedding.t78',            'embedding network file - must return valid network.')

--cmd:option('-modelsFolder',       './Results/ThuMar3114:43:522016/',            'Models Folder') -- grid 135
--cmd:option('-network',            'Embedding.t73',            'embedding network file - must return valid network.')

cmd:text('===>Platform Optimization')
cmd:option('-batchSize',          14,                    'batch size')
cmd:option('-threads',            16,                      'number of threads')
cmd:option('-type',               'cuda',                 'float or cuda')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Data Options')
cmd:option('-imagePath','/data1/october_11st/october_11st_imgs/', 'image path directory')

opt = cmd:parse(arg or {})
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)

opt.network = opt.modelsFolder .. opt.network 
opt.save = paths.concat('./StaticsResults', opt.save)

lu = loadutils(opt.imagePath)

----------------------------------------------------------------------
-- Model + Loss:
print ("load...", opt.network)
--local EmbeddingNet = require(opt.network)
require 'cudnn'
local EmbeddingNet = torch.load(opt.network)
print ("EmbeddingNet:", EmbeddingNet)
print ('complete')

cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = true


EmbeddingNet:cuda()
local EmbeddingWeights, EmbeddingGradients = EmbeddingNet:getParameters()
if opt.load ~= '' then
    print ("load...", opt.load)
    local w = torch.load( opt.load )
    EmbeddingWeights:copy(w)
end
EmbeddingNet:evaluate()

local SiameseDistanceNet = nn.SiameseDistanceNet(EmbeddingNet)
SiameseDistanceNet:cuda()
SiameseDistanceNet:evaluate()

------------------------- Output files configuration -----------------
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
----------------------------------------------------------------------

--local fashion_test_pair = 'fashion_pair_test.csv'
local fashion_test_pair = 'fashion_pair_valid.csv'
--local fashion_test_pair = 'fashion_pair_train.csv'

print ('load pairs', fashion_test_pair)
local Resolution = lu:Resolution()
print ("Resolution", Resolution)


local test_pairs = lu:LoadPairs( fashion_test_pair)
print ('#loaded pairs:', #test_pairs)

local outputs = {}

function CalculateDistance()
    local batchSize = opt.batchSize
    local counter = 0
    local batch = torch.Tensor():type( 'torch.CudaTensor' )
    local nsz = torch.LongStorage(5)

    while counter < #test_pairs do
    --while counter < 34 do
        local bsize = math.min( #test_pairs - counter, batchSize )
        nsz[1] = 2
        nsz[2] = bsize
        nsz[3] = Resolution[1]
        nsz[4] = Resolution[2]
        nsz[5] = Resolution[3]

        if batch:dim() ~= 5 or batch:size(2) ~= bsize then
            batch:resize(nsz)
        end

        for i=1,bsize do
            --print( "counter", counter, "i", i )
            local a_name = test_pairs[counter + i][1]
            local t_name = test_pairs[counter + i][2]
            assert(a_name ~= nil)
            assert(t_name ~= nil)
            local is_positive = test_pairs[counter + i][3]
            local a_img = lu:LoadNormalizedResolutionImageCenterCrop(a_name)
            local t_img = lu:LoadNormalizedResolutionImageCenterCrop(t_name)
            assert(a_img ~= nil)
            assert(t_img ~= nil)
            batch[1][i]:copy(a_img)
            batch[2][i]:copy(t_img)
        end

        batch:cuda()

        --local f = EmbeddingNet:forward( batch[1] )
        --s = f:pow(2.0):sum(2):pow(0.5)
        --for j=1,s:size(1) do
        --    if s[j][1] ~= 1.0 then
        --        print (s)
        --    end
        --end

        local y = SiameseDistanceNet:forward( {batch[1], batch[2]} )

        for j=1,y:size(1) do
            elmt = {test_pairs[counter+j][1],test_pairs[counter+j][2],tostring(test_pairs[counter+j][3]),tostring(y[j][1])}
            print ("elmt", elmt)
            table.insert(outputs,elmt)
        end
        xlua.progress(counter, #test_pairs)

        counter = counter + batchSize
    end

    csvigo.save({path='distance_pair.csv', data=outputs})
end

CalculateDistance()
