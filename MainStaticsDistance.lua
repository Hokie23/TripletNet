require 'SiameseDistanceNet'
require "csvigo"
require 'cutorch'
require 'xlua'
require 'trepl'
require 'cunn'
require 'loadutils'

----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Testing a Siamese Distance network on Fashion')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
--cmd:option('-modelsFolder',       './Results/ThuMar1712:51:352016/',            'Models Folder')
cmd:option('-modelsFolder',       './Results/TueMar1515:07:552016/',            'Models Folder')
cmd:option('-network',            'resception.lua',            'embedding network file - must return valid network.')

cmd:text('===>Platform Optimization')
cmd:option('-batchSize',          16,                    'batch size')
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
print ("load...", opt.load)
local w = torch.load( opt.load )
print ("load...", opt.network)
local EmbeddingNet = require(opt.network)

EmbeddingNet:cuda()
local EmbeddingWeights, EmbeddingGradients = EmbeddingNet:getParameters()
EmbeddingWeights:copy(w)

print ("EmbeddingNet:", EmbeddingNet)
local SiameseDistanceNet = nn.SiameseDistanceNet(EmbeddingNet)
SiameseDistanceNet:cuda()

------------------------- Output files configuration -----------------
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
----------------------------------------------------------------------

local fashion_test_pair = 'fashion_pair_test.csv'

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

        batch:resize(nsz)

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
