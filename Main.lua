require 'DataContainer'
require 'TripletNet2'
require 'TripletNet'
require 'cutorch'
require 'eladtools'
require 'optim'
require 'xlua'
require 'trepl'
require 'DistanceRatioCriterion'
require 'TripletEmbeddingCriterion'
require 'cunn'
require 'WorkerParam'

----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a Triplet network on CIFAR 10/100')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
-- cmd:option('-network',            'Model.lua',            'embedding network file - must return valid network.')
-- cmd:option('-network',            'resception_512.lua',            'embedding network file - must return valid network.')
--cmd:option('-network',            'resception_128_relu.lua',            'embedding network file - must return valid network.')
cmd:option('-network',            'resception_135_grid.lua',            'embedding network file - must return valid network.')
cmd:option('-LR',                 0.001,                    'learning rate')
cmd:option('-LRDecay',            1e-6,                   'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.95,                    'momentum')
-- cmd:option('-batchSize',          128,                    'batch size')
-- cmd:option('-batchSize',          1,                    'batch size')
--cmd:option('-batchSize',          8,                    'batch size')
cmd:option('-batchSize',          6,                    'batch size')
--cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-optimization',       'adadelta',                  'optimization method')
cmd:option('-epoch',              -1,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'float or cuda')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Data Options')
cmd:option('-dataset',            'fashion',              'Dataset - Cifar10 or Cifar100')
--cmd:option('-size',               640000,                 'size of training list' )
--cmd:option('-size',               640,                 'size of training list' )
--cmd:option('-size',               180,                 'size of training list' )
cmd:option('-size',               12,                 'size of training list' )
--cmd:option('-size',               64,                 'size of training list' )
--cmd:option('-size',               640,                 'size of training list' )
--cmd:option('-size',               64000,                 'size of training list' )
cmd:option('-normalize',          1,                      '1 - normalize using only 1 mean and std values')
cmd:option('-whiten',             false,                  'whiten data')
cmd:option('-augment',            true,                  'Augment training data')
cmd:option('-preProcDir',         './PreProcData/',       'Data for pre-processing (means,P,invP)')

cmd:text('===>Misc')
cmd:option('-visualize',          true,                  'display first level filters after each epoch')


opt = cmd:parse(arg or {})
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)

opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.save = paths.concat('./Results', opt.save)
opt.preProcDir = paths.concat(opt.preProcDir, opt.dataset .. '/')

print( string.format('preProcDir: %s', opt.preProcDir) )
os.execute('mkdir -p ' .. opt.preProcDir)
if opt.augment then
    require 'image'
end


----------------------------------------------------------------------
-- Model + Loss:
local EmbeddingNet = require(opt.network)
--local TripletNet = nn.TripletNet2(EmbeddingNet)
local TripletNet = nn.TripletNet(EmbeddingNet)
local Loss = nn.DistanceRatioCriterion()
--local Loss = nn.TripletEmbeddingCriterion(0.2)
TripletNet:cuda()
Loss:cuda()


local Weights, Gradients = TripletNet:getParameters()
local EmbeddingWeights, EmbeddingGradients = EmbeddingNet:getParameters()

if paths.filep(opt.load) then
    local w = torch.load(opt.load)
    print('Loaded', opt.load)
    Weights:copy(w)
end

--TripletNet:RebuildNet() --if using TripletNet instead of TripletNetBatch

local data = require 'TripleData'
local SizeTrain = opt.size or 640000
--local SizeTest = SizeTrain*0.1
local SizeTest = 6400 
--local SizeTest = 64


------------------------- Output files configuration -----------------
os.execute('mkdir -p ' .. opt.save)
os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local network_filename = paths.concat(opt.save, "Embedding.t7")
local weights_filename = paths.concat(opt.save, 'Weights.t7')
local log_filename = paths.concat(opt.save,'ErrorProgress')
local Log = optim.Logger(log_filename)
Log.showPlot = false


----------------------------------------------------------------------
print '==> Embedding Network'
print(EmbeddingNet)
print '==> Triplet Network'
print(TripletNet)
print '==> Loss'
print(Loss)

TrainSampleStage = {
    current= 1
}

---------------------------------------------------------------------
function ReGenerateTrain(net, selection_mode)
    if selection_mode then
        return GenerateListTriplets(data.TrainData,SizeTrain, "train")
    else
        --return SelectListTriplets(net,data.TrainData,SizeTrain, 'torch.FloatTensor')
        if TrainSampleStage.isend then
            data.TrainData.IsEnd = true
            ShuffleTrain(data.TrainData, TrainSampleStage)
        end

        return SelectListTriplets(net,data.TrainData,SizeTrain, 'torch.CudaTensor', TrainSampleStage)
    end
end

local TestList = GenerateListTriplets(data.TestData,SizeTest, "test")

local TrainDataContainer = DataContainer{
    Data = data.TrainData.data,
    TensorType = 'torch.CudaTensor',
    BatchSize = opt.batchSize,
    Augment = opt.augment,
    ListGenFunc = ReGenerateTrain,
    BulkImage = false,
    NumEachSet = 3,
    LoadImageFunc = LoadNormalizedResolutionImage,
    Resolution = {3, 299, 299}
}

local TestDataContainer = DataContainer{
    Data = data.TestData.data,
    List = TestList,
    TensorType = 'torch.CudaTensor',
    BatchSize = opt.batchSize,
    BulkImage = false,
    NumEachSet = 3,
    LoadImageFunc = LoadNormalizedResolutionImage,
    Resolution = {3, 299, 299}
}


local function ErrorCount(y)
    loss = Loss:forward(y)
    return loss
end

local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}


local optimizer = Optimizer{
    Model = TripletNet,
    Loss = Loss,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    Parameters = {Weights, Gradients},
}

-------------
local threads = require 'threads'
local nthread =1
local thread_pool = threads.Threads( nthread, function(idx)
                    require 'DataContainer'
                    require 'nn'
                    require 'torch'
                    require 'threads'
                    require 'cudnn'
                    require 'WorkerParam'
                    require 'preprocess'
                    require 'image'
                    require 'math'
                    require 'preprocess'
                    require 'loadutils'

                    print ("thread init:" .. idx)
                end,
                function(idx)
                    print ("set data: " .. idx)
                end)
                
function Train(DataC, epoch)
    print ("RunTrain")
    local err = 0
    local num = 0
    local nepoch =1

    print ("Train epoch:", epoch)
    thread_pool:synchronize()
    DataC:Reset()
    collectgarbage()

    while DataC:IsContinue() do
        TripletNet:evaluate()
        EmbeddingNet:evaluate()
        DataC:GenerateList(EmbeddingNet, true)
        EmbeddingNet:training()
        TripletNet:training()
        local mylist = DataC:GetNextBatch()
        --print ("LoadImageFunc:", DataC.LoadImageFunc)
        local jobparam = WorkerParam(mylist, DataC.TensorType, DataC.Resolution, DataC.LoadImageFunc, DataC.NumEachSet)
        -- print ("train #", x)
        thread_pool:addjob(
                function (param)
                    local function catnumsize(num,size)
                        local stg = torch.LongStorage(#size+1)
                        stg[1] = num
                        for i=2,stg:size() do
                            stg[i]=size[i-1]
                        end
                        return stg
                    end
                    --print ( string.format("%x, getnextbatch()", __threadid) )
                    local batchlist = param.BatchList
                    local batch = torch.Tensor():type(param.TensorType)
                    local size = #batchlist
                    nsz = catnumsize(param.NumEachSet, catnumsize(size,  param.Resolution))
                    batch:resize(nsz)

                    for j=1,#batchlist do
                        for i=1, param.NumEachSet do
                            local filename = batchlist[j].names[i]
                            local jitter = batchlist[j].jitter[i]
                            local ok, img = pcall(param.LoadImageFunc,filename, jitter)
                            assert(img ~= nil)
                            if ok == false then
                                print ('jitter, w1', jitter.w1, 'h1', jitter.h1, 'bFlip', jitter.bFlip, "aspect_ratio", jitter.aspect_ratio)
                                print ("image load error", filename, "jitter", jitter)
                                print ("error:", img)
                                return nil
                            end

                            batch[i][j]:copy(img)
                        end
                    end

                    return batch
                end,
                function (x)
                    if x == nil then
                        print ('x is nil')
                        return
                    end

                    --local y = optimizer:optimize({x[1],x[2],x[3]})
                    local y = optimizer:optimize({x[1],x[2],x[3]}, 1)
                    -- local y = optimizer:optimize({x[1],x[2],x[3]}, 1)
                    local lerr = ErrorCount(y)

                    --print("y:", y)

                    -- print( "lerr: ", lerr*100.0/y[1]:size(1) )
                    print( string.format("[epoch #%d]: Train lerr: %e", epoch, lerr ) )

                    err = err + lerr
                    xlua.progress(num*DataC.BatchSize, DataC:size()*nepoch)
                    num = num + 1
                end,
                jobparam 
            )
    end

    thread_pool:synchronize()

    if num == 0 then
        return 0
    end
    return (err/num)
end

function Test(DataC, epoch)
    print ("RunTest")
    thread_pool:synchronize()
    DataC:Reset()
    TripletNet:evaluate()
    local err = 0
    local num = 0
    while DataC:IsContinue() do
        local mylist = DataC:GetNextBatch()
        local jobparam = WorkerParam(mylist, DataC.TensorType, DataC.Resolution, DataC.LoadImageFunc, DataC.NumEachSet)
        thread_pool:addjob(
                function (param)
                    local function catnumsize(num,size)
                        local stg = torch.LongStorage(#size+1)
                        stg[1] = num
                        for i=2,stg:size() do
                            stg[i]=size[i-1]
                        end
                        return stg
                    end
                    --print ( string.format("%x, getnextbatch()", __threadid) )
                    local batchlist = param.BatchList
                    local batch = torch.Tensor():type(param.TensorType)
                    local size = #batchlist
                    nsz = catnumsize(param.NumEachSet, catnumsize(size,  param.Resolution))
                    batch:resize(nsz)

                    --print ("batch:resize:", nsz)
                    for j=1,#batchlist do
                        for i=1, param.NumEachSet do
                            local filename = batchlist[j].names[i]
                            local jitter = batchlist[j].jitter[i]
                            --print (filename, jitter)
                            local img = param.LoadImageFunc(filename, jitter)
                            batch[i][j]:copy(img)
                        end
                    end
                    return batch
                end,
                function (x)
                    if x == nil then
                        return
                    end
                    --local y = TripletNet:forward({x[1],x[2],x[3]})
                    local y = TripletNet:forward({x[1],x[2],x[3]})
                    local lerr = ErrorCount(y)
                    --print( "Test lerr: ", lerr*100.0/y[1]:size(1) )
                    print( string.format("[epoch:%d]: Test lerr: %e", epoch, lerr ) )
                    err = err + lerr
                    xlua.progress(num*DataC.BatchSize, DataC:size())
                    num = num +1
                end,
                jobparam
            )
    end
    thread_pool:synchronize()
    if num == 0 then
        return 0
    end
    return (err/num)
end


local bestErr = 10000
local epoch = 1
print '\n==> Starting Training\n'
while epoch ~= opt.epoch do
    print('Epoch ' .. epoch)
    local ErrTrain = Train(TrainDataContainer, epoch)
    torch.save(network_filename .. epoch, EmbeddingNet)
    torch.save(weights_filename .. epoch, Weights)
    print( string.format('[epoch #%d] Training Error = %f', epoch,  ErrTrain) )
    local ErrTest = Test(TestDataContainer, epoch)
    if bestErr > ErrTest then
        bestErr = ErrTest
        torch.save(network_filename, EmbeddingNet)
        torch.save(weights_filename, Weights)
        torch.save(weights_filename .. 'embedding.t7', EmbeddingWeights)
    end

    print( string.format('[epoch #%d] Test Error = %f', epoch, ErrTest) )
    Log:add{['Training Error']= ErrTrain* 100, ['Test Error'] = ErrTest* 100}
    Log:style{['Training Error'] = '-', ['Test Error'] = '-'}
    Log:plot()
    print ("ploted\n")

    if opt.visualize then
        require 'image'
        local weights = EmbeddingNet:get(1).weight:clone()
        --win = image.display(weights,5,nil,nil,nil,win)
        image.saveJPG(paths.concat(opt.save,'Filters_epoch'.. epoch .. '.jpg'), image.toDisplayTensor(weights))
    end

    epoch = epoch+1
end

print ("End Training\n")
