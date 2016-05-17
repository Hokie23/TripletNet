require 'DataContainer'
require 'TripletNet2'
require 'TripletNet'
require 'cutorch'
require 'eladtools'
require 'optim'
require 'xlua'
require 'trepl'
require 'DistanceRatioCriterion'
require 'DistanceRatioSoftMaxCriterion'
require 'DistancePseudoRatioCriterion'
require 'PairwiseDistanceOffset'
require 'TripletEmbeddingCriterion'
require 'cunn'
require 'WorkerParam'
require 'cudnn'
metrics = require 'metrics'
require 'eval'
--local async = require('async')

--async.repl()

cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = true


istest = true

local debugger = require( 'fb.debugger' )

----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a Triplet network on Fashion Database')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Network Port')
cmd:option('-port', 8083, 'REPL Listen port')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
-- cmd:option('-network',            'Model.lua',            'embedding network file - must return valid network.')
-- cmd:option('-network',            'resception_512.lua',            'embedding network file - must return valid network.')
--cmd:option('-network',            'resception_128_relu.lua',            'embedding network file - must return valid network.')
--cmd:option('-network',            'resception_135_grid.lua',            'embedding network file - must return valid network.')
--cmd:option('-network',            'layerdrop_135_grid.lua',            'embedding network file - must return valid network.')
--cmd:option('-network',            'attention_128_offset.lua',            'embedding network file - must return valid network.')
cmd:option('-network',            'attention_128_grid.lua',            'embedding network file - must return valid network.')
cmd:option('-LR',                 0.001,                    'learning rate')
cmd:option('-LRDecay',            1e-6,                   'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.95,                    'momentum')
cmd:option('-distance_ratio',           1.0,                    'distance ratio')
cmd:option('-max_distance_ratio',           1.0,                    'distance ratio')
cmd:option('-distance_increment',           0.02,                    'distance increment')
--cmd:option('-distance_ratio',           0.05,                    'distance ratio')
--cmd:option('-max_distance_ratio',           1.0,                    'distance ratio')
--cmd:option('-distance_increment',           0.001,                    'distance increment')
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
--cmd:option('-normalize',          1,                      '1 - normalize using only 1 mean and std values')
cmd:option('-whiten',             false,                  'whiten data')
cmd:option('-augment',            true,                  'Augment training data')
cmd:option('-preProcDir',         './PreProcData/',       'Data for pre-processing (means,P,invP)')

cmd:text('===>Misc')
cmd:option('-visualize',          true,                  'display first level filters after each epoch')


opt = cmd:parse(arg or {})
distance_ratio = opt.distance_ratio
max_distance_ratio = opt.max_distance_ratio
distance_increment = opt.distance_increment
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)

print ("REPL port", opt.port)
--async.repl.listen( {host='0.0.0.0', port=opt.port} )


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
--local EmbeddingWeights, EmbeddingGradients = EmbeddingNet:getParameters()
EmbeddingNet:cuda()
--local TripletNet = nn.TripletNet2(EmbeddingNet)
--local TripletNet = nn.TripletNet(EmbeddingNet,nn.PairwiseDistanceOffset(2) )
local TripletNet = nn.TripletNet(EmbeddingNet)
--local Loss = nn.DistanceRatioCriterion()

--local Loss = nn.DistanceRatioCriterion(distance_ratio)
--local ErrorLoss = nn.DistanceRatioCriterion(distance_ratio)
--local Loss = nn.DistanceRatioSoftMaxCriterion()
--local ErrorLoss = nn.DistanceRatioSoftMaxCriterion()
--local Loss = nn.TripletEmbeddingCriterion(0.2)
local Loss = nn.DistancePseudoRatioCriterion(distance_ratio,1, 0)
local ErrorLoss = nn.DistancePseudoRatioCriterion(distance_ratio, 1, 0)

local Weights, Gradients = TripletNet:getParameters()
TripletNet:cuda()
Loss:cuda()
ErrorLoss:cuda()


first_weight = {Weights[1]}


--debugger.enter()
--if Weights[1] ~= EmbeddingWeights[1] then
--    error('miss matched')
--end
--if Weights[2] ~= EmbeddingWeights[2] then
--    error('miss matched')
--end

if paths.filep(opt.load) then
    local w = torch.load(opt.load)
    print('Loaded', opt.load)
    Weights:copy(w)
end

--TripletNet:RebuildNet() --if using TripletNet instead of TripletNetBatch

local data = require 'TripleData'
--local data = require 'TripleDataWithProperties'
local SizeTrain = opt.size or 640000
--local SizeTest = SizeTrain*0.1
local SizeTest = 6400 
--local SizeTest = 64


------------------------- Output files configuration -----------------
if istest ~= false then
    os.execute('mkdir -p ' .. opt.save)
    os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)
    cmd:log(opt.save .. '/Log.txt', opt)
end

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
    isend = false,
    total_size = #data.TrainData.data.anchor_name_list,
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
        end

        --return SelectListTriplets(net,data.TrainData,SizeTrain, 'torch.CudaTensor', TrainSampleStage)
        return SelectListTripletsSimple(data.TrainData,SizeTrain, 'torch.CudaTensor', TrainSampleStage)
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
    LoadImageFunc = LoadNormalizedResolutionImageCenterCrop,
    Resolution = {3, 299, 299}
}


local function ErrorCount(y)
    --if torch.type(y) == 'table' then
        --y = y[#y]
    --end
    --return (y[{{},2}]:ge(y[{{},1}]):sum())
    -- y[{{},1}] = negative distance
    -- y[{{},2}] = positive distance
--    return (y[{{},2}]:ge(y[{{},1}]):mean())
   -- loss = Loss:forward(y)
    --local neg_loss = -(y[1]-1.0)
    --local loss = y[2]:mean() + neg_loss:mean()
    --return loss*0.5

    return ErrorLoss:forward(y,1)

end

local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}

function hookfunction(y,yt,err)
    print ("Hook function: err", err)
end

local optimizer = Optimizer{
    Model = TripletNet,
    Loss = Loss,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    --HookFunction = hookfunction,
    Parameters = {Weights, Gradients},
}

-------------
local threads = require 'threads'
local nthread = 1
local thread_pool = threads.Threads( nthread, function(idx)
                    require 'DataContainer'
                    require 'nn'
                    require 'torch'
                    require 'threads'
                    require 'cudnn'
                    cudnn.benchmark = true
                    cudnn.fastest = true
                    cudnn.verbose = true
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

print ("-------261") 
local bestTrainErr = 999999
local baselineTrainErr = 0
local bestAP = 0


function Train(DataC, epoch)
    print ("RunTrain")
    local err = 0
    local num = 0
    local nepoch =1

    print ("Train epoch:", epoch)
    thread_pool:synchronize()

    ShuffleTrain(data.TrainData, TrainSampleStage)
    DataC:Reset()
    TrainSampleStage.isend = false
    while TrainSampleStage.isend == false do
        collectgarbage()
        TripletNet:evaluate()
        EmbeddingNet:evaluate()
        DataC:GenerateList(EmbeddingNet)
        EmbeddingNet:training()
        TripletNet:training()

        while true do
            collectgarbage()
            local mylist = DataC:GetNextBatch()
            if mylist == nil then
                break
            end
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
                                    print ("image load error", filename, "jitter", jitter)
                                    if jitter ~= nil then
                                        print ('jitter, w1', jitter.w1, 'h1', jitter.h1, 'bFlip', jitter.bFlip, "aspect_ratio", jitter.aspect_ratio)
                                    end
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
                        --debugger.enter()
                        local y = optimizer:optimize({x[1],x[2],x[3]}, 1)
                        --local y2 = EmbeddingNet:forward( x[1] )
                        --local d = (y2 - y[2]):abs():max()
                        --print ("d", d)
                        --debugger.enter()
                        -- local y = optimizer:optimize({x[1],x[2],x[3]}, 1)
                        local lerr = ErrorCount(y)

                        --print("y:", y)

                        -- print( "lerr: ", lerr*100.0/y[1]:size(1) )
                        print( string.format("[epoch:%d, mdist=%f]: Train lerr: %f(%f), best=%f, baseline=%f", epoch, distance_ratio, lerr, lerr/distance_ratio, bestTrainErr, baselineTrainErr ) )

                        err = err + lerr
                        xlua.progress(TrainSampleStage.current, TrainSampleStage.total_size )
                        num = num + 1
                    end,
                    jobparam 
                )
            thread_pool:synchronize()
            --W1 = optimizer.Parameters[1]
            --G1 = optimizer.Parameters[2]
            --ew1 = W1
            --W1:copy(optimizer.Parameters[1])
            --G1:copy(optimizer.Parameters[2])
            collectgarbage()

        end
    end

    thread_pool:synchronize()

    if num == 0 then
        return 0
    end
    return (err/num)
end

print ("-----367")
function Test(DataC, epoch)
    print ("RunTest")
    thread_pool:synchronize()
    DataC:Reset()
    TripletNet:evaluate()
    local err = 0
    local num = 0
    local conf = {}
    local label = {}
    while true do
        collectgarbage()
        local mylist = DataC:GetNextBatch()
        if mylist == nil then
            break
        end
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

                    --debugger.enter()
                    for xx=1,y:size(1) do
                        table.insert(conf, y[xx][1])
                        table.insert(label, -1)
                        table.insert(conf, y[xx][2])
                        table.insert(label, 1)
                    end
                    --print( "Test lerr: ", lerr*100.0/y[1]:size(1) )
                    print( string.format("[epoch:%d, mdist=%f]: Test lerr: %f(%f), baselineTrainErr=%f", epoch, distance_ratio, lerr, lerr/distance_ratio, baselineTrainErr ) )
                    err = err + lerr
                    xlua.progress(num*DataC.BatchSize, DataC:size())
                    num = num +1
                end,
                jobparam
            )
        thread_pool:synchronize()
    end
    thread_pool:synchronize()
    if num == 0 then
        return 0, 0, 0, 0
    end

    --debugger.enter()
    local confTensor = torch.Tensor(conf):type('torch.DoubleTensor')
    local labelTensor = torch.Tensor(label):type('torch.DoubleTensor')
    local rec, prec, ap, threshold1 = precision_recall(confTensor, labelTensor)
    return (err/num), rec, prec, ap
end


print ("-----436")
local bestErr = 10000
local epoch = 1
local subepoch = 1
local baselineTrainErrList = {}
local baselineTrainDelta = 0
print '\n==> Starting Training\n'
while epoch ~= opt.epoch do
    print('Epoch ' .. epoch)
    local ErrTrain = Train(TrainDataContainer, epoch)
    collectgarbage()

    EmbeddingNet:clearState()
    TripletNet:clearState()

    local lightmodel = TripletNet.nets[1]:clone('weight', 'bias', 'running_mean', 'running_std', 'running_var')
    local tw, tgradp = TripletNet:parameters()

    --optimizer.Parameters = {tw, tgradp},

    torch.save(network_filename .. 'tripletnet.t7' .. epoch, lightmodel)
    --torch.save(weights_filename .. 'optim.w.t7' .. epoch, optimizer.Parameters[1])
    --torch.save(weights_filename .. epoch, tw)
    --torch.save(weights_filename .. 'tripletnet.t7' .. epoch, TripletNet)
    print( string.format('[epoch #%d:%f]:%s Training Error = %f(%f), bestTrainErr=%f, baselineTrainErr=%f', epoch, distance_ratio, opt.save, ErrTrain, ErrTrain/distance_ratio, bestTrainErr, baselineTrainErr) )

    local ErrTest, rec, prec, AP = Test(TestDataContainer, epoch)
    print( string.format('[epoch #%d:%f] Test Error = %f(%f), baselineTrainErr=%f, AP=%f, bestAP=%f', epoch, distance_ratio, ErrTest, ErrTest/distance_ratio, baselineTrainErr, AP, bestAP) )
    --if bestErr > ErrTest then
    if bestAP < AP  then
        print ("Save Best")
        bestErr = ErrTest
        bestAP = AP
        torch.save(network_filename .. 'best.embedding.model.t7', lightmodel)
        torch.save(network_filename .. 'best.tripletnet.t7', TripletNet)
        torch.save(weights_filename .. 'best.tripletnet.w.t7', tw)
        torch.save(weights_filename .. 'best.optim.w.t7', optimizer.Parameters[1])
    end

    print( string.format('[epoch #%d:%f] Test Error = %f(%f), baselineTrainErr=%f, AP=%f, bestAP=%f', epoch, distance_ratio, ErrTest, ErrTest/distance_ratio, baselineTrainErr, AP, bestAP) )
    Log:add{['Training Error']= ErrTrain* 100, ['Test Error'] = ErrTest* 100, ['Average Precision'] = AP*100}
    Log:style{['Training Error'] = '-', ['Test Error'] = '-', ['Average Precision'] = '-'}
    Log:plot()
    print ("ploted\n")

    if opt.visualize then
        require 'image'
        local weights = EmbeddingNet:get(1).weight:clone()
        --win = image.display(weights,5,nil,nil,nil,win)
        image.saveJPG(paths.concat(opt.save,'Filters_epoch'.. epoch .. '.jpg'), image.toDisplayTensor(weights))
    end


    if epoch == 1 then
        baselineTrainErr = ErrTrain
        bestTrainErr = baselineTrainErr
        baselineTrainStd = 1
    else
        baselineTrainDelta = ErrTrain - baselineTrainErr
        baselineTrainErr = baselineTrainErr + 0.95*baselineTrainDelta
        if bestTrainErr > baselineTrainErr then
            bestTrainErr = baselineTrainErr
        else
            if bestTrainErr * 1.01 >= baselineTrainErr or math.abs(baselineTrainDelta)/baselineTrainErr < 0.012 then
                distance_ratio = distance_ratio + distance_increment*(max_distance_ratio - distance_ratio)
                if distance_ratio > max_distance_ratio then
                    distance_ratio = max_distance_ratio
                end
                Loss:ResetTargetValue(distance_ratio, 1)
                ErrorLoss:ResetTargetValue(distance_ratio, 1)
                subepoch = 0
                bestTrainErr = bestTrainErr + 0.95*(baselineTrainErr - bestTrainErr)
            end
        end
    end
    epoch = epoch+1
    subepoch = subepoch + 1
end

print ("End Training\n")
