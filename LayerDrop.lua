require 'nn'
require 'cudnn'
require 'cunn'
local nninit = require 'nninit'

local LayerDrop, parent = torch.class('nn.LayerDrop', 'nn.Container')

function LayerDrop:__init(deathRate, nChannels, nOutChannels, stride, name)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.gate = true
    self.train = true
    self.deathRate = deathRate
    nOutChannels = nOutChannels or nChannels
    stride = stride or 1

    self.name = name

    self.net = nn.Sequential()
    self.net:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 3,3, stride,stride, 1,1)
                            :init('weight', nninit.kaiming, {gain = 'relu'})
                            :init('bias', nninit.constant, 0))
    self.net:add(cudnn.SpatialBatchNormalization(nOutChannels)
                            :init('weight', nninit.normal, 1.0, 0.002)
                            :init('bias', nninit.constant, 0))
    self.net:add(cudnn.ReLU(true))
    self.net:add(cudnn.SpatialConvolution(nOutChannels, nOutChannels, 3,3, 1,1, 1,1)
                            :init('weight', nninit.kaiming, {gain = 'relu'})
                            :init('bias', nninit.constant, 0))
    self.net:add(cudnn.SpatialBatchNormalization(nOutChannels))

    self.skip = nn.Sequential()
    self.skip:add(nn.Identity())
    self.skip:add( cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, stride, stride, 0, 0)
                            :init('weight', nninit.sparse, 0.2)
                            :init('bias', nninit.constant, 0))
    self.modules = {self.net, self.skip}
end

function LayerDrop:updateOutput(input)
    self.output = self.skip:forward(input)
    --self.output:resizeAs(skip_forward):copy(skip_forward)
    self.gate = false
    if self.train then
        if torch.rand(1)[1] < self.deathRate then -- only compute convolutional output when gate is open
            self.gate = true
            self.output:add(self.net:forward(input))
        end
    else
        self.output:add(self.net:forward(input):mul(1-self.deathRate))
    end
    return self.output
end

function LayerDrop:updateGradInput(input, gradOutput)
    --self.gradInput = self.gradInput or input.new()
    --self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))
    --print ("name", self.name, "input", input:size(), "gate", self.gate)
    self.gradInput = self.skip:updateGradInput(input, gradOutput)
    if self.gate then
        self.gradInput:add(self.net:updateGradInput(input, gradOutput))
    end
    --print ("name", self.name, "self.gradInput", self.gradInput:size())
    return self.gradInput
end

function LayerDrop:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    if self.gate then
        self.net:accGradParameters(input, gradOutput, scale)
    end
end

---- Adds a residual block to the passed in model ----
function addLayerDrop(model, deathRate, nChannels, nOutChannels, stride, name)
    model:add(nn.LayerDrop(deathRate, nChannels, nOutChannels, stride, name))
    model:add(cudnn.ReLU(true))
    return model
end
