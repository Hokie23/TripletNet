require 'nn'
require 'cudnn'
require 'cunn'
require 'LinearWithoutBias'
local nninit = require 'nninit'

local AttentionLSTM, parent = torch.class('nn.AttentionLSTM', 'nn.Container')

function AttentionLSTM:__init(deathRate, nChannels, nOutputChannels, nHidden, name)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.gate = true
    self.train = true
    self.deathRate = deathRate
    self.nHidden = nHidden or 128
    self.nChannels = nChannels
    self.nOutputChannels = nOutputChannels
    self.name = name

    local x = nn.Identity()()
    -- batch x nChannels x 8 x 8
    local r3 = cudnn.SpatialConvolution(nChannels, nOutputChannels, 1, 1, 1, 1)
                            :init('weight', nninit.sparse, 0.8)
                            :init('bias', nninit.constant, 0)(x)
    local x2 = nn.Reshape(nOutputChannels, 8*8)(r3)
    -- batch (32) x features (nOutputChannels) x annotations (64)
    local x3 = cudnn.SpatialConvolution(nChannels, nHidden, 1, 1, 1, 1)
                            :init('bias', nninit.constant, 0)(x)

    -- batch (32) x hiddens (128) x height (8) x width (8)
    local h = nn.Reshape(nChannels)(cudnn.SpatialAveragePooling(8, 8, 1, 1, 0, 0)(x)) 
    -- batch x nChannels x 1 x 1
    local h1 = nn.LinearWithoutBias(nChannels, nHidden)(h)
    -- batch (32) x hiddens (42)
    local h2 = nn.Replicate(8*8, 3)(h1)
    -- batch (32) x hiddens (42) x annotations (49)
    local h3 = nn.Reshape(nHidden, 8, 8)(h2)
    -- batch (32) x hiddens (42) x height (8) x width (8)

    local a1 = nn.Tanh()(nn.CAddTable()({h3, x3}))
    local a2 = cudnn.SpatialConvolution(nHidden, 1, 1, 1, 1, 1)(a1)
    -- batch (32) x softmax (1) x height (8) x width (8)
    local a3 = nn.SoftMax()(nn.Reshape(8*8)(a2))
    -- batch (32) x annotations (64)
    
    local a4 = nn.Replicate(nOutputChannels, 2)(a3)
    -- batch (32) x features (nOutputChannels) x annotations (64)

    local context = nn.Sum(3)(nn.CMulTable()({a4, x2}))
    -- batch (32) x features (nOutputChannels)

    --local g = nn.gModule({h,x}, {a3,context})
    local g = nn.gModule({x}, {context})
    self.g = g
    self.modules = {g}
end

function AttentionLSTM:updateOutput(input)
    self.output = self.g:forward(input)
    return self.output
end

function AttentionLSTM:updateGradInput(input, gradOutput)
    self.gradInput = self.g:updateGradInput(input, gradOutput)
    return self.gradInput   -- first hidden, second feature
end

function AttentionLSTM:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    self.g:accGradParameters(input, gradOutput, scale)
end

--function AttentionLSTM:parameters()
--    return self.g:parameters()
--end
--
--function AttentionLSTM:zeroGradParameters()
--    return self.g:zeroGradParameters()
--end
--

---- Adds a residual block to the passed in model ----
function addAttentionLSTM(model, deathRate, nChannels, nOutChannels, nHidden, name)
    model:add(nn.AttentionLSTM(deathRate, nChannels, nOutChannels, nHidden, name))
    return model
end
