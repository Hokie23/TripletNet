
require 'nn'
local HammingLoss, parent = torch.class('nn.HammingLoss', 'nn.Criterion')

function HammingLoss:__init()
    parent.__init(self)
    self.Target = torch.Tensor()
end


function HammingLoss:updateOutput(input, target)
    local mse_ap = torch.eq(input[1] - input[2])
    local mse_an = torch.abs(input[1]-input[3])
    self.output =
    -- self.output = self.MSE:updateOutput(input,self.Target)
    return self.output
end

function HammingLoss:updateGradInput(input, target)
    if not self.Target:isSameSizeAs(input) then
        self:createTarget(input, target)
    end

    self.gradInput = self.SoftMax:updateGradInput(input, self.MSE:updateGradInput(self.SoftMax.output,self.Target))
    -- self.gradInput  = self.MSE:updateGradInput(input,self.Target)
    return self.gradInput
end

function HammingLoss:type(t)
    parent.type(self, t)
    self.MSE:type(t)
    self.Target = self.Target:type(t)
    return self
end
