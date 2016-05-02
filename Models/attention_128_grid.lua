require 'cudnn'
require 'AttentionLSTM'

local model = torch.load("./pretrained/model_19.t7")


-- 299 x 299
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil

addAttentionLSTM(model, 0.9, 2048, 128, 128, 'L1')
model:add( nn.ReLU(true) )
model:add( nn.Normalize(2) )

return model
