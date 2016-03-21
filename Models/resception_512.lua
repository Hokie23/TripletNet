require 'cudnn'

local model = torch.load("./pretrained/model_19.t7")

-- 299 x 299
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil
model:add( nn.Linear(2048,512) )
model:add( nn.Normalize(2) )

return model
