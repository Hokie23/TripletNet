require 'cudnn'
require 'LayerDrop'

local model = torch.load("./pretrained/model_19.t7")


-- 299 x 299
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil

addLayerDrop(model, 0.9, 2048, 1024, 1, 'L1')
addLayerDrop(model, 0.8, 1024, 512, 1, 'L3')
addLayerDrop(model, 0.7, 512, 128, 1, 'L4')
addLayerDrop(model, 0.6, 128, 64, 1, 'L5')
addLayerDrop(model, 0.55, 64, 32, 1, 'L6')
addLayerDrop(model, 0.5, 32, 15, 1, 'L7')
model:add( cudnn.SpatialAveragePooling(4, 4, 2, 2, 0, 0) )   -- batchx15x8x8
model:add( nn.Reshape(3*3*15) )
model:add( nn.Normalize(2) )

return model
