require 'cudnn'

local model = torch.load("./pretrained/model_19.t7")


-- 299 x 299
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil
model.modules[#model.modules] = nil
model:add( cudnn.SpatialConvolution(2048,1843, 1, 1, 1, 1, 0, 0) )
model:add( cudnn.SpatialBatchNormalization(1843) )
model:add( cudnn.ReLU() )
model:add( cudnn.SpatialConvolution(1843,1024, 1, 1, 1, 1, 0, 0) )
model:add( cudnn.SpatialBatchNormalization(1024) )
model:add( cudnn.ReLU() )
model:add( cudnn.SpatialConvolution(1024,512, 1, 1, 1, 1, 0, 0) )
model:add( cudnn.SpatialBatchNormalization(512) )
model:add( cudnn.ReLU() )
model:add( cudnn.SpatialConvolution(512,128, 1, 1, 1, 1, 0, 0) )
model:add( cudnn.SpatialBatchNormalization(128) )
model:add( cudnn.ReLU() )
model:add( cudnn.SpatialConvolution(128,64, 1, 1, 1, 1, 0, 0) )
model:add( cudnn.SpatialBatchNormalization(64) )
model:add( cudnn.ReLU() )
model:add( cudnn.SpatialConvolution(64,32, 1, 1, 1, 1, 0, 0) )
model:add( cudnn.SpatialBatchNormalization(32) )
model:add( cudnn.ReLU() )
model:add( cudnn.SpatialConvolution(32,15, 1, 1, 1, 1, 0, 0) ) -- batchx32x8x8
model:add( cudnn.SpatialBatchNormalization(15) )
model:add( cudnn.ReLU() )
model:add( cudnn.SpatialAveragePooling(4, 4, 2, 2, 0, 0) )   -- batchx15x8x8
--model:add( nn.View(3*3*15,135) )   -- batchx15x8x8
model:add( nn.Reshape(3*3*15) )
model:add( nn.Normalize(2) )

return model
