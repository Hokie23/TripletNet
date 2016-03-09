
require 'cudnn'

local model = nn.Sequential() 

-- Convolution Layers

-- 220 x 220
model:add(cudnn.SpatialConvolution(3, 64, 5, 5, 1, 1, 2, 2 ))
model:add(cudnn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
model:add(nn.Dropout(0.25))

-- 110 x 110
model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
model:add(nn.Dropout(0.25))


-- 55 x 55
model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 0, 0))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))

-- 27 x 27
model:add(nn.Dropout(0.25))
model:add(cudnn.SpatialConvolution(256, 128, 2, 2, 1, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(nn.ReLU())
model:add(cudnn.SpatialAveragePooling(27, 27, 1, 1, 0, 0))
model:add(nn.ReLU())
--model:add(nn.View(27*27*128))
model:add(nn.View(128))
model:add(nn.BatchNormalization(128))
model:add(nn.Normalize(2))

return model
