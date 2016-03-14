
local image_utils = require 'image_utils'

local cnn_model_mean =
  -- for inception-v3-2015-12-05, resception
  torch.FloatTensor{0.4853717905167, 0.45622173301884, 0.4061366788954}
local cnn_model_std =
  -- for inception-v3-2015-12-05, resception
  torch.FloatTensor{0.22682182875849, 0.22206057852892, 0.22145828935297}

local loadSize = {3, 342, 342}
local sampleSize = {3, 299, 299}

function preprocess(image_file_path)
    local ok, input = pcall(image_utils.loadImage,image_file_path, loadSize)
    if ok == false then
        return nil
    end
    local output = image_utils.random_jitter(input, sampleSize)
    output = image_utils.mean_std_norm(output, cnn_model_mean, cnn_model_std)
    return output
end

