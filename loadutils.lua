require "csvigo"
require 'image'
require 'math'
require 'preprocess'

local image_utils = require 'image_utils'

local loadutils = torch.class("loadutils")

function loadutils:__init(defaultPathes)
    self.loadSize = GetLoadSize()
    self.sampleSize = GetSampleSize()
    self.defaultPathes = defaultPathes
    self.ImageResolution = self.sampleSize
end

function loadutils:SetDefaultImagePath(pathes)
    self.defaultPathes = pathes
end

function loadutils:Resolution()
    return self.ImageResolution
end

function loadutils.isColorImage(img)
    if img == nil then
        return false
    end
    return img:size(1) == 3 and img:dim() == 3
end

function loadutils:LoadNormalizedResolutionImage(filename, jitter)
    for i=1,#self.defaultPathes do
        local imagepath = self.defaultPathes[i] .. filename
        if paths.filep( imagepath ) then
            if jitter == nil then
                return preprocess(imagepath)
            else
                return preprocess_with_jitter(imagepath, jitter)
            end
        end
    end
    return nil
end

function loadutils.dist(a, b)
    local d = (a -b)*(a-b)
    return d
end

function loadutils:LoadNormalizedResolutionImageCenterCrop(filename)
    local output = nil
    for i=1,#self.defaultPathes do
        local imagepath = self.defaultPathes[i] .. filename
        if paths.filep( imagepath ) then
            local ok, input = pcall(image_utils.loadImage, imagepath, self.loadSize, 1.0)
            if ok == false then
                print ("error: 1")
                return nil
            end

            ok, output = pcall(image_utils.center_crop,input, self.sampleSize)
            if ok == false then
                print ("error: 2")
                return nil
            end

            if output == nil then
                print ("output is nil")
                return nil
            end

            output = preprocess_mean_std_norm(output)
        end
    end
    return output
end

function loadutils:CheckImage(filename)
    local ok = false
    for i=1,#self.defaultPathes do
        local imagepath = self.defaultPathes[i] .. filename
        if path.exists( imagepath ) ~= false then
            ok = pcall(image_utils.loadImage, imagepath, self.loadSize, 1.0)
        end
    end
    return ok
end


function loadutils:LoadPairs(filepath, check_imagefile)
    label_pairs = csvigo.load( {path=filepath, mode='large'} )
    local check_image = check_imagefile or false
    local output_pairs = {}
    local imagepoolbyname = {}

    for i=1,#label_pairs do
        m = label_pairs[i]
        local a_name = m[1]
        local t_name = m[2]
        local p_or_n = m[3]

        local cond_var = (function() 
            -- loop body
            if check_image then
                if imagepoolbyname[a_name] == nil then
                    if loadutils:CheckImage(a_name) == false then
                        return 
                    end
                    imagepoolbyname[a_name] = true
                end
                if imagepoolbyname[t_name] == nil then
                    if loadutils:CheckImage(t_name) == false then
                        return 
                    end
                    imagepoolbyname[t_name] = true
                end
            end

            --print ("p_or_n1=", p_or_n, p_or_n=='1')
            table.insert(output_pairs, {a_name, t_name, p_or_n == '1'}) 
            return 
        end) ()

        -- exit loop body
        if cond_var == "break" then 
            break
        end
    end

    return output_pairs
end


