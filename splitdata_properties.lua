require "csvigo"
require 'image'
require 'math'
require 'preprocess'

local opt = opt or {}
local DataPath = opt.datapath or '/data1/fantajeon/torch/TripletNet/properties/'
local imagePath = opt.imagepath or '/data2/freebee/Images/'

local properties_filename = 'excel_1459334593242.shoes.csv'
local properties_train_filename = DataPath .. 'excel_1459334593242.shoes.train.csv'
local properties_valid_filename = DataPath .. 'excel_1459334593242.shoes.valid.csv'
local properties_test_filename = DataPath .. 'excel_1459334593242.shoes.test.csv'

function isColorImage(img)
    if img == nil then
        return false
    end
    return img:size(1) == 3 and img:dim() == 3
end

function LoadNormalizedResolutionImage(filename, jitter)
    local imagepath = imagePath .. filename
    if jitter == nil then
        return preprocess(imagepath)
    else
        return preprocess_with_jitter(imagepath, jitter)
    end
end

function LoadData(filepath)
    local Data = {anchor_name_list={}}
    local anchor_name_list = {}
    local ImagePoolByName = {}

    label_pairs = csvigo.load( {path=DataPath .. filepath, mode='large'} )
    for i=2,#label_pairs do
    --for i=1,#label_pairs,1000 do
        m = label_pairs[i]
        local a_name = m[2]
        if a_name ~= nil then
            if ImagePoolByName[a_name] == nil then
                local img = LoadNormalizedResolutionImage(a_name)
                if isColorImage(img) then
                    ImagePoolByName[a_name] = true
                end
            end

            table.insert(anchor_name_list, m)
        end
        print (i, #label_pairs, 100.0*(i/#label_pairs), "anchor", #anchor_name_list)
    end

    print("loaded: " .. #anchor_name_list)
    return anchor_name_list 
end

function SplitData(Data)
    local TrainData = {}
    local TestData = {}
    local ValidData = {}

    local train_anchor_name_list = {}
    local test_anchor_name_list = {}
    local valid_anchor_name_list = {}
    local positive_cnt = 0

    -- split positive
    for i=1, #Data do
        local fields = Data[i]
        if positive_cnt == 9 then
            positive_cnt = 10
            table.insert(ValidData, fields)
        elseif positive_cnt == 10 then
            positive_cnt = 0
            table.insert(TestData, fields)
        else
            positive_cnt = positive_cnt + 1
            table.insert(TrainData, fields)
        end
    end

    print ( string.format("split data=T(%d),V:(%d),Test=%d", #TrainData, #ValidData, #TestData) )

    return TrainData, ValidData, TestData
end


function write_csv(Data, path)
    csvigo.save({path=path,  data=Data})
end



Data = LoadData(properties_filename)
TrainData, ValidData, TestData = SplitData(Data)

write_csv(TrainData, properties_train_filename)
write_csv(ValidData, properties_valid_filename)
write_csv(TestData, properties_test_filename)

