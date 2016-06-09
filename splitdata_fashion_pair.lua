require "csvigo"
require 'image'
require 'math'
require 'preprocess'

local opt = opt or {}
local PreProcDir = opt.preProcDir or './'
local DataPath = opt.datapath or '/data1/fantajeon/torch/TripletNet/'
local imagePath = opt.imagepath or '/data1/october_11st/october_11st_imgs/'

local fashion_pair_filename = 'fashion_pair.csv'
local fashion_pair_train_filename = 'fashion_pair_train.csv'
local fashion_pair_valid_filename = 'fashion_pair_valid.csv'
local fashion_pair_test_filename = 'fashion_pair_test.csv'

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
    local Data = {data={},imagepool={}}
    local positive_pairs = {}
    local negative_pairs = {}
    local anchor_name_to_idx = {}
    local anchor_name_list = {}
    local anchor_count = 0
    local category_name = {}
    local ImagePool = {}
    local count_imagepool = 0
    local ImagePoolByName = {}

    label_pairs = csvigo.load( {path=DataPath .. filepath, mode='large'} )
    for i=1,#label_pairs do
    --for i=1,#label_pairs,1000 do
        m = label_pairs[i]
        local category = m[1]
        local a_name = m[2]
        local t_name = m[3]
        local p_or_n = m[4]
        local bcontinue = false

        local bstop = (function() 
            if a_name == t_name or a_name == nil then
                return "incorrect"
            end
            --print ("anchor_name=", a_name)
            if ImagePoolByName[a_name] == nil then
                local img = LoadNormalizedResolutionImage(a_name)
                if isColorImage(img) then
                    count_imagepool = count_imagepool + 1
                    ImagePoolByName[a_name] = true
                else
                    return "non-color"
                end
            end
            if ImagePoolByName[t_name] == nil then
                local img = LoadNormalizedResolutionImage(t_name)
                if isColorImage(img) then
                    count_imagepool = count_imagepool + 1
                    ImagePoolByName[t_name] = true
                else
                    return "non-color"
                end
            end

            if anchor_name_to_idx[a_name] == nil then
                table.insert(anchor_name_list, a_name)
                anchor_count = anchor_count + 1
                anchor_name_to_idx[a_name] = anchor_count
            end
            if p_or_n == '1' then
                if positive_pairs[a_name] == nil then
                    positive_pairs[a_name] = {{t_name, category_name}}
                else
                    table.insert(positive_pairs[a_name], {t_name, category_name} )
                end
            else 
                if negative_pairs[a_name] == nil then
                    negative_pairs[a_name] = {{t_name, category_name}}
                else
                    table.insert(negative_pairs[a_name], {t_name, category_name} )
                end
            end
            return "ok"
        end)()
        print (i, #label_pairs, 100.0*(i/#label_pairs), count_imagepool, "anchor", #anchor_name_list)
    end

    print("loaded: " .. #anchor_name_list)
    print("loaded imagepool: " .. #ImagePool)
    Data.data.anchor_name_list = anchor_name_list
    Data.data.positive = positive_pairs
    Data.data.negative = negative_pairs

    return Data
end

function SplitData(Data)
    local TrainData = {data={}}
    local TestData = {data={}}
    local ValidData = {data={}}

    local train_anchor_name_list = {}
    local test_anchor_name_list = {}
    local valid_anchor_name_list = {}
    local positive_cnt = 0

    -- split positive
    for i=1, #Data.data.anchor_name_list do
        local anchor_name = Data.data.anchor_name_list[i]
        local pos_of_anchor = Data.data.positive[anchor_name] 
        if pos_of_anchor ~= nil then
            if positive_cnt == 9 then
                positive_cnt = 10
                table.insert(valid_anchor_name_list, Data.data.anchor_name_list[i])
            elseif positive_cnt == 10 then
                positive_cnt = 0
                table.insert(test_anchor_name_list, Data.data.anchor_name_list[i])
            else
                positive_cnt = positive_cnt + 1
                table.insert(train_anchor_name_list, Data.data.anchor_name_list[i])
            end
        end
    end

    print ( string.format("split data=T(%d),V:(%d),Test=%d", #train_anchor_name_list, #valid_anchor_name_list, #test_anchor_name_list) )

    TrainData.data.anchor_name_list = train_anchor_name_list
    TrainData.data.positive = Data.data.positive
    TrainData.data.negative = Data.data.negative
    TrainData.data.all_negative_list = Data.NegativeList
    TrainData.Resolution = {3, 299, 299}

    ValidData.data.anchor_name_list = valid_anchor_name_list
    ValidData.data.positive = Data.data.positive
    ValidData.data.negative = Data.data.negative
    ValidData.data.all_negative_list = Data.NegativeList
    ValidData.Resolution = {3, 299, 299}

    TestData.data.anchor_name_list = test_anchor_name_list
    TestData.data.positive = Data.data.positive
    TestData.data.negative = Data.data.negative
    TestData.data.all_negative_list = Data.NegativeList
    TestData.Resolution = {3, 299, 299}

    print ("Train Data size:", #TrainData.data.anchor_name_list)
    print ("Valid Data size:", #ValidData.data.anchor_name_list)
    print ("Test Data size:", #TestData.data.anchor_name_list)

    return TrainData, ValidData, TestData
end


function write_csv(Data, path)
    local array = {}
    for i=1,#Data.data.anchor_name_list do
        local anchor_name = Data.data.anchor_name_list[i]
        local pos_of_anchor = Data.data.positive[anchor_name]
        for j=1,#pos_of_anchor do
            local pair = {pos_of_anchor[j][2], anchor_name, pos_of_anchor[j][1], 1}
            table.insert(array, pair)
        end

        local neg_of_anchor = Data.data.negative[anchor_name]
        if neg_of_anchor ~= nil then
            for j=1,#neg_of_anchor do
                local pair = {neg_of_anchor[j][2], anchor_name, neg_of_anchor[j], 0}
                table.insert(array, pair)
            end
        end
    end

    csvigo.save({path=path,  data=array})
end



Data = LoadData(fashion_pair_filename)
TrainData, ValidData, TestData = SplitData(Data)

write_csv(TrainData, fashion_pair_train_filename)
write_csv(ValidData, fashion_pair_valid_filename)
write_csv(TestData, fashion_pair_test_filename)

