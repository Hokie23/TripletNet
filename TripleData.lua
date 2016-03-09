require "csvigo"
require 'image'
require 'math'
require 'preprocess'

local opt = opt or {}
local PreProcDir = opt.preProcDir or './'
local Whiten = opt.whiten or false
local DataPath = opt.datapath or '/data1/fantajeon/torch/TripletNet/'
local SimpleNormalization = (opt.normalize==1) or false
local imagePath = opt.imagepath or '/data1/october_11st/october_11st_imgs/'

local TestData
local TrainData
local Classes
local ImagePool = {}
local shortestAxisNormalLength = 242

function isColorChannel(db, imagename)
    local d = db.imagepool[db.imagepoolbyname[imagename]]
    return d:size(1) == 3 and d:dim() == 3
end

function isColorImage(img)
    if img == nil then
        return false
    end
    return img:size(1) == 3 and img:dim() == 3
end

function GenerateListTriplets(db, size)
    print ("generate list triplets", size)
    local data = db.data
    local list = torch.IntTensor(size,3)
    -- print ("generate db: ", db)
    -- print ("Generate List Triplet data:", db)
    -- print ("anchor_name_list:", data.anchor_name_list)
    local nClasses = #data.anchor_name_list 
    for i=1, size do
        --print ("generate list #" .. i .. "/#" .. size)
        local c1, anchor_name, positive_name, negative_name
        while true do
            c1 = math.random(nClasses)
            anchor_name = data.anchor_name_list[c1]

            pos_of_anchor = data.positive[anchor_name]
            if pos_of_anchor ~= nil then
                --print ("anchor_name", anchor_name)
                -- print ("pos_list_anchor_name", data.positive)
                --print ("pos_of_anchor", pos_of_anchor)
                positive_name = pos_of_anchor[math.random(#pos_of_anchor)]
                break
            end
        end

        local neg_of_anchor = data.negative[anchor_name]
        if neg_of_anchor == nil or math.random(2) == 1 then
            local n1 = c1
            local n3 = math.random(nClasses)
            while n3 == n1 do
                n3 = math.random(nClasses)
            end

            negative_name = data.anchor_name_list[n3]
        else
            local n3 = math.random(#neg_of_anchor)
            negative_name = neg_of_anchor[n3]
        end


        list[i][1] = db.imagepoolbyname[anchor_name]
        list[i][3] = db.imagepoolbyname[positive_name]
        list[i][2] = db.imagepoolbyname[negative_name]

        if list[i][1] == list[i][3] then
            print("same index", positive_name, anchor_name)
        end
        --print("list1", list[i][1])
        --print("list2", list[i][2])
        --print("list3", list[i][3])
        --print ("generated list:", list)
    end
    return list
end


function hash_c(c1,c2)
    return (c1*10 + c2 -10)
end

function CreateDistanceTensor(data,labels, model)
    local Rep = ForwardModel(model,data)
    local Dist = torch.ByteTensor(data:size(1),data:size(1)):zero()
    for i=1,data:size(1) do
        for j=i+1,data:size(1) do
            Dist[i][j] = math.ceil(torch.dist(Rep[i],Rep[j]))
        end
    end
    return Dist
end


save_filename = PreProcDir .. '/fashion_data.t7' 
print ("check chached file:" .. save_filename)
if path.exists(save_filename) ~= false then
    print ("cached model")
    Data = torch.load(save_filename )
    --print ("Data:", Data.TrainData, Data.TestData)
    return Data
end

function LoadNormalizedResolutionImage(filename)
    local imagepath = imagePath .. filename
    return preprocess(imagepath)
end
--function LoadNormalizedResolutionImage(filename)
--    local imagepath = imagePath .. filename
--
--    print ('loading:' .. imagepath)
--    im = image.load(imagepath)
--    if im == nil then
--        print ("failed to load:" .. imagepath)
--    end
--    --print ("src_width: " .. im:size()[2] )
--    --print ("src_height:" .. im:size()[3] )
--    height = im:size()[2]
--    width = im:size()[3]
--    shortest = math.min(width,height)
--
--    r = shortestAxisNormalLength/shortest
--    d_width = math.ceil(width*r)
--    d_height = math.ceil(height*r)
--    --print ("r:" .. r)
--    im = image.scale(im,d_width,d_height)
--
--    --print ("t_width: " .. im:size()[2] )
--    --print ("t_height:" .. im:size()[3] )
--    im = image.crop(im, 'c', shortestAxisNormalLength, shortestAxisNormalLength)
--    im = im:add(-128):div(128)
--    return im
--end

function countTableSize(table)
    local n = 0
    for k, v in pairs(table) do
        n = n + 1
    end
    return n
end

function LoadData(filepath)
    local Data = {data={},imagepool={}}
    local positive_pairs = {}
    local negative_pairs = {}
    local anchor_name_to_idx = {}
    local anchor_name_list = {}
    local anchor_count = 0
    local ImagePool = {}
    local count_imagepool = 0
    local ImagePoolByName = {}

    label_pairs = csvigo.load( {path=DataPath .. filepath, mode='large'} )

    for i=1,#label_pairs,3 do
    --for i=1,#label_pairs,10000 do
        m = label_pairs[i]
        local a_name = m[2]
        local t_name = m[3]
        local p_or_n = m[4]
        local bcontinue = false

        bcontinue = false

        if a_name ~= t_name then
            --print ("anchor_name=", a_name)
            if ImagePoolByName[a_name] == nil then
                local img = LoadNormalizedResolutionImage(a_name)
                if isColorImage(img) then
                    table.insert(ImagePool,img)
                    count_imagepool = count_imagepool + 1
                    ImagePoolByName[a_name] = count_imagepool
                else
                    --print("ancnor continue")
                    bcontinue = true
                end
            end
            if ImagePoolByName[t_name] == nil then
                local img = LoadNormalizedResolutionImage(t_name)
                if isColorImage(img) then
                    table.insert(ImagePool,img)
                    count_imagepool = count_imagepool + 1
                    ImagePoolByName[t_name] = count_imagepool
                else
                    --print("positive continue")
                    bcontinue = true
                end
            end

            if bcontinue == false then
                if anchor_name_to_idx[a_name] == nil then
                    table.insert(anchor_name_list, a_name)
                    anchor_count = anchor_count + 1
                    anchor_name_to_idx[a_name] = anchor_count
                end
                if p_or_n == '1' then
                    if positive_pairs[a_name] == nil then
                        positive_pairs[a_name] = {t_name}
                    else
                        table.insert(positive_pairs[a_name], t_name )
                    end
                else 
                    if negative_pairs[a_name] == nil then
                        negative_pairs[a_name] = {t_name}
                    else
                        table.insert(negative_pairs[a_name], t_name )
                    end
                end
                --print ("positive_pair", #positive_pairs)
                --print ("negative_pair", #negative_pairs)
            else
                print ("continue")
            end
        end
        print (i, #label_pairs, 100.0*(i/#label_pairs), count_imagepool)
    end

    print("loaded: " .. #anchor_name_list)
    print("loaded imagepool: " .. #ImagePool)
    Data.data.anchor_name_to_idx = anchor_name_to_idx
    Data.data.anchor_name_list = anchor_name_list
    Data.data.positive = positive_pairs
    Data.data.negative = negative_pairs
    Data.imagepool = ImagePool
    Data.imagepoolbyname = ImagePoolByName

    return Data
end

local Data = LoadData('fashion_pair.csv')

function SplitData(Data)
    local TrainData = {data={},}
    local TestData = {data={}}

    local train_anchor_name_list = {}
    local test_anchor_name_list = {}
    local positive_cnt = 0
    local negative_cnt = 0

    -- split positive
    for i=1, #Data.data.anchor_name_list do
        local anchor_name = Data.data.anchor_name_list[i]
        if Data.data.positive[anchor_name] ~= nil then
            if positive_cnt >= 10 then
                postivie_cnt = 0
                table.insert(test_anchor_name_list, Data.data.anchor_name_list[i])
            else
                positive_cnt = positive_cnt + 1
                table.insert(train_anchor_name_list, Data.data.anchor_name_list[i])
            end
        else
            positive_cnt = positive_cnt + 1
            table.insert(train_anchor_name_list, Data.data.anchor_name_list[i])
        end
    end

    TrainData.data.anchor_name_to_idx = Data.data.anchor_name_to_idx
    TrainData.data.anchor_name_list = train_anchor_name_list
    TrainData.data.positive = Data.data.positive
    TrainData.data.negative = Data.data.negative
    TrainData.imagepool = Data.imagepool
    TrainData.imagepoolbyname = Data.imagepoolbyname

    TestData.data.anchor_name_list = test_anchor_name_list
    TestData.data.anchor_name_to_idx = Data.data.anchor_name_to_idx
    TestData.data.positive = Data.data.positive
    TestData.data.negative = Data.data.negative
    TestData.imagepool = Data.imagepool
    TestData.imagepoolbyname = Data.imagepoolbyname

    return TrainData, TestData
end
--TestData = LoadData('fashion_pair_test.csv')

TrainData, TestData = SplitData(Data)
RetData= {TrainData=TrainData,
    TestData=TestData
}

print ('save:' .. save_filename)
torch.save( save_filename, RetData)

print ("return")

return RetData
