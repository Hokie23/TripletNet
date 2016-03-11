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

function LoadNormalizedResolutionImage(filename)
    --print ("loading..." .. filename)
    local imagepath = imagePath .. filename
    return preprocess(imagepath)
end


function dist(a, b)
    local d = (a -b)*(a-b)
    return d
    --return torch.dist(a,b)
end

function SelectListTriplets(embedding_net, db, size, TensorType)
    print ("select list triplets", size)

    local data = db.data
    print ("dbsize:", #data.anchor_name_list)
    local list = {}
    local nClasses = #data.anchor_name_list 

    local candidate_anchor_list = {}
    for i=1, size do
        local anchor_name = data.anchor_name_list[i]
        if anchor_name ~= nil then
            pos_of_anchor = data.positive[anchor_name]
            if pos_of_anchor ~= nil then
                print ("insert", anchor_name)
                table.insert(candidate_anchor_list, anchor_name)
            end
        end
    end

    print ("canidate list", #candidate_anchor_list)

    for i=1, size do
        local anchor_img
        local anchor_vector, positive_vector, negative_vector

        local ap_dist, an_dist
        print ("generate list #" .. i .. "/#" .. #candidate_anchor_list)
        local c1, anchor_name, hard_positive_name, semi_hard_negative_name
        c1 = math.random(#candidate_anchor_list)

        local nsz = torch.LongStorage(4)
        while true do
            anchor_name = candidate_anchor_list[c1]
            local batch = torch.Tensor():type( TensorType )
            --print ( 'anchor_name:', anchor_name )

            anchor_img = LoadNormalizedResolutionImage(anchor_name)
            nsz[1] = 1
            nsz[2] = 3
            nsz[3] = 299
            nsz[4] = 299
            batch:resize(nsz)
            batch[1]:copy(anchor_img)

            --print ("anchor_image:size", anchor_img:size())
            a_output = embedding_net:forward( batch )
            anchor_vector = a_output:clone()

            pos_of_anchor = data.positive[anchor_name]
            if pos_of_anchor ~= nil then
                local dupcheck = {}
                local batch_name = {}
                minbatchSize = math.min(#pos_of_anchor,4)
                nsz[1] = minbatchSize
                batch:resize(nsz)
                for pi=1,minbatchSize do
                    positive_name = pos_of_anchor[math.random(#pos_of_anchor)]
                    if dupcheck[positive_name] == nil then
                        dupcheck[positive_name] = true
                        local img = LoadNormalizedResolutionImage(positive_name) 
                        table.insert(batch_name, positive_name)
                        batch[pi]:copy(img)
                    end
                end

                positive_vector = embedding_net:forward( batch )
                local max_pdist = -1 
                local max_im = {}
                for pi=1,minbatchSize do
                    bdist = dist(anchor_vector, positive_vector[pi])
                    --print ("positive_dist", bdist)
                    if bdist > max_pdist then
                        max_pdist = bdist
                        hard_positive_name = batch_name[pi]
                    end
                end

                ap_dist = max_pdist
                if ap_dist ~= 0 then
                    break
                end
            else
                print ("empty positive", anchor_name)
            end
            c1 = math.random(#candidate_anchor_list)
        end

        local neg_batch = torch.Tensor():type( TensorType )
        nsz[1] = 4
        neg_batch:resize( nsz )
        bisfound = false

        --print ("anchor_name", anchor_name, "ap_dist:", ap_dist)
        for trial=1,10 do
            local neg_batch_name = {}
            for ni=1,4 do
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
                local img = LoadNormalizedResolutionImage(negative_name)
                table.insert(neg_batch_name, negative_name)
                neg_batch[ni]:copy(img)
            end

            negative_vector = embedding_net:forward(neg_batch)
            local small_dist = 99999
            semi_hard_negative_name = nil
            for ni=1,4 do
                bdist = dist(anchor_vector, negative_vector[ni])
                --print ("negative bdist", bdist)
                if bdist > 0 then
                    if (bdist > ap_dist and bdist < small_dist) or math.random(5) == 1 then
                        small_dist = bdist
                        semi_hard_negative_name = neg_batch_name[ni]
                    end
                end
            end

            if semi_hard_negative_name ~= nil then
                an_dist = small_dist
                bisfound = true
                break
            end
        end

        print ("anchor_name", anchor_name, "ap_dist:", ap_dist, "an_dist", an_dist)

        --print(anchor_name, negative_name, positive_name)
       
        if bisfound then
            local exemplar = {anchor_name, semi_hard_negative_name, hard_positive_name}
            --print ("exemplar", exemplar)
            table.insert(list, exemplar)
        end
    end

    print ("Selection Generate:", #list)
    return list
end

function GenerateListTriplets(db, size)
    print ("generate list triplets", size)
    local data = db.data
    local list = {}
    local nClasses = #data.anchor_name_list 
    for i=1, size do
        --print ("generate list #" .. i .. "/#" .. size)
        local c1, anchor_name, positive_name, negative_name
        c1 = i
        while true do
            anchor_name = data.anchor_name_list[c1]

            pos_of_anchor = data.positive[anchor_name]
            if pos_of_anchor ~= nil then
                positive_name = pos_of_anchor[math.random(#pos_of_anchor)]
                break
            end
            c1 = math.random(nClasses)
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

        local exemplar = {anchor_name, negative_name, positive_name}
        table.insert(list, exemplar)
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

    for i=1,#label_pairs,1000 do
        m = label_pairs[i]
        local a_name = m[2]
        local t_name = m[3]
        local p_or_n = m[4]
        local bcontinue = false

        bcontinue = false

        if a_name ~= t_name and a_name ~= nil then
            --print ("anchor_name=", a_name)
            if ImagePoolByName[a_name] == nil then
                local img = LoadNormalizedResolutionImage(a_name)
                if isColorImage(img) then
                    count_imagepool = count_imagepool + 1
                    ImagePoolByName[a_name] = true
                else
                    --print("ancnor continue")
                    bcontinue = true
                end
            end
            if ImagePoolByName[t_name] == nil then
                local img = LoadNormalizedResolutionImage(t_name)
                if isColorImage(img) then
                    count_imagepool = count_imagepool + 1
                    ImagePoolByName[t_name] = true
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
        print (i, #label_pairs, 100.0*(i/#label_pairs), count_imagepool, "anchor", #anchor_name_list)
    end

    print("loaded: " .. #anchor_name_list)
    print("loaded imagepool: " .. #ImagePool)
    Data.data.anchor_name_to_idx = anchor_name_to_idx
    Data.data.anchor_name_list = anchor_name_list
    Data.data.positive = positive_pairs
    Data.data.negative = negative_pairs

    return Data
end

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
        local pos_of_anchor = Data.data.positive[anchor_name] 
        if pos_of_anchor ~= nil then
            if positive_cnt >= 10 then
                postivie_cnt = 0
                table.insert(test_anchor_name_list, Data.data.anchor_name_list[i])
            else
                positive_cnt = positive_cnt + 1
                table.insert(train_anchor_name_list, Data.data.anchor_name_list[i])
            end
        end
    end

    TrainData.data.anchor_name_to_idx = Data.data.anchor_name_to_idx
    TrainData.data.anchor_name_list = train_anchor_name_list
    TrainData.data.positive = Data.data.positive
    TrainData.data.negative = Data.data.negative
    TrainData.Resolution = {3, 299, 299}

    TestData.data.anchor_name_list = test_anchor_name_list
    TestData.data.anchor_name_to_idx = Data.data.anchor_name_to_idx
    TestData.data.positive = Data.data.positive
    TestData.data.negative = Data.data.negative
    TestData.Resolution = {3, 299, 299}

    print ("Train Data size:", #TrainData.data.anchor_name_list)
    print ("Test Data size:", #TestData.data.anchor_name_list)

    return TrainData, TestData
end

function save_data()
    torch.save(PreProcDir .. 'fashion_save.t7', 'save')
    torch.save(PreProcDir .. 'train.resolution.t7', TrainData.Resolution)
    torch.save(PreProcDir .. 'train.data.anchor_name_list.t7', TrainData.data.anchor_name_list)
    torch.save(PreProcDir .. 'train.data.positive.t7', TrainData.data.positive)
    torch.save(PreProcDir .. 'train.data.negative.t7', TrainData.data.negative)

    torch.save(PreProcDir .. 'test.resolution.t7', TestData.Resolution)
    torch.save(PreProcDir .. 'test.data.anchor_name_list.t7', TestData.data.anchor_name_list)
    torch.save(PreProcDir .. 'test.data.positive.t7', TestData.data.positive)
    torch.save(PreProcDir .. 'test.data.negative.t7', TestData.data.negative)
end

function load_data()
    if path.exists( PreProcDir .. 'fashion_save.t7') then
        return nil
    end
    TrainData = {data={},Resolution={}}
    TrainData.Resolution = torch.load(PreProcDir .. 'train.resolution.t7')
    TrainData.data.anchor_name_list = torch.load(PreProcDir .. 'train.data.anchor_name_list.t7')
    TrainData.data.positive = torch.load(PreProcDir .. 'train.data.positive.t7')
    TrainData.data.negative = torch.load(PreProcDir .. 'train.data.negative.t7')

    TestData = {data={},Resolution={}}
    TestData.Resolution = torch.load(PreProcDir .. 'test.resolution.t7')
    TestData.data.anchor_name_list = torch.load(PreProcDir .. 'test.data.anchor_name_list.t7')
    TestData.data.positive = torch.load(PreProcDir .. 'test.data.positive.t7')
    TestData.data.negative = torch.load(PreProcDir .. 'test.data.negative.t7')

    return RetData { TrainData = TrainData, TestData = TestData }
end

save_filename = PreProcDir .. '/fashion_data.t7' 

print ("check chached file:" .. save_filename)
if path.exists(save_filename) ~= false then
    print ("cached model")
    Data = torch.load(save_filename )

    print ("Train Data size:", #Data.TrainData.data.anchor_name_list)
    print ("Test Data size:", #Data.TestData.data.anchor_name_list)
    return Data
end

local Data = LoadData('fashion_pair.csv')

TrainData, TestData = SplitData(Data)
RetData= {TrainData=TrainData, TestData=TestData}

print ('save:' .. save_filename)
torch.save( save_filename, RetData)
save_data()

print ("return")

return RetData
