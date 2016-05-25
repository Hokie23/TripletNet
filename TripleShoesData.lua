require "csvigo"
require 'image'
require 'math'
require 'loadutils'

local debugger = require('fb.debugger')

local dbname = 'shoes'
local opt = opt or {}
local PreProcDir = opt.preProcDir or './'
local Whiten = opt.whiten or false
local DataPath = opt.datapath or '/data1/fantajeon/torch/TripletNet/'
local SimpleNormalization = (opt.normalize==1) or false
local imagepath = nil
assert(dbname~= nil, 'dbname required')
imagePath = opt.imagepath or '/data2/freebee/Images/'
assert(imagePath~= nil, 'imaegPath is empty')


local TestData
local TrainData
local Classes
local ImagePool = {}
local lu = loadutils( {imagePath} )

function dist(a, b)
    --local d = (a -b)*(a-b)
    --return d:pow(0.5)
    local ok, d = pcall(torch.dist,a,b)
    if ok == false then
        error(d)
    end
    return d
end

function LoadNormalizedResolutionImageCenterCrop(filename, jitter)
    return lu:LoadNormalizedResolutionImageCenterCrop(filename)
end

function LoadNormalizedResolutionImage(filename, jitter)
    --print ("LoadNormalizedResolutionImage", filename)
    return lu:LoadNormalizedResolutionImage(filename, jitter)
end

function ShuffleTrain(db, SampleState)
    print ('shuffing...')
    local rand = math.random
    local data = db.data
    local nclasses = #data.anchor_name_list
    for i=nclasses,2,-1 do
        j = rand(i)
        data.anchor_name_list[i], data.anchor_name_list[j] = data.anchor_name_list[j], data.anchor_name_list[i]
    end
    SampleState.current = 1
end

function SelectListTriplets(embedding_net, db, size, TensorType, SampleStage)
    print ("select list triplets", size)

    local data = db.data
    print ("dbsize:", #data.anchor_name_list)
    local list = {}
    local nClasses = #data.anchor_name_list 

    local isend = false
    local current = SampleStage.current or 1
    while #list < size do
        local anchor_img
        local anchor_vector, positive_vector, negative_vector
        local anchor_jitter, positive_jitter, negative_jitter

        local ap_dist, an_dist
        print ("generate list #" .. current .. "/#" .. #data.anchor_name_list .. string.format("[%d-#s%d]",current, SampleStage.current) )
        local c1, anchor_name, hard_positive_name, semi_hard_negative_name
        --c1 = math.random(#candidate_anchor_list)
        c1 = current

        local nsz = torch.LongStorage(4)
        while true do
            hard_positive_name = nil
            anchor_name = data.anchor_name_list[c1]
            local batch = torch.Tensor():type( TensorType )
            --print ( 'anchor_name:', anchor_name )

            anchor_img, anchor_jitter = LoadNormalizedResolutionImage(anchor_name)
            assert(anchor_img ~= nil)
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
                local batch_jitter = {}
                minbatchSize = math.min(#pos_of_anchor,4)
                nsz[1] = minbatchSize
                batch:resize(nsz)
                for pi=1,minbatchSize do
                    positive_name = pos_of_anchor[math.random(#pos_of_anchor)]
                    local img, p_jitter = LoadNormalizedResolutionImage(positive_name) 
                    assert(img ~= nil)
                    table.insert(batch_name, positive_name)
                    table.insert(batch_jitter, p_jitter)
                    batch[pi]:copy(img)
                end

                positive_vector = embedding_net:forward( batch )
                local max_pdist = -1 
                if minbatchSize == 1 then
                    ok, bdist = pcall(dist,anchor_vector, positive_vector)
                    if ok == false then
                        print ("******:", minbatchSize )
                        print("positive error:", positive_vector:size())
                    end
                    max_pdist = bdist
                    hard_positive_name = batch_name[1]
                    positive_jitter = batch_jitter[1]
                else
                    for pi=1,minbatchSize do
                        ok, bdist = pcall(dist,anchor_vector, positive_vector)
                        if bdist > max_pdist then
                            max_pdist = bdist
                            hard_positive_name = batch_name[pi]
                            positive_jitter = batch_jitter[pi]
                        end
                    end
                end

                ap_dist = max_pdist
                if ap_dist >= 0 then
                    --print ("hard_positive_name:", hard_positive_name)
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
        local small_dist = 99999
        semi_hard_negative_name = nil
        for trial=1,10 do
            local neg_batch_name = {}
            local neg_batch_jitter = {}
            for ni=1,4 do
                local neg_of_anchor = data.negative[anchor_name]
                if neg_of_anchor == nil or math.random(2) == 1 then
                    local n1 = c1
                    local n3 = math.random( #data.all_negative_list )
                    negative_name = data.all_negative_list[n3]
                    while negative_name  == anchor_name do
                        n3 = math.random( #data.all_negative_list )
                        negative_name = data.all_negative_list[n3]
                    end
                else
                    local n3 = math.random(#neg_of_anchor)
                    negative_name = neg_of_anchor[n3]
                end
                local img, n_jitter = LoadNormalizedResolutionImage(negative_name)
                assert(img ~= nil )
                table.insert(neg_batch_name, negative_name)
                table.insert(neg_batch_jitter, n_jitter)
                neg_batch[ni]:copy(img)
            end

            negative_vector = embedding_net:forward(neg_batch)
            --local small_dist = 99999
            --semi_hard_negative_name = nil
            for ni=1,4 do
                bdist = dist(anchor_vector, negative_vector[ni])
                --print ("negative bdist", bdist)
                if bdist > 0.000002 then
                    if (bdist > ap_dist and bdist < small_dist) or math.random(10) == 1 then
                        small_dist = bdist
                        semi_hard_negative_name = neg_batch_name[ni]
                        negative_jitter = neg_batch_jitter[ni]
                    end
                end
            end

            if trial > 5 and semi_hard_negative_name ~= nil then
                an_dist = small_dist
                bisfound = true
                break
            end
        end

        --print(anchor_name, negative_name, positive_name)
       
        if bisfound then
            print ("anchor_name", anchor_name, "ap_dist:", ap_dist, "an_dist", an_dist)
            print ("anchor_name", anchor_name, "n:", semi_hard_negative_name, "p:", hard_positive_name)
            local exemplar_name = {anchor_name, semi_hard_negative_name, hard_positive_name}
            local exemplar_jitter = {anchor_jitter, negative_jitter, positive_jitter}
            local exemplar = {names=exemplar_name, jitter=exemplar_jitter}
            table.insert(list, exemplar)
            print ("exemplar", exemplar)
        end

        --current = current + 100
        current = current + 1
        if current > #data.anchor_name_list then
            isend = true
            current = 1
        end
    end

    SampleStage.isend = isend
    SampleStage.current = current
    --print ("Selection Generate:", #list)
    return list
end

function SelectListTripletsSimple(db, size, TensorType, SampleStage)
    local data = db.data
    local list = {}
    print ("generate simple list triplets", size, string.format("%d/%d", SampleStage.current, #data.anchor_name_list) )

    local isend = false
    local current = SampleStage.current or 1
    --for i=1, size,100 do
    while #list < size do
        --print ("generate list #" .. string.format("%d:%d",#list,current) .. "/#" .. size)
        local c1, anchor_name, positive_name, negative_name

        c1 = current
        local isbreak = (function() 
            anchor_name = data.anchor_name_list[c1]

            pos_of_anchor = data.positive[anchor_name]
            if pos_of_anchor == nil then
                return nil
            end
            positive_name = pos_of_anchor[math.random(#pos_of_anchor)]

            local neg_of_anchor = data.negative[anchor_name]
            if neg_of_anchor == nil or math.random(2) == 1 then
                local n1 = c1
                local n3 = math.random( #data.all_negative_list )
                negative_name = data.all_negative_list[n3]
                --print ("0", #data.all_negative_list, "n:", negative_name)
                while negative_name  == anchor_name do
                    n3 = math.random( #data.all_negative_list )
                    negative_name = data.all_negative_list[n3]
                end
            else
                local n3 = math.random(#neg_of_anchor)
                negative_name = neg_of_anchor[n3]
            end

            local exemplar_names = {anchor_name, negative_name, positive_name}
            local exemplar = {names=exemplar_names, jitter={}}

            --print( exemplar )
            assert(anchor_name ~= nil)
            assert(negative_name ~= nil)
            assert(positive_name ~= nil)

            table.insert(list, exemplar)
            return nil
        end)()

        if isbreak == "break" then
            break
        end

        --current = current + 1
        --current = current + 100
        current = current + 20
        if current > #data.anchor_name_list then
            isend = true
            current = 1
        end
    end
    SampleStage.isend = isend
    SampleStage.current = current
    --print ("Selection Generate:", #list)
    return list
end


function GenerateListTriplets(db, size, prefix, SampleStage)
    print ("generate list triplets", size)
    local data = db.data
    local list = {}
    local nClasses = #data.anchor_name_list 
    --for i=1, size, 100 do
    for i=1, size, 20 do
    --for i=1, size do
        print ("generate list #" .. i .. "/#" .. size)
        local c1, anchor_name, positive_name, negative_name
        c1 = i
        while true do
            anchor_name = data.anchor_name_list[c1]

            pos_of_anchor = data.positive[anchor_name]
            if pos_of_anchor ~= nil then
                positive_name = pos_of_anchor[math.random(#pos_of_anchor)]
                break
            else
                print ("pos empty", anchor_name)
            end
            c1 = math.random(nClasses)
        end

        local neg_of_anchor = data.negative[anchor_name]
        if neg_of_anchor == nil or math.random(2) == 1 then
            local n1 = c1
            local n3 = math.random( #data.all_negative_list )
            negative_name = data.all_negative_list[n3]
            print ("0", #data.all_negative_list, "n:", negative_name)
            while negative_name  == anchor_name do
                n3 = math.random( #data.all_negative_list )
                negative_name = data.all_negative_list[n3]
                print("1")
            end
        else
            print ("2")
            local n3 = math.random(#neg_of_anchor)
            negative_name = neg_of_anchor[n3]
        end

        local exemplar_names = {anchor_name, negative_name, positive_name}
        local exemplar = {names=exemplar_names, jitter={}}

        print( prefix, exemplar )
        assert(anchor_name ~= nil)
        assert(negative_name ~= nil)
        assert(positive_name ~= nil)

        table.insert(list, exemplar)
    end
    return list
end

function countTableSize(table)
    local n = 0
    for k, v in pairs(table) do
        n = n + 1
    end
    return n
end

function LoadDataShoes(filepath, check_imagefile)
    local Data = {data={},imagepool={}}
    local positive_pairs = {}
    local negative_pairs = {}
    local anchor_name_to_idx = {}
    local anchor_name_list = {}
    local anchor_count = 0
    local ImagePool = {}
    local count_imagepool = 0
    local ImagePoolByName = {}
    local check_imagefile = check_imagefile or false

    label_pairs = csvigo.load( {path=DataPath .. filepath, mode='large'} )

    --for i=1,#label_pairs,1000 do
    for i=1,#label_pairs do
        --print (label_pairs)
        --debugger.enter()
        m = label_pairs[i]
        local a_name = m[2]
        local t_name = m[3]
        local p_or_n = m[4]
        local bcontinue = false

        local cond = (function() 
                if a_name == t_name or a_name == nil then
                    return "error"
                end
                if check_imagefile then
                    if ImagePoolByName[a_name] == nil then
                        local img = LoadNormalizedResolutionImage(a_name)
                        if lu.isColorImage(img) then
                            ImagePoolByName[a_name] = true
                        else
                            return "error"
                        end
                    end
                    if ImagePoolByName[t_name] == nil then
                        local img = LoadNormalizedResolutionImage(t_name)
                        if lu.isColorImage(img) then
                            ImagePoolByName[t_name] = true
                        else
                            return "error"
                        end
                    end
                end

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

                return "succeeded"
            end)()
        if cond == "succeeded" then
            print (i, #label_pairs, 100.0*(i/#label_pairs), "anchor", #anchor_name_list)
        else
            print("error", a_name)
        end
    end

    print("loaded: " .. #anchor_name_list)
    print("loaded imagepool: " .. #ImagePool)
    Data.data.anchor_name_list = anchor_name_list
    Data.data.positive = positive_pairs
    Data.data.negative = negative_pairs
    Data.Resolution = {3,299,299}

    print ("Data Size:", #Data.data.anchor_name_list)

    return Data
end

local save_filename = PreProcDir .. '/' .. dbname .. '_save.t7' 
function save_data()
    torch.save(save_filename, 'save')
    torch.save(PreProcDir .. '/train.resolution.t7', TrainData.Resolution)
    torch.save(PreProcDir .. '/train.data.anchor_name_list.t7', TrainData.data.anchor_name_list)
    torch.save(PreProcDir .. '/train.data.positive.t7', TrainData.data.positive)
    torch.save(PreProcDir .. '/train.data.negative.t7', TrainData.data.negative)
    torch.save(PreProcDir .. '/train.data.all_negative_list.t7', TrainData.data.all_negative_list)

    torch.save(PreProcDir .. '/test.resolution.t7', TestData.Resolution)
    torch.save(PreProcDir .. '/test.data.anchor_name_list.t7', TestData.data.anchor_name_list)
    torch.save(PreProcDir .. '/test.data.positive.t7', TestData.data.positive)
    torch.save(PreProcDir .. '/test.data.negative.t7', TestData.data.negative)
    torch.save(PreProcDir .. '/test.data.all_negative_list.t7', TestData.data.all_negative_list)
end

function load_cached_data()
    local checkfile = save_filename
    if path.exists( checkfile ) == false then
        print ( string.format("cannot find %s", checkfile) )
        return nil
    end
    TrainData = {data={},Resolution={}}
    TrainData.Resolution = torch.load(PreProcDir .. '/train.resolution.t7')
    TrainData.data.anchor_name_list = torch.load(PreProcDir .. '/train.data.anchor_name_list.t7')
    TrainData.data.positive = torch.load(PreProcDir .. '/train.data.positive.t7')
    TrainData.data.negative = torch.load(PreProcDir .. '/train.data.negative.t7')
    TrainData.data.all_negative_list = torch.load(PreProcDir .. '/train.data.all_negative_list.t7')

    TestData = {data={},Resolution={}}
    TestData.Resolution = torch.load(PreProcDir .. '/test.resolution.t7')
    TestData.data.anchor_name_list = torch.load(PreProcDir .. '/test.data.anchor_name_list.t7')
    TestData.data.positive = torch.load(PreProcDir .. '/test.data.positive.t7')
    TestData.data.negative = torch.load(PreProcDir .. '/test.data.negative.t7')
    TestData.data.all_negative_list = torch.load(PreProcDir .. '/test.data.all_negative_list.t7')

    return { TrainData = TrainData, TestData = TestData }
end

function FilterOutEmptyPositive(Data)
    local filtered_anchor = {}
    for i=1,#Data.data.anchor_name_list do
        local anchor_name = Data.data.anchor_name_list[i]
        local pos_of_anchor = Data.data.positive[anchor_name] 
        if pos_of_anchor ~= nil then
            table.insert(filtered_anchor,anchor_name)
        end
    end

    print ("#anchorlist", #Data.data.anchor_name_list, "-> #filtered list", #filtered_anchor)
    Data.data.anchor_name_list = filtered_anchor

    return Data
end

function LoadNegativeData(negative_filepath)
    print (string.format( DataPath .. negative_filepath ) )
    local negative_namelist = {}
    negative_list = csvigo.load( {path=DataPath .. negative_filepath, mode='large'} )

    print ("loaded: ", #negative_list)
    for i=1,#negative_list,100 do
        xlua.progress(i, #negative_list)
        neg_name = negative_list[i][1]
        local img = LoadNormalizedResolutionImage(neg_name)
        if lu.isColorImage(img) then
            table.insert(negative_namelist, neg_name)
        end
    end

    print ( string.format("#negative = %d", #negative_namelist) )
    return negative_namelist
end


local NegativeList = {}
negative_cache_filename = PreProcDir .. '/negative_list.t7'
if path.exists(negative_cache_filename) then
    print ("load cached negative data")
    NegativeList = torch.load(negative_cache_filename)
    NegativeList.cache = true
else
    NegativeList = LoadNegativeData('negative_list.txt')
    torch.save(negative_cache_filename, NegativeList)
    NegativeList.cache = false
end 

print ("#negative_list: ", #NegativeList)
if path.exists(save_filename) then
    print ("load cached train/validation data")
    RetData = load_cached_data()
    RetData.cache = true
else
    TrainData = LoadDataShoes('shoes_pair.train.csv', true)
    TestData = LoadDataShoes('shoes_pair.valid.csv', true)
    TrainData.data.all_negative_list = NegativeList
    TestData.data.all_negative_list = NegativeList
    RetData= {TrainData=TrainData, TestData=TestData}
    RetData.cache = false
end

RetData.TrainData = FilterOutEmptyPositive(RetData.TrainData)
RetData.TestData = FilterOutEmptyPositive(RetData.TestData)

print ('save:' .. save_filename)
if RetData.cache == false then
    torch.save( save_filename, RetData)
    save_data()
end

print ("return")

return RetData
