require "csvigo"
require 'image'
require 'math'
require 'loadutils'
require 'sysutils'

local debugger = require('fb.debugger')

local opt = opt or {}
local PreProcDir = opt.preProcDir or './'
local Whiten = opt.whiten or false
local DataPath = opt.datapath or '/data1/fantajeon/torch/TripletNet/properties/'
local imagePath = opt.imagepath or '/data2/freebee/Images/'

local TestData
local TrainData
local Classes
local ImagePool = {}
local lu = loadutils(imagePath )

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

function SelectListTripletsSimple(db, size, TensorType, SampleStage)
    local data = db.data
    local list = {}
    print ("generate list triplets", size, string.format("%d/%d", SampleStage.current, #data.anchor_name_list) )

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
            if neg_of_anchor == nil or math.random(3) == 0 then
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

        current = current + 1
        --current = current + 1000
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


function GenerateListTriplets(db, size, prefix)
    print ("generate list triplets", size)
    local data = db.data
    local list = {}
    local nClasses = #data.anchor_name_list 
    --for i=1, size,100 do
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


function ParsingShoes(m)
    local path, file, extension = splitfilename(m[2])
    local filename = file .. extention
    local category = m[3]
    local box = {x=m[4],y=m[5],width=m[6],height=m[7]}
    local properties = torch.Tensor(39):zero()

    local gender = m[8]
    if gender = '성인남성' then
        properties[1] = 1
    elseif gender = '성인여성' then
        properties[2] = 1
    elseif gender = '아동남성' then
        properties[3] = 1
    elseif gender = '아동여성' then
        properties[4] = 1
    end

    local material = m[9]
    if material = '가죽' then
        properties[5] = 1
    elseif material = '천' then
        properties[6] = 1
    elseif material = '고무' then
        properties[7] = 1
    end

    local ankle = m[10]
    if ankle = '복숭아뼈아래' then
        properties[8] = 1
    elseif ankle = '복숭아뼈위' then
        properties[9] = 1
    elseif ankle = '종아리' then
        properties[10] = 1
    elseif ankle = '무릎' then
        properties[11] = 1
    end

    local heels = m[11]
    if heels = '로우' then
        properties[12] = 1
    elseif heels = '미드' then
        properties[13] = 1
    elseif heels = '하이' then
        properties[14] = 1
    end

    local pattern = m[12]
    if pattern = '무지' then
        properties[15] = 1
    elseif pattern = '브랜드 로고' then
        properties[16] = 1
    elseif pattern = '스트라이프' then
        properties[17] = 1
    elseif pattern = '도트' then
        properties[18] = 1
    elseif pattern = '호피/지브라' then
        properties[19] = 1
    elseif pattern = '도형' then
        properties[20] = 1
    elseif pattern = '이니셜' then
        properties[21] = 1
    elseif pattern = '밀리터리' then
        properties[22] = 1
    elseif pattern = '배색' then
        properties[23] = 1
    elseif pattern = '꽃무늬' then
        properties[24] = 1
    elseif pattern = '뱀피' then
        properties[25] = 1
    elseif pattern = '체크' then
        properties[26] = 1
    elseif pattern = '일러스트' then
        properties[27] = 1
    end

    -- forefood
    if m[13] = 'O' then
        properties[28] = 1
    end
    -- 끈
    if m[14] = 'O' then
        properties[29] = 1
    end
    -- 지퍼
    if m[15] = 'O' then
        properties[30] = 1
    end
    -- 앞트임
    if m[16] = 'O' then
        properties[31] = 1
    end
    -- 뒷트임
    if m[17] = 'O' then
        properties[32] = 1
    end
    -- 메쉬
    if m[18] = 'O' then
        properties[33] = 1
    end
    -- 리본장식
    if m[19] = 'O' then
        properties[34] = 1
    end
    -- 단추장식
    if m[20] = 'O' then
        properties[35] = 1
    end
    -- 버클
    if m[21] = 'O' then
        properties[36] = 1
    end
    -- 클립
    if m[22] = 'O' then
        properties[37] = 1
    end
    -- 벨트
    if m[23] = 'O' then
        properties[38] = 1
    end
    -- 비즈/징
    if m[24] = 'O' then
        properties[39] = 1
    end

    return filename, box, properties
end

function LoadData(filepath, check_imagefile)
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
        local isbreak = function() 
                m = label_pairs[i]
                local a_name, box, properties_vector = ParsingShoes(m) 
                if a_name == nil then
                    return ""
                end

                if check_imagefile then
                    if ImagePoolByName[a_name] == nil then
                        local img = LoadNormalizedResolutionImage(a_name)
                        if isColorImage(img) then
                            ImagePoolByName[a_name] = true
                            return ""
                        end
                    end
                end

                table.insert(anchor_name_list, {filename=a_name, box=box, properties=properties_vector})
                return ""
            end()
        if isbreak == "break" then
            break
        end
        print (i, #label_pairs, 100.0*(i/#label_pairs), "anchor", #anchor_name_list)
    end

    print("loaded: " .. #anchor_name_list)
    print("loaded imagepool: " .. #ImagePool)
    Data.data.anchor_name_list = anchor_name_list
    Data.Resolution = {3,299,299}

    print ("Data Size:", #Data.data.anchor_name_list)

    return Data
end

function save_data()
    torch.save(PreProcDir .. '/shoes_save.t7', 'save')
    torch.save(PreProcDir .. '/train.resolution.t7', TrainData.Resolution)
    torch.save(PreProcDir .. '/train.data.anchor_name_list.t7', TrainData.data.anchor_name_list)
    torch.save(PreProcDir .. '/train.data.all_negative_list.t7', TrainData.data.all_negative_list)

    torch.save(PreProcDir .. '/test.resolution.t7', TestData.Resolution)
    torch.save(PreProcDir .. '/test.data.anchor_name_list.t7', TestData.data.anchor_name_list)
    torch.save(PreProcDir .. '/test.data.all_negative_list.t7', TestData.data.all_negative_list)
end

function load_cached_data()
    local checkfile = PreProcDir .. '/shoes_save.t7'
    if path.exists( checkfile ) == false then
        print ( string.format("cannot find %s", checkfile) )
        return nil
    end
    TrainData = {data={},Resolution={}}
    TrainData.Resolution = torch.load(PreProcDir .. '/train.resolution.t7')
    TrainData.data.anchor_name_list = torch.load(PreProcDir .. '/train.data.anchor_name_list.t7')
    TrainData.data.all_negative_list = torch.load(PreProcDir .. '/train.data.all_negative_list.t7')

    TestData = {data={},Resolution={}}
    TestData.Resolution = torch.load(PreProcDir .. '/test.resolution.t7')
    TestData.data.anchor_name_list = torch.load(PreProcDir .. '/test.data.anchor_name_list.t7')
    TestData.data.all_negative_list = torch.load(PreProcDir .. '/test.data.all_negative_list.t7')

    return { TrainData = TrainData, TestData = TestData }
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
        if isColorImage(img) then
            table.insert(negative_namelist, neg_name)
        end
    end

    print ( string.format("#negative = %d", #negative_namelist) )
    return negative_namelist
end

--save_filename = PreProcDir .. '/fashion_data.t7' 
save_filename = PreProcDir .. '/shoes_save.t7' 

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
    TrainData = LoadData('excel_1459334593242.shoes.train.csv', false)
    TestData = LoadData('excel_1459334593242.shoes.valid.csv', false)
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
