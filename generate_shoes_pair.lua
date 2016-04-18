require 'nn'
require 'csvigo'
require 'cudnn'
require 'properties_util'
require 'xlua'
local debugger = require 'fb.debugger'

torch.setdefaulttensortype('torch.FloatTensor')

--filepath = './properties/excel_1459334593242.shoes.train.csv'
--pair_output_filename = 'shoes_pair.train.csv'
filepath = './properties/excel_1459334593242.shoes.valid.csv'
pair_output_filename = 'shoes_pair.valid.csv'
category_name = 'shoes'

db = torch.Tensor():type( 'torch.CudaTensor' )

local loss = nn.PairwiseDistance(1)
loss:cuda()
loss:evaluate()

function distance(X, Y)
    --local off = torch.cmin(X:eq(0):sum(2), Y:eq(0):sum(2))
    --local hd = X:eq(Y):sum(2)
    --local d = (hd- off ):add( -17 ):neg()
    --return d

    return X:ne(Y):sum(2)
end

function getKnn(Data, anchor_index, k, threshold_distance)
    local X = Data.data.anchor_name_list[anchor_index].properties
    local batchX = torch.repeatTensor(X, db:size(1), 1):type( 'torch.CudaTensor' )

    collectgarbage()
    batchX:cuda()

    local bucket = {}
    local d = distance(batchX, db)
    for j=1,d:size(1) do
        table.insert(bucket,{index=j,value=d[j][1]})
    end
    table.sort(bucket, function(a, b) 
                return a.value < b.value 
            end )

    local result = {}
    for i=1,#bucket do
        index = bucket[i].index
        if #result >= k then
            break
        end
        if bucket[i].index ~= anchor_index then
            if bucket[i].value < threshold_distance then
                table.insert(result, {index, bucket[i].value} )
            end
        end
    end

    local neg_result = {}
    for i=#bucket,k+1,-500 do
        if #neg_result >= k or bucket[i].value < 3 then
            break
        end

        index = bucket[i].index
        table.insert(neg_result, {index, bucket[i].value} )
    end
    return result, neg_result
end

function LoadData(filepath)
    local Data = {data={},imagepool={}}
    local positive_pairs = {}
    local negative_pairs = {}
    local anchor_name_list = {}
    local ImagePool = {}
    local count_imagepool = 0
    local ImagePoolByName = {}

    label_pairs = csvigo.load( {path=filepath, mode='large'} )

    --for i=1,#label_pairs,1000 do
    for i=1,#label_pairs do
        local isbreak = (function() 
                m = label_pairs[i]
                local a_name, box, properties = ParsingShoes(m) 
                --print (a_name, box, properties)
                if a_name == nil then
                    return ""
                end

                table.insert(anchor_name_list, {filename=a_name, box=box, properties=properties})
                return ""
            end)()
        if isbreak == "break" then
            break
        end
        print (i, #label_pairs, 100.0*(i/#label_pairs), "anchor", #anchor_name_list)
    end

    print("loaded: " .. #anchor_name_list)
    Data.data.anchor_name_list = anchor_name_list
    Data.Resolution = {3,299,299}

    print ("Data Size:", #Data.data.anchor_name_list)

    return Data
end


Data = LoadData(filepath)
function build_db()
    db_size = #Data.data.anchor_name_list
    local feature_dim = Data.data.anchor_name_list[1].properties:size(1)
    local nsz = torch.LongStorage(2)
    nsz[1] = db_size
    nsz[2] = feature_dim
    db:resize(nsz)
    for i=1,db_size do
        local z = Data.data.anchor_name_list[i].properties
        db[i]:copy(z)
    end

    db:cuda()
end

print ('building db....')
build_db()

f_pair_out = torch.DiskFile( pair_output_filename, "w")
for i=1,#Data.data.anchor_name_list do
    local a_name = Data.data.anchor_name_list[i].filename
    result, negresult = getKnn(Data, i, 50, 1)
    for j=1,#result do
        local p_name = Data.data.anchor_name_list[ result[j][1] ].filename
        str = string.format( '%s,%s,%s,1,%s\n', category_name, a_name, p_name, result[j][2])
        f_pair_out:writeString( str )
    end
    for j=1,#negresult do
        local p_name = Data.data.anchor_name_list[ negresult[j][1] ].filename
        str = string.format( '%s,%s,%s,0,%s\n', category_name, a_name, p_name, negresult[j][2])
        f_pair_out:writeString( str )
    end

    xlua.progress( i, #Data.data.anchor_name_list )
end
f_pair_out:close()
