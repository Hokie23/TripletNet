require 'nn'
require 'csvigo'
require 'cudnn'
require 'properties_util'
require 'xlua'
local debugger = require 'fb.debugger'

torch.setdefaulttensortype('torch.FloatTensor')

filepath = './properties/excel_1459334593242.shoes.train.csv'
pair_output_filename = 'shoes_pair.train.csv'
category_name = 'shoes'

local loss = nn.PairwiseDistance(1)
loss:cuda()
loss:evaluate()

function distance(X, Y)
    local d = X:eq(Y):sum(2)
    return d
end

function getKnn(Data, i, k, threshold_distance)
    local X = Data.data.anchor_name_list[i].properties
    local feature_dim = X:size(1)
    local nsz = torch.LongStorage(2)
    local compare_batch = 2048
    local batchX = torch.repeatTensor(X, compare_batch, 1):type( 'torch.CudaTensor' )
    local Y = torch.Tensor():type( 'torch.CudaTensor' )
    nsz[1] = compare_batch
    nsz[2] = feature_dim

    collectgarbage()
    batchX:cuda()

    Y:resize(nsz)
    Y:cuda()


    collectgarbage()
    local bucket = {}
    for i=1,#Data.data.anchor_name_list do
        table.insert(bucket,{index=i,value=9999})
    end

    local dbsize = #Data.data.anchor_name_list
    for i=1,dbsize,compare_batch do
        local bsize = math.min( dbsize - i, compare_batch )
        if bsize ~= compare_batch then
            batchX = torch.repeatTensor(X, bsize, 1):type( 'torch.CudaTensor' )
            nsz[1] = bsize
            Y:resize(nsz)
            Y:cuda()
        end
        for j=1,bsize do
            local z = Data.data.anchor_name_list[i+j-1].properties
            Y[j]:copy(z)
        end
        batchX:cuda()
        Y:cuda()

        --print (Y:type(), batchX:type())
        --d = loss:forward({batchX,Y})
        d = distance(batchX, Y)
        for j=1,bsize do
            bucket[i+j-1].value = d[j][1]
        end
        
    end
    table.sort(bucket, function(a, b) 
                return a.value > b.value 
            end )

    local result = {}
    for i=1,k do
        index = bucket[i].index
        if bucket[i].value < threshold_distance then
            table.insert(result, index )
        end
    end
    return result

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

f_pair_out = torch.DiskFile( pair_output_filename, "w")
for i=1,#Data.data.anchor_name_list,10000 do
    local a_name = Data.data.anchor_name_list[i].filename
    result = getKnn(Data, i, 50, 5)
    for j=1,#result do
        local p_name = Data.data.anchor_name_list[ result[j] ]
        str = string.format( '%s,%s,%s,1\n', category_name, a_name, p_name)
        f_pair_out:writeString( str )
    end
end
f_pair_out:close()
