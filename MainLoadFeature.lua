
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'xlua'
require 'loadutils'
require "stringutils"
require 'webutils'
require "xlua"
require 'nngraph'
require 'AttentionLSTM'

local json = require('JSON.lua')

local app = require('waffle')
local async = require('async')
local lu = loadutils({''})


async.repl()

async.repl.listen( {host='0.0.0.0', port=8082} )
cmd = torch.CmdLine()

cmd:addTime()
cmd:text()
cmd:text('Extracting a feature on Fashion')
cmd:text()
cmd:text('==>Options')
cmd:option('-dim', 128, 'model dimension')
cmd:option('-batch', 1024*50, 'compare batch size')
cmd:option('-max_feature', 1492181, 'model dimension')
cmd:option('-feature_list', '', 'write to result')
cmd:option('-model', '', 'model file name')

opt = cmd:parse(arg or {})
if #opt.feature_list == 0 then
    error("must be specified feature_list")
end

if #opt.model == 0 then
    error("need model file path")
end

local feature_dim = opt.dim
local feature_list = opt.feature_list
local max_feature = opt.max_feature
local compare_batch = opt.batch
local Resolution = lu:Resolution()

local model = torch.load(opt.model)
model:cuda()
model:evaluate()

local loss = nn.PairwiseDistance(2)
loss:cuda()
loss:evaluate()

torch.setdefaulttensortype('torch.FloatTensor')
local feature_pool = torch.Tensor():type('torch.FloatTensor')
print ("resize file")
local nsz = torch.LongStorage(2)
nsz[1] = max_feature
nsz[2] = feature_dim
print ("batch resize")
feature_pool:resize(nsz)

print ("reading..", feature_list)
file = io.open( feature_list, "r")
local count = 0
local meta_pool = {}
while true do 
    local line = file:read("*line")
    if line == nil then break end

    local fields = line:split("\t")
    local feature = fields[5]:split(",")
    if #feature ~= feature_dim then
        print (count, table.maxn(feature), #feature, feature)
        break
    end
    count = count + 1
    
    for y=1,feature_dim do
        feature_pool[count][y] = feature[y]
    end
    table.insert(meta_pool, {content_id=fields[1], mid_category=fields[2], category=fields[3], imagepath=fields[4]} )

    if math.fmod(count,2000) == 0 then
        xlua.progress(count, 2000000)
    end

    if count >= max_feature then
        break
    end

    --if count == 20000 then
    --    break
    --end
end

print ("count:", count)
print ("feature_pool:", feature_pool:size())
file:close()

function view(req, res)
    local template = './html/hello.html'
    res.render( template, { count = count } )
end

function hello(req, res)
    local template = './html/hello.html'
    res.render( template, { count = count } )
end

function extract_feature(imagepath)
    print ('imagepath', imagepath)
    local nsz = torch.LongStorage(4)
    local batch = torch.Tensor():type( 'torch.CudaTensor' )
    local img = lu:LoadNormalizedResolutionImageCenterCrop(imagepath)
    if img == nil then
        error( string.format("Load error:%s", imagepath) )
    end
    nsz[1] = 1
    nsz[2] = Resolution[1]
    nsz[3] = Resolution[2]
    nsz[4] = Resolution[3]

    batch:resize(nsz)

    batch[1]:copy(img)
    batch:cuda()
    y = model:forward( batch )
    return y
end

function retrieval(y)
    return distance_from_pool(y)
end


function distance_from_pool(X)
    local nsz = torch.LongStorage(2)
    local batchX = torch.repeatTensor(X, compare_batch, 1)
    local Y = torch.Tensor():type( 'torch.CudaTensor' )
    nsz[1] = compare_batch
    nsz[2] = feature_dim

    collectgarbage()
    batchX:cuda()

    Y:resize(nsz)
    Y:cuda()

    local bucket = {}
    for i=1,feature_pool:size(1) do
        table.insert(bucket,{index=i,value=9999})
    end

    print( "pool:", feature_pool:size(1))
    local mind = 9999
    local max_images = math.min(count, feature_pool:size(1))
    for i=1,max_images,compare_batch do
        local bsize = math.min( max_images - i, compare_batch )
        local z = feature_pool[{{i,i+bsize-1},{}}]
        if bsize ~= compare_batch then
            batchX = torch.repeatTensor(X, bsize, 1)
            nsz[1] = bsize
            Y:resize(nsz)
        end
        --print ("#z", z:size())
        --print ("#Y", Y:size())
        Y:copy(z)
        --d = (X - Y)*(X - Y)
        --bucket[i].value = d
        d = loss:forward({batchX,Y})
        for j=1,bsize do
            bucket[i+j-1].value = d[j]
        end
    end
    print ("sorting...")

    table.sort(bucket, function(a, b) 
                return a.value < b.value 
            end )
    local result = {}
    for k=1,100 do
        print ("bucket", bucket[k])
        index = bucket[k].index
        if index > 0 then
            table.insert(result, {rank=k,content_id=meta_pool[index].content_id, mid_category=meta_pool[index].mid_category,
                category=meta_pool[index].category, image_url="http://10.202.35.87/october_11st/" .. meta_pool[index].imagepath, distance=bucket[k].value} )
        end
    end

    for k=1,3 do
        print ( string.format("bucket[%d]: %d, %f", k, bucket[k].index, bucket[k].value) )
    end
    print ("end---")

    return result
end


function query(req, res)
    collectgarbage()
    res.header('Access-Control-Allow-Origin', '*')
    res.header('Content-Type', 'application/json')
    local ok, param_value = pcall(json.decode,json,req.body)
    if ok == false then
        print ("failed to query")
        result = { result=false }
        res.send(result)
        return
    end
    local img_url = param_value.image_url
    print ("image_url", img_url)
    --buf, header = DownloadFileFromURL(img_url)
    ok, body, header, status, code = pcall(HTTPRequest,'GET', img_url)
    if ok == false then
        print ("request failed to get image url:", img_url)
        result = { result=false }
        res.send(result)
        return
    end
    if code ~= 200 then
        print ("code:", code)
        result = { result=false }
        res.send(result)
        return
    end
    local content_type = header["content-type"]
    if string.find(content_type,'image/jpeg') ~= nil or string.find(content_type,'image/jpg') ~= nil then
        imgext = ".jpg"
    elseif string.find(content_type,'image/png') ~= nil then
        imgext = ".png"
    elseif string.find(content_type,'image/gif') ~= nil then
        imgext = ".gif"
    else
        print ("header", header)
        print ("uncompatible image format", content_type)
        res.send( string.format("uncompatible image format: %s", content_type) )
        return
    end

    local filename = string.format("/tmp/%s_%d%s", os.date('%y%m%d_%H:%M:%S'), math.random(100), imgext)
    print ("filename:", filename)
    print ('header', header)

    print ('status:', status, 'code:', code, 'body:', #body)

    file = io.open( filename, "wb+")
    file:write( body )
    file:close()

    local ok, status = pcall( extract_feature, filename )
    if ok == false then
        print("fail to extract feature from image", status)
        res.send("error")
        return
    end

    local ok, retrieval = pcall( retrieval, status )

    result = { query=img_url,
            result=true,
            retrieved_list=retrieval }

    print (header)
    res.send( string.format("%s", json:encode(result)) )
end

app.get('/',hello)
app.post('/query',query)
app.listen({host='0.0.0.0', port=8080})
