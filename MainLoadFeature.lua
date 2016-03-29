
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'xlua'
require "stringutils"
require 'webutils'

local app = require('waffle')
local async = require('async')


async.repl()

async.repl.listen( {host='0.0.0.0', port=8082} )
cmd = torch.CmdLine()

cmd:addTime()
cmd:text()
cmd:text('Extracting a feature on Fashion')
cmd:text()
cmd:text('==>Options')
cmd:option('-dim', 128, 'model dimension')
cmd:option('-max_feature', 2000000, 'model dimension')
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
--local model = torch.load(opt.model)
--model:cuda()
--model:evaluate()

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

    if count == 100 then
        break
    end
end

print ("count:", count)
file:close()


function hello(req, res)
    local template = './html/hello.html'
    res.render( template, { count = count } )
end

function query(req, res)
    local img_url = req.body
    print ("image_url", img_url)
    --buf, header = DownloadFileFromURL(img_url)
    body, header, status, code = HTTPRequest(img_url)
    if code ~= 200 then
        res.send(header)
        return
    end
    local content_type = header["content-type"]
    if content_type == 'image/jpeg' or content_type == 'image/jpg' then
        imgext = ".jpg"
    elseif content_type == 'image/png' then
        imgext = ".png"
    else
        res.send( string.format("uncompatible image format: %s", content_type) )
        return
    end

    local filename = string.format("/tmp/%s_%d%s", os.date('%y%m%d_%H:%M:%S'), math.random(100), imgext)
    print ("filename:", filename)
    print ('header', header)

    print ('status:', status, 'code:', code, 'body:', #body)
    res.send( string.format("%s",header) )

    file = io.open( filename, "wb+")
    file:write( body )
    file:close()

    print (header)
end

app.get('/',hello)
app.post('/query',query)
app.listen()
