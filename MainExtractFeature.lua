--local py = require('fb.python')
require 'csvigo'
require 'cutorch'
require 'cunn'
require 'loadutils'
require 'xlua'
require 'cudnn'
require 'nngraph'
require 'AttentionLSTM'


cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = true

cmd = torch.CmdLine()

cmd:addTime()
cmd:text()
cmd:text('Extracting a feature on Fashion')
cmd:text()
cmd:text('==>Options')
cmd:option('-batch_list', '/data1/october_11st/batch_list', 'batch list')
cmd:option('-output_list', '', 'write to result')
cmd:option('-model', '', 'model file name')
cmd:option('-weight', '', 'weight file name')
cmd:option('-batchSize',          32,                    'batch size')
cmd:option('-cache', true, 'cache batch list')

opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')


local batchSize = opt.batchSize
local batch_list = opt.batch_list or error("must be batch_list")
local output_list = opt.output_list
if #output_list == 0 then
    error("must be specified output_list paramter")
else
    print ("write to ", output_list)
end
local output_file = io.open( output_list, "w+")

if #opt.model == 0 then
    error("must be specified model file")
end

local model = torch.load(opt.model)
model:cuda()

if #opt.weight > 0 then
    local weight, grad = model:getParameters()
    local w = torch.load( opt.weight )
    weight:copy(w)
end
model:evaluate()
local lu = loadutils( {imagePath} )

local Resolution = lu:Resolution()
lu:SetDefaultImagePath({''}) -- must be empty string because batch_list's image path is absolute.

batch_items = csvigo.load( {path=batch_list, mode='large'} )
local nsz = torch.LongStorage(4)
local batch = torch.Tensor():type( 'torch.CudaTensor' )

nsz[1] = opt.batchSize
nsz[2] = Resolution[1]
nsz[3] = Resolution[2]
nsz[4] = Resolution[3]

print ("nsz", nsz)

batch:resize(nsz)

local current_batch =0
local batch_info = {}

local replacePath = "/data1/october_11st/october_11st_imgs/"
for i=1,#batch_items do
    local content_id, mid_category, category_name, imagepath = unpack(batch_items[i])
    local cond = (function() 
            --if category_name ~= 'shoes' then
            --    return "stop/category"
            --end
            imagepath = imagepath:gsub("/userdata2/index_11st_20151020/october_11st_imgdata/",replacePath )
            local img = lu:LoadNormalizedResolutionImageCenterCrop(imagepath)
            if img == nil then
                return string.format("stop/image-nil(%s)", imagepath)
            end

            table.insert( batch_info, {content_id,mid_category,category_name,imagepath} )
            current_batch = current_batch + 1
            batch[current_batch]:copy(img)

            if current_batch == batchSize then
                batch:cuda()
                local y = model:forward( batch )
                current_batch = 0

                for j=1,y:size(1) do
                    local feature = string.format("%f", y[j][1])
                    for yi=2,y:size(2) do
                        feature = feature .. "," .. string.format("%f", y[j][yi])
                    end

                    local info = batch_info[j]

                    local onlyfilename = info[4]:gsub(replacePath,'')
                    output_file:write(string.format("%s\t%s\t%s\t%s\t%s\n", info[1], info[2], info[3], onlyfilename, feature ) )
                end

                io.flush(output_file)
                xlua.progress(i, #batch_items)

                batch_info = {}
            end

            return "succeeded"
        end)()

    if cond == "break" then
        break
    --elseif cond == "succeeded" then
    --    print (imagepath, '...ok')
    --else
    --    print (imagepath, '...failed', category_name, 'reason:', cond)
    end
end


io.close(output_file)
