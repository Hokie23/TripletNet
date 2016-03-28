--local py = require('fb.python')
require 'csvigo'
require 'cutorch'
require 'cunn'
require 'loadutils'
require 'xlua'
require 'cudnn'

cmd = torch.CmdLine()

cmd:addTime()
cmd:text()
cmd:text('Extracting a feature on Fashion')
cmd:text()
cmd:text('==>Options')
cmd:option('-batch_list', '/data1/october_11st/batch_list', 'batch list')
cmd:option('-output_list', '', 'write to result')
cmd:option('-model', '', 'model file name')
cmd:option('-batchSize',          2,                    'batch size')
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
model:evaluate()
local lu = loadutils(imagePath )

local Resolution = lu:Resolution()
lu:SetDefaultImagePath('') -- must be empty string because batch_list's image path is absolute.

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

local replacePath = "/data1/october_11st/october_11st_imgs/"
for i=1,#batch_items do
    local content_id, mid_category, category_name, imagepath = unpack(batch_items[i])
    imagepath = imagepath:gsub("/userdata2/index_11st_20151020/october_11st_imgdata/",replacePath )
    local img = lu:LoadNormalizedResolutionImageCenterCrop(imagepath)

    if img ~= nil  then
        current_batch = current_batch + 1
        batch[current_batch]:copy(img)

        if current_batch == batchSize then
            local y = model:forward( batch )
            current_batch = 0
            batch:cuda()

            for j=1,y:size(1) do
                local feature = string.format("%f", y[j][1])
                for yi=2,y:size(2) do
                    feature = feature .. "," .. string.format("%f", y[j][yi])
                end
                output_file:write(string.format("%s\t%s\t%s\t%s\t%s\n", content_id, mid_category, category_name, imagepath:gsub(replacePath, ''), feature ) )
            end

            io.flush(output_file)
            xlua.progress(i, #batch_items)
        end
    end
end


io.close(output_file)
