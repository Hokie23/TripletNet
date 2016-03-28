--local py = require('fb.python')
require 'csvigo'
require 'loadutils'
require 'cudnn'

cmd = torch.CmdLine()

cmd:addTime()
cmd:text()
cmd:text('Extracting a feature on Fashion')
cmd:text()
cmd:text('==>Options')
cmd:option('-batch_list', '/data1/october_11st/batch_list', 'batch list')
cmd:option('-output_list', '', 'batch list')
cmd:option('-model', '', 'model file name')

opt = cmd:parse(arg or {})
local batch_list = opt.batch_list or error("must be batch_list")
local output_list = opt.batch_list
if #output_list == 0 then
    error("must be specified output_list paramter")
end

if #opt.model == 0 then
    error("must be specified model file")
end
local model = torch.load(opt.model)
local lu = loadutils(imagePath )
lu:SetDefaultImagePath('') -- must be empty string because batch_list's image path is absolute.

batch_items = csvigo.load( {path=batch_list, mode='large'} )
for i=1,#batch_items do
    local content_id, mid_category, category_name, imagepath = unpack(batch_items[i])
end

