require "csvigo"
require 'loadutils'
require 'math'

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Testing a Siamese Distance network on Fashion')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Save/Load Options')
cmd:option('-data',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

opt = cmd:parse(arg or {})

local distance_pair_file = 'distance_pair.csv'

local distance_pair = csvigo.load( {path=distance_pair_file, mode='large'} )

print ('#loaded:', #distance_pair)
local max_bin = 2000
local pos_histogram = {}
local neg_histogram = {}

for i=1,max_bin do
    table.insert(pos_histogram,0)
    table.insert(neg_histogram,0)
end

for i=1,#distance_pair do
    m = distance_pair[i]
    d = tonumber( m[4] ) * 1000
    dbin = math.floor(d) + 1
    if m[3] == 'true' then
        pos_histogram[dbin] = pos_histogram[dbin] + 1
    else
        neg_histogram[dbin] = neg_histogram[dbin] + 1
    end
end

local final_histogram = {}
for i=1,max_bin do
    table.insert(final_histogram, { tostring((i-1)/1000), tostring(pos_histogram[i]), tostring(neg_histogram[i]) } )
end

csvigo.save({path='distance_pair_histogram.csv', data=final_histogram})
