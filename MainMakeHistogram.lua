require "csvigo"
require 'loadutils'
require 'math'
require "gnuplot"

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Testing a Siamese Distance network on Fashion')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Save/Load Options')
cmd:option('-data',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Output Meta options')
cmd:option('-project', '', 'project name')

opt = cmd:parse(arg or {})

if opt.project == '' then
    print ('-project option must be required')
    return
end

local project = opt.project
local distance_pair_file = project .. '.csv'

local distance_pair = csvigo.load( {path=distance_pair_file, mode='large'} )

print ('#loaded:', #distance_pair)
local max_bin = 1000
local pos_histogram = {}
local neg_histogram = {}

for i=1,max_bin do
    table.insert(pos_histogram,0)
    table.insert(neg_histogram,0)
end

local total_sum = {0,0}
for i=1,#distance_pair do
    m = distance_pair[i]
    d = (tonumber( m[4] ) / 2.0)*max_bin
    dbin = math.floor(d) + 1
    print ("dbin", dbin)
    if m[3] == 'true' then
        pos_histogram[dbin] = pos_histogram[dbin] + 1
        total_sum[1] = total_sum[1] + 1
    else
        neg_histogram[dbin] = neg_histogram[dbin] + 1
        total_sum[2] = total_sum[2] + 1
    end
end

local final_histogram = {}
for i=1,max_bin do
    table.insert(final_histogram, { tostring((i-1)/max_bin * 2), tostring(pos_histogram[i]), tostring(neg_histogram[i]) } )
end

csvigo.save({path=project .. '_histogram.csv', data=final_histogram})

local xbin = {}
for i=1,max_bin do
    table.insert(xbin, (i-1)/1000*2)
end

print ("xbin", xbin)

x = torch.linspace(-2*math.pi,2*math.pi)
gnuplot.svgfigure(project .. '_histogram.svg')
gnuplot.plot('Cos',x/math.pi,torch.cos(x),'~')
--gnuplot.plot('Postivie', torch.Tensor(xbin), torch.Tensor(pos_histogram), '~')
gnuplot.plot({'Positive', torch.Tensor(xbin), torch.Tensor(pos_histogram)/total_sum[1], '-'}, {'Negative', torch.Tensor(xbin), torch.Tensor(neg_histogram)/total_sum[2], '-'})
gnuplot.plotflush()


