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

--print ('#loaded:', #distance_pair)
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
    --print ("dbin", dbin)
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

local xbin = torch.Tensor(xbin)
local pos_histogram = torch.Tensor(pos_histogram)/total_sum[1]
local neg_histogram = torch.Tensor(neg_histogram)/total_sum[2]
--print ("xbin", xbin)

gnuplot.svgfigure(project .. '_histogram.svg')
gnuplot.plot({'Positive', xbin, pos_histogram, '-'}, {'Negative', xbin, neg_histogram, '-'})
gnuplot.plotflush()

local p_mu = xbin * pos_histogram
local n_mu = xbin * neg_histogram

print ("project", project, "p_mu", p_mu, "n_mu", n_mu, 'diff', n_mu - p_mu)

function _DKL(p, q)
    local cPQ = torch.Tensor( q:size(1))
    local cQ = q:clone()
    local cP = p:clone()

    local i = 0
    cPQ:map2(cP, cQ, function (x, p, q)
                if p== 0 or q == 0 then
                    return 0
                else
                    return torch.log(p/q)
                end
            end)

    local s = cP * cPQ
    return s
end

function DistKL(p, q)
    return _DKL(p,q) + _DKL(q,p)
end

print ("KL Divergence", DistKL(pos_histogram, neg_histogram))

