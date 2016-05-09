require "csvigo"
require 'loadutils'
require 'math'
require "gnuplot"

metrics = require 'metrics'
debugger = require 'fb.debugger'
precisionrecall = require 'precisionrecall'
require 'eval'

workspace = 'fashion'
projectlist = { {project='distance_pair_128_att_0411', linestyle='-'}, 
        {project='distance_pair_128_0411', linestyle='-'}, 
        {project='distance_pair_128_att_valid', linestyle='-'}, 
        {project='distance_pair_128_0502', linestyle='-'},
        {project='distance_pair_128_0428', linestyle='-'},
        {project='distance_pair_128_0508', linestyle='-'},
        {project='distance_pair_128_0508_forceince', linestyle='-'},
        {project='distance_pair_1024', linestyle='-'}
    }

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Testing a Siamese Distance network on Fashion')
cmd:text()
cmd:text('==>Options')

opt = cmd:parse(arg or {})

pre_recall_plot = {}
roc_plot = {}

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


for i=1,#projectlist do
    local project = projectlist[i].project
    assert(project ~= nil, 'project must be set')
    local distance_pair_file = project .. '.csv'

    local distance_pair = csvigo.load( {path=distance_pair_file, mode='large'} )

    --print ('#loaded:', #distance_pair)
    local max_bin = 10000
    local pos_histogram = {}
    local neg_histogram = {}

    for i=1,max_bin do
        table.insert(pos_histogram,0)
        table.insert(neg_histogram,0)
    end

    conf = torch.Tensor( #distance_pair )
    label = torch.Tensor( #distance_pair )

    local max_distance = 0
    for i=1,#distance_pair do
        m = distance_pair[i]
        d = tonumber(m[4])
        conf[i] = d
        if m[3] == 'true' then
            label[i] = 1
        else
            label[i] = -1 
        end
        if d > max_distance then
            max_distance = d
        end
    end
    assert(max_distance ~= nil, 'max_distance is nil')

    for i=1,conf:size(1) do
        conf[i] = (max_distance - conf[i])/max_distance
    end

    local legend = projectlist[i].legend or projectlist[i].project
    local linestyle = projectlist[i].linestyle

    --local rec, prec, ap, sortind = precisionrecall(conf, label, 0, 0.001)
    --print ("ap-1", ap)
    local rec, prec, ap, threshold1 = precision_recall(conf, label)
    print ("ap-2", ap)

    local roc_points, threshold = metrics.roc.points(conf, label)
    print (rec:type(), roc_points:type())
    table.insert(roc_plot, {legend, torch.squeeze(roc_points[{{},{1}}]), torch.squeeze(roc_points[{{},{2}}]), linestyle})
    --table.insert(roc_plot, {legend, torch.squeeze(roc_points[{{},{1}}]), torch.squeeze(roc_points[{{},{2}}]):mul(-1):add(1), linestyle})
    --debugger.enter()

    --table.insert(pre_recall_plot, {legend, rec, prec:mul(-1):add(1), linestyle} )
    table.insert(pre_recall_plot, {string.format("%s(ap=%f)", legend, ap), rec, prec, linestyle} )

    --print (rec, prec)

    local total_sum = {0,0}

    for i=1,#distance_pair do
        m = distance_pair[i]
        d = (tonumber( m[4] ) / max_distance)*(max_bin-1)
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
        table.insert(final_histogram, { tostring((i-1)/max_bin * max_distance), tostring(pos_histogram[i]), tostring(neg_histogram[i]) } )
    end

    csvigo.save({path=project .. '_histogram.csv', data=final_histogram})

    local xbin = {}
    for i=1,max_bin do
        table.insert(xbin, (i-1)/max_bin*max_distance)
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
    print ("Max Distance", max_distance)
    print ("KL Divergence", DistKL(pos_histogram, neg_histogram))
end

gnuplot.svgfigure(workspace .. '_precision_recall.svg')
gnuplot.plot(pre_recall_plot)
gnuplot.xlabel('recall')
gnuplot.ylabel('precision')
gnuplot.title('Precision-Recall')
gnuplot.plotflush()

gnuplot.svgfigure(workspace .. '_roc.svg')
gnuplot.plot(roc_plot)
gnuplot.xlabel('false positive rate')
gnuplot.ylabel('true positive rate')
gnuplot.title('Receiver operating characteristic')
gnuplot.plotflush()

os.exit(0)
