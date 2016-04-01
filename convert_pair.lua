require 'csvigo'

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('distance pair csv file')
cmd:text()
cmd:text('==>Options')
cmd:option('-distance_pair', '', 'csv file name')
cmd:option('-output', '', 'csv file name')

opt = cmd:parse(arg or {})

imagepairs = csvigo.load( {path=opt.distance_pair, mode='large'} )

output = assert(io.open(opt.output, 'w+'))

output:write(string.format('label\tdistance\n'))
for i=1,#imagepairs do
    local src,dst,p_or_n,dist = unpack(imagepairs[i])
    print (p_or_n, dist)
    local output_label = p_or_n == 'true' and '1' or '0'
    output:write( string.format('%s\t%s\n', output_label, dist) )
end

output:close()
