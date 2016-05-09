require 'TripletNet'
require 'AttentionLSTM'


network_file = './Results/TueMay315:55:572016/Embedding.t7best.tripletnet.t7'
output_file = './Results/TueMay315:55:572016/tripletnet.best.t7'
weight_file = './Results/TueMay315:55:572016/Weights.t7optim.w.t797'
n = torch.load(network_file)

if weight_file ~= nil then
    local w1 = torch.load(weight_file)
    w = n:getParameters()
    w:copy(w1)
end
torch.save(output_file, n.nets[1])
