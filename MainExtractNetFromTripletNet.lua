require 'TripletNet'
require 'AttentionLSTM'


network_file = './Results/WedApr2017:12:262016/Embedding.t7best.tripletnet.t7'
output_file = './Results/WedApr2017:12:262016/tripletnet.best.t7'
n = torch.load(network_file)
torch.save(output_file, n.nets[1])
