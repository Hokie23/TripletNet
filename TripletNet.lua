require 'nn'
require 'nngraph'
local TripletNet, parent = torch.class('nn.TripletNet', 'nn.gModule')


local shared_parameters = {'weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_std', 'running_var'}
local function CreateTripletNet(EmbeddingNet, inputs, distMetric)
  local embeddings = {}
  local dists = {}
  local nets = {EmbeddingNet}
  local num = #inputs
  for i=1,num do
      if i < num then
          nets[i+1] = nets[1]:clone(unpack( shared_parameters))
      end
      embeddings[i] = nets[i](inputs[i])
  end
  local embedMain = embeddings[1]

  for i=1,num-1 do
      table.insert(dists, nn.View(-1,1)(distMetric:clone()({embedMain,embeddings[i+1]})) )
  end
  return nets, dists, embeddings
end

function TripletNet:__init(EmbeddingNet, num, distMetric, collectFeat)
    self.num = num or 3
    self.distMetric = distMetric or nn.PairwiseDistance(2)
    self.EmbeddingNet = EmbeddingNet
    self.nets = {}

    local inputs = {}
    local outputs = {}
    local dists

    for i=1,self.num do
      inputs[i] = nn.Identity()()
    end

    local start_layer = 1
    local net = EmbeddingNet:clone( unpack( shared_parameters) )

    local nets, dists, embeddings = CreateTripletNet(net, inputs, self.distMetric)
    for i=1,self.num do
        table.insert(self.nets, nets[i])
    end

    print("TripletNet net:", nets)
    print("TripletNet dist:", dists)
    outputs = nn.JoinTable(2)(dists)

    -- print("TripleNet:", outputs)
    print("TripleNet self.num:", self.num)
    print("TripleNet dists:", dists)
    print("TripleNet inputs:", inputs)
    print("TripleNet outputs:", outputs)

    parent.__init(self, inputs, {outputs})
end

function TripletNet:shareWeights()
    --for i=1,self.num-1 do
    --      for j=1,#self.nets[i] do
    --        self.nets[i+1][j]:share(self.nets[1][j], unpack(shared_parameters))
    --      end
    --end
end


function TripletNet:type(t)
    parent.type(self, t)
    self:shareWeights()
end
