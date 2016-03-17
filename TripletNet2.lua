require 'nn'
require 'nngraph'
local TripletNet2, parent = torch.class('nn.TripletNet2', 'nn.gModule')


local function CreateTripletNet2(EmbeddingNet, inputs)
    local embeddings = {}
    local outputs = {}
    local nets = {EmbeddingNet}
    print("CreateTripleNet of EmbeddingNet #1:", EmbeddingNet)
    print("CreateTripleNet of EmbeddingNet #2:", nets)
    local num = #inputs
    for i=1,num do
        if i < num then
            nets[i+1] = nets[1]:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
        end
        local n = nets[i](inputs[i])
        embeddings[i] = n
        print ("embeddings[i]", embeddings[i])
        outputs[i] = n
    end
    return nets, outputs, embeddings
end

--function TripletNet2:updateOutput(input, target)
--    --print ("updateOutput(input) size=:", input)
--    local output = parent.updateOutput(self,input, target)
--    -- print ("updateOutput(target):", target)
--    --print ("updateOutput(output):", output:size())
--    -- print ("updateOutput(output)=:", output[1][1], output[1][2])
--    return output
--end

function TripletNet2:__init(EmbeddingNet, num, distMetric, collectFeat)
--collectFeat is of for {{layerNum = number, postProcess = module}, {layerNum = number, postProcess = module}...}
    self.num = num or 3
    self.distMetric = distMetric or nn.PairwiseDistance(2)
    self.EmbeddingNet = EmbeddingNet
    self.nets = {}
    local collectFeat = collectFeat or {{layerNum = #self.EmbeddingNet}}
    local inputs = {}
    local outputs = {}
    local dists

    for i=1,self.num do
      inputs[i] = nn.Identity()()
    end

    local start_layer = 1
    local currInputs = inputs
    for f=1,#collectFeat do
        local end_layer = collectFeat[f].layerNum
        local net = nn.Sequential()
        for l=start_layer,end_layer do
            net:add(self.EmbeddingNet:get(l))
        end

        local nets, net_outputs, embeddings = CreateTripletNet2(net, currInputs)
        currInputs = {}
        for i=1,self.num do
            if not self.nets[i] then self.nets[i] = {} end
            table.insert(self.nets[i], nets[i])
            table.insert(currInputs, embeddings[i])
            table.insert(outputs, net_outputs[i])
        end

        start_layer = end_layer+1
    end

    -- print("TripleNet:", outputs)
    --print("TripleNet:", #collectFeat)
    --print("TripleNet self.num:", self.num)
    --print("TripleNet inputs:", inputs)
    --print("TripleNet outputs:", outputs)

    -- triplet net input and output definition
    parent.__init(self, inputs, outputs)
end

function TripletNet2:shareWeights()
    for i=1,self.num-1 do
          for j=1,#self.nets[i] do
            self.nets[i+1][j]:share(self.nets[1][j],'weight','bias','gradWeight','gradBias','running_mean','running_std')
          end
    end
end


function TripletNet2:type(t)
    parent.type(self, t)
    self:shareWeights()
end
