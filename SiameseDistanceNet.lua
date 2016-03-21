require 'nn'
require 'nngraph'

local SiameseDistanceNet, parent = torch.class('nn.SiameseDistanceNet', 'nn.gModule')

local function CreateSiameseDistanceNet(EmbeddingNet, distMetric, inputs)
    local embeddings = {}
    local outputs = {}
    local nets = {EmbeddingNet}
    print("Create SiameseDistanceNet of EmbeddingNet #1:", EmbeddingNet)
    print("Create SiameseDistanceNet of EmbeddingNet #2:", nets)
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

    dists = nn.View(-1,1)( distMetric:clone()( {embeddings[1], embeddings[2]} ) )
    return nets, dists, embeddings
end

function SiameseDistanceNet:__init(EmbeddingNet, distMetric, collectFeat)
    self.num = 2  -- must be 2
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

        local nets, net_outputs, embeddings = CreateSiameseDistanceNet(net, self.distMetric, currInputs)
        currInputs = {}
        for i=1,self.num do
            if not self.nets[i] then self.nets[i] = {} end
            table.insert(self.nets[i], nets[i])
            table.insert(currInputs, embeddings[i])
        end

        table.insert(outputs, net_outputs)
        start_layer = end_layer+1
    end

    -- triplet net input and output definition
    parent.__init(self, inputs, outputs)
end

function SiameseDistanceNet:shareWeights()
    --for j=1,#self.nets[i] do
    --    self.nets[1]:share( self.EmbeddingNet,'weight','bias','gradWeight','gradBias','running_mean','running_std')
    --end
    for i=1,self.num-1 do
          for j=1,#self.nets[i] do
            self.nets[i+1][j]:share(self.nets[1][j],'weight','bias','gradWeight','gradBias','running_mean','running_std')
          end
    end
end


function SiameseDistanceNet:type(t)
    parent.type(self, t)
    self:shareWeights()
end
