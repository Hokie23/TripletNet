
local debugger = require( 'fb.debugger')
local PairwiseDistanceOffset, parent = torch.class('nn.PairwiseDistanceOffset', 'nn.PairwiseDistance')

function PairwiseDistanceOffset:__init(p)
    parent.__init(self)

    -- state
    self.gradInput = {}
    self.diff = torch.Tensor()
    self.norm = p
    self.margin = 0.005

    input = nn.Identity()()
    input1 = nn.SelectTable(1)(input)
    input2 = nn.SelectTable(2)(input)
    x1 = nn.SelectTable(1)(input1)
    x2 = nn.SelectTable(1)(input2)
    b1 = nn.SelectTable(2)(input1)
    b2 = nn.SelectTable(2)(input2)

    d = nn.PairwiseDistance(2)( {x1, x2} )
    b = nn.CAddTable()({b1,b2})
    c = nn.CSubTable()({d,b})
    c = nn.AddConstant(-self.margin,false)({c})
    y = nn.Clamp(0,2)({c})
    self.gModule = nn.gModule({input},{y})
end

function PairwiseDistanceOffset:updateOutput(inputset)
    return self.gModule:updateOutput(inputset)
end

function PairwiseDistanceOffset:updateGradInput(inputset, gradOutput)
    return self.gModule:updateGradInput(inputset,gradOutput)
end

function PairwiseDistanceOffset:clearState()
    nn.utils.clear(self, 'diff', 'outExpand', 'grad', 'ones')
    --self.gModule.clearState()
    return parent.clearState(self)
end
