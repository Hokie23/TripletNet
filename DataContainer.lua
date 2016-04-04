require 'torch'
require 'dok'


require 'image'
local thread = require 'threads'
local DataContainer = torch.class('DataContainer')

local function CatNumSize(num,size)
    local stg = torch.LongStorage(size:size()+1)
    stg[1] = num
    for i=2,stg:size() do
        stg[i]=size[i-1]
    end
    return stg
end
function DataContainer:__init(...)
    local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a DataContainer ',
    {arg='BatchSize', type='number', help='Number of Elements in each Batch',req = true},
    {arg='TensorType', type='string', help='Type of Tensor', default = 'torch.FloatTensor', req=true},
    {arg='ExtractFunction', type='function', help='function used to extract Data, Label and Info', default= function(...) return ... end},
    {arg='List', type='userdata', help='source of DataContainer', req=false, default=nil},
    {arg='Data', type='userdata', help='Data', req = true},
    {arg='ListGenFunc', type='function', help='Generate new list'},
    {arg='Augment', type='boolean', help='augment data',default=false},
    {arg='BulkImage', type='boolean', help='how to load bulk of list', default=false},
    {arg='Resolution', type='number', help='how to load bulk of list', req=true},
    {arg='LoadImageFunc', type='function', help='how to load bulk of list', req = true},
    {arg='NumEachSet', type='number', help='each set', req=true, default=3}
    )


    --self.CurrentItemMutex = thread.Mutex()
    self.IsEnd = false
    self.BatchSize = args.BatchSize
    self.TensorType = args.TensorType
    self.ExtractFunction = args.ExtractFunction
    self.Augment = args.Augment
    self.Batch = torch.Tensor():type(self.TensorType)
    self.Data = args.Data
    self.List = args.List
    self.ListGenFunc = args.ListGenFunc
    self.NumEachSet = args.NumEachSet
    self.BulkImage = args.BulkImage
    self.Resolution = args.Resolution
    self.LoadImageFunc = args.LoadImageFunc
    self:Reset()
end

function DataContainer:Reset()
    self.CurrentItem = 1
end

function DataContainer:size()
    return #self.List
end

function DataContainer:Reset()
    self.IsEnd = false
    self.CurrentItem = 1
end


function DataContainer:__tostring__()
    local str = 'DataContainer:\n'
    if self:size() > 0 then
        str = str .. ' + num samples : '.. self:size()
    else
        str = str .. ' + empty set...'
    end
    return str
end

--function DataContainer:ShuffleItems()
--    local RandOrder = torch.randperm(self.List.exemplar_list:size(1)):long()
--    self.List = self.List:indexCopy(1,RandOrder,self.List)
--    print('(DataContainer)===>Shuffling Items')
--
--end

function DataContainer:GenerateList(net)
    self.List = self.ListGenFunc(net)
end

function DataContainer:LoadBatch(batchlist)
    local batch = torch.Tensor():type(self.TensorType)
    local size = #batchlist
    nsz = CatNumSize(self.NumEachSet, CatNumSize(size,  self.Resolution))
    batch:resize(nsz)
    for i=1, self.NumEachSet do
        mylist = batchlist[{{i},{}}]
        for j=1,mylist:size(1) do
            local filename = mylist[j].name
            local jitter = mylist[j].jitter
            local img = self.LoadImageFunc(filename, jitter)
            self.Batch[i][j]:copy(img)
        end
    end

    return batch
end

function DataContainer:Lock()
    --self.CurrentItemMutex.lock()
end

function DataContainer:IsContinue()
    return self.IsEnd == false
end

function DataContainer:Unlock()
    --self.CurrentItemMutex.unlock()
end


function DataContainer:GetNextBatch(net)
    self:Lock()
    local size = math.min(self:size()-self.CurrentItem + 1, self.BatchSize)
    if size <= 0 then
        self:Unlock()
        return nil
    end

    local mylist = {}
    for i=0,(size-1) do
        table.insert(mylist,self.List[self.CurrentItem+i])
    end
    self.CurrentItem = self.CurrentItem + size
    self:Unlock()

    return mylist
end
