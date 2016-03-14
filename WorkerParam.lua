local WorkerParam = torch.class('WorkerParam')

function WorkerParam:__init(BatchList, TensorType, Resolution,  LoadImageFunc, NumEachSet)
    self.BatchList = BatchList
    self.TensorType = TensorType
    self.Resolution = Resolution
    self.LoadImageFunc = LoadImageFunc
    self.NumEachSet = NumEachSet
end
