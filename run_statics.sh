#!/bin/bash

#CUDA_VISIBLE_DEVICES=3 th MainStaticsDistance.lua -load /data1/fantajeon/torch/TripletNet/Results/FriMar1817:20:512016/Weights.t718 
#CUDA_VISIBLE_DEVICES=3 th MainStaticsDistance.lua -load "/data1/fantajeon/torch/TripletNet/Results/ThuApr716:46:042016/Embedding.t71"
#CUDA_VISIBLE_DEVICES=3 th MainStaticsDistance.lua
CUDA_VISIBLE_DEVICES=2 th MainStaticsDistance.lua  -load "/data1/fantajeon/torch/TripletNet/Results/ThuApr718:01:002016/Weights.t7best.tripletnet.w.t7"
#CUDA_VISIBLE_DEVICES=1 th MainStaticsDistance.lua
