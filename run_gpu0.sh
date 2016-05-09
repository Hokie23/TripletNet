#!/bin/bash

#CUDA_VISIBLE_DEVICES=3 th Main.lua -load /data1/fantajeon/torch/TripletNet/Results/FriMar1817:20:512016/Weights.t718 
CUDA_VISIBLE_DEVICES=3 th Main.lua -port 8801 -load './Result/WedApr2714:53:202016/Weights.t7optim.w.t79' -distance_ratio 0.01
#CUDA_VISIBLE_DEVICES=3 th Main.lua -load /data1/fantajeon/torch/TripletNet/Results/MonApr417:43:532016/Weights.t9
#CUDA_VISIBLE_DEVICES=1 th Main.lua
