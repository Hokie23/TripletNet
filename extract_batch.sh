#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 th MainExtractFeature.lua -output_list ./result_128D.out -model './Results/FriApr817:32:232016/tripletnet.best.t7'
#CUDA_VISIBLE_DEVICES=3 th MainExtractFeature.lua -output_list ./result_128D_shoes.out -model './Results/ThuApr2115:34:432016/tripletnet.best.t7'
CUDA_VISIBLE_DEVICES=3 th MainExtractFeature.lua -output_list ./result_128D_shoes.out -model './Results/ThuApr2115:34:432016/Embedding.t7best.embedding.model.t7'
