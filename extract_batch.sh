#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 th MainExtractFeature.lua -output_list ./result_128D.out -model './Results/FriApr817:32:232016/tripletnet.best.t7'
CUDA_VISIBLE_DEVICES=3 th MainExtractFeature.lua -output_list ./result_128D_shoes.out -model './Results/WedApr2017:12:262016/tripletnet.best.t7'
