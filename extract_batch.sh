#!/bin/bash

CUDA_VISIBLE_DEVICES=0 th MainExtractFeature.lua -output_list ./result_128D.out -model './Results/FriApr817:32:232016/tripletnet.best.t7'
