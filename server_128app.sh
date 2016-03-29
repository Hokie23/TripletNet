#!/bin/bash

CUDA_VISIBLE_DEVICES=2 th MainLoadFeature.lua -feature_list ./result_128D.out -model './Results/WedMar2314:56:452016/Embedding.t715'
