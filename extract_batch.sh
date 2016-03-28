#!/bin/bash

CUDA_VISIBLE_DEVICES=1 th MainExtractFeature.lua -output_list ./result.out -model './Results/WedMar2314:56:452016/Embedding.t715'
