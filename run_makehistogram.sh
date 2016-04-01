#!/bin/bash

CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_valid
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_0328_norm
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_1024_norm
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_0328
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_512
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_512_a
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_1024
