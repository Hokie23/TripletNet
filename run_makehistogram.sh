#!/bin/bash

CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_512
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_1024
