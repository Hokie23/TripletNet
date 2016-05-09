#!/bin/bash

#run_statics_gpu3.sh
#cal_ret=$?
#if [$cal_ret -ne 0]; then
#    echo 'failed to make histogram files'
#    exit 1
#fi
CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua 

#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_0428
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_0411
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_att
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_att_0411
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_att_valid_a
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_att_valid_b
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_att_valid_c
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_att_valid_d
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_att_valid_f
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_valid
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_train
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_relu_valid
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_0328_norm
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_1024_norm
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128_0328
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_512
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_512_a
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_128
#CUDA_VISIBLE_DEVICES=0 th MainMakeHistogram.lua -project distance_pair_1024


if [$? -ne 0]; then
    echo 'Failed to make histogram'
    exit 2
fi
echo 'Congratulation! Finished making histogram'
exit 0
