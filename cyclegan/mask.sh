#!/usr/bin/env python
batch_size=8
outf="linewidth1_final"
mode="mask"
lambda_mask=0.0002
lambda_cyc=15.0
lambda_perc=7.0
lambda_id=5.0
stage=4
epoch=1
num_img=11
export CUDA_VISIBLE_DEVICE=0,1
srun python3 -u cyclegan.py\
 --lambda_id ${lambda_id}\
 --batch_size ${batch_size}\
 --outf ${outf}\
 --loss_mode ${mode}\
 --epoch ${epoch}\
 --lambda_mask ${lambda_mask}\
 --lambda_cyc ${lambda_cyc}\
 --stage ${stage}\
 --lambda_perc ${lambda_perc}\
 --num_img ${num_img}