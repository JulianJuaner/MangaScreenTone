#!/usr/bin/env python
batch_size=8
outf="manga_newA_2_4_3_15_10_5_10"
mode="mask"
run_mode="test00"
lambda_mask=0.0003
lambda_cyc=10.0
lambda_perc=10.0
lambda_id=7.0
stage=4
epoch=19
num_img=6
srun python3 -u cyclegan.py\
 --run_mode ${run_mode}\
 --batch_size ${batch_size}\
 --outf ${outf}\
 --loss_mode ${mode}\
 --epoch ${epoch}\
 --lambda_mask ${lambda_mask}\
 --lambda_cyc ${lambda_cyc}\
 --stage ${stage}\
 --lambda_perc ${lambda_perc}\
 --num_img ${num_img}