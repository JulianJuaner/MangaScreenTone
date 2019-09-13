#!/usr/bin/env python
batch_size=1
outf="manga_newA_2_4_3_15_10_5_10"
run_mode="test"
mode="mask"
lambda_mask=0.0003
lambda_cyc=15.0
lambda_perc=10.0
lambda_id=5.0
stage=4
epoch=15
num_img=10
srun python3 -u cyclegan.py \
 --run_mode ${run_mode}  \
  --lambda_id ${lambda_id} \
 --batch_size ${batch_size} \
 --outf ${outf} \
 --loss_mode ${mode} \
 --epoch ${epoch} \
 --lambda_mask ${lambda_mask} \
 --lambda_cyc ${lambda_cyc} \
 --stage ${stage} \
 --lambda_perc ${lambda_perc} \
 --num_img ${num_img}
