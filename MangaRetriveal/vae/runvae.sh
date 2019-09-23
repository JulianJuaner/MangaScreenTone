#!/usr/bin/env python
#Weighted: 287, EXP: 284
#32_pow: 256
#32: 256
#24, 31
batch_size=1
modelf="norm_64"
start=1
lr=0.00003
mode='train'
srun python3 -u model.py\
 --lr ${lr}\
 --mode ${mode}\
 --batchSize ${batch_size}\
 --outf  ${modelf}\
 --start ${start}