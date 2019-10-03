#!/usr/bin/env python
#Weighted: 287, EXP: 284
#32_pow: 256
#32: 256
#25
#24, 31
#48_no_rm: 40
batch_size=1
modelf="456_64layer"
start=4
lr=0.00003
mode='test'
srun python3 -u model.py\
 --lr ${lr}\
 --mode ${mode}\
 --batchSize ${batch_size}\
 --outf  ${modelf}\
 --start ${start}