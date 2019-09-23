#!/usr/bin/env python
batch_size=256
modelf="48_norm"
test_mode="norm_layer"
start=30
srun python3 -u online.py\
 --batchSize ${batch_size}\
 --test_mode ${test_mode}\
 --modelf ${modelf}\
 --start ${start}