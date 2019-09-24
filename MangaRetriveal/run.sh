#!/usr/bin/env python
batch_size=256
modelf="norm_64_final"
test_mode="layer456_norm_"
start=25
srun python3 -u online.py\
 --batchSize ${batch_size}\
 --test_mode ${test_mode}\
 --modelf ${modelf}\
 --start ${start}