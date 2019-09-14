#!/usr/bin/env python
batch_size=1
modelf="32_pow"
test_mode="2layer_512"
start=256
srun python3 -u online.py\
 --batchSize ${batch_size}\
 --test_mode ${test_mode}\
 --modelf ${modelf}\
 --start ${start}