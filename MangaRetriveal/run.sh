#!/usr/bin/env python
batch_size=256
modelf="norm_32_resize"
test_mode="32resize_layer"
start=9
srun python3 -u online.py\
 --batchSize ${batch_size}\
 --test_mode ${test_mode}\
 --modelf ${modelf}\
 --start ${start}