#!/usr/bin/env python
batch_size=256
modelf="456_64layer"
test_mode="64layer"
start=4
srun python3 -u online.py\
 --batchSize ${batch_size}\
 --test_mode ${test_mode}\
 --modelf ${modelf}\
 --start ${start}