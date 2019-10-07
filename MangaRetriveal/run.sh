#!/usr/bin/env python
batch_size=1
modelf="Image_feat"
test_mode="Ilayer"
start=20
srun python3 -u online.py\
 --batchSize ${batch_size}\
 --test_mode ${test_mode}\
 --modelf ${modelf}\
 --start ${start}