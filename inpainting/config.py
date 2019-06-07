
EPOCH = 1
LR = 1e-4
BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
IMGSIZE = (512, 512)
#Test with gaobr result.
GABOR = True
FILENUM = 200
CUDA = True

#Train
ROOTDIR = './data/rawdata/'
INPUT = './data/input/'
OUTPUT = './data/valid/'
MASKDIR = './data/mask/'

#Test
T_ROOTDIR = './data/test/rawdata/'
T_INPUT = './data/test/input/'
T_OUTPUT = './data/test/valid/'
T_MASKDIR = './data/test/mask/'

if GABOR:
    INPUT = './data/gabor/input/'
    OUTPUT = '../../PCA/PCAresult/'

    T_ROOTDIR = './data/gabor/test/rawdata/'
    T_INPUT = './data/gabor/test/input/'
    T_OUTPUT = './data/gabor/test/output/'
