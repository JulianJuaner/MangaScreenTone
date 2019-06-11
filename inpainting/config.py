LOSS_OVERALL = False

EPOCH = 10
LR = 1e-4
BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
IMGSIZE = (512, 512)
#Test with gaobr result.
GABOR = False
FILENUM = 1000
CUDA = True
TEST_MODE = False

#Train
ROOTDIR = './data/rawdata/'
INPUT = './data/input/'
OUTPUT = './data/valid/'
MASKDIR = './data/mask/'
SCTDIR = './data/screentone/'

#Test
T_ROOTDIR = './data/test/rawdata/'
T_INPUT = './data/test/input/'
T_OUTPUT = './data/test/valid/'
T_MASKDIR = './data/test/mask/'
T_SCTDIR = './data/test/screentone/'

if GABOR:
    INPUT = './data/gabor/input/'
    OUTPUT = '../../PCA/PCAresult/'

    T_ROOTDIR = './data/gabor/test/rawdata/'
    T_INPUT = './data/gabor/test/input/'
    T_OUTPUT = './data/gabor/test/output/'
