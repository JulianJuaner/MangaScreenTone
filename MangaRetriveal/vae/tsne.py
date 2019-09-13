
from datasets import *
from sklearn.manifold import TSNE
import time

 
def TSNEFeature(folder, outf, N=3):
    images = make_dataset(os.path.join(folder, 'feature512NC'))
    books = os.listdir(os.path.join(folder,'simline'))
    books.sort()
    
    start = time.time()
    tsne = TSNE(n_components=N)

    for book in books:
        try:
            os.makedirs(os.path.join(folder, outf, book), exist_ok=True)
        except:
            pass

    for i in range(len(images)):
        feature = np.load(images[i]).reshape((512, -1)).transpose()
        feature = tsne.fit_transform(feature).transpose().reshape((N, 73, 51))
        if i%1 == 0:
            print(i)
        np.save(os.path.join(folder, outf, images[i].split('/')[-2],
		                     images[i].split('/')[-1]), feature)

    print('time used:', time.time() - start)

def countZeros(feature):
    feature[feature<10]=0
    print(np.min(feature), np.max(feature))
    print(np.count_nonzero(feature), np.count_nonzero(feature.reshape(512, -1), axis = 1))

if __name__ == '__main__':
    Folder = '../../../../data/manga'
    OUTF = 'tsne3'
    #TSNEFeature(Folder, OUTF)
    images = make_dataset(os.path.join(Folder, 'feature512NC'))
    #books = os.listdir(os.path.join(Folder,'simline'))
    #books.sort()
    for i in range(len(images)):
        feature = np.load(images[i])
        countZeros(feature)
    pass
