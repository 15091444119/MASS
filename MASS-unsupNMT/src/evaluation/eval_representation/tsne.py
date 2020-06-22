import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg') # plot on server

def tsne(array, labels, save_path):
    model = TSNE(n_iter=1000) 
    Y = model.fit_transform(array) 
    plt.scatter(Y[:,0], Y[:,1], c=labels)
    plt.savefig(save_path)
    plt.close()

def test_tsne():
    a = np.array([[1, 2, 3], [2, 3, 4]])
    tsne(a, [1, 2], "./tmp")

if __name__ == "__main__":
   test_tsne()
    

