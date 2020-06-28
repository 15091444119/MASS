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

def bilingual_tsne_with_sentences(src_rep, tgt_rep, src_sentences, tgt_sentences, plot_num, save_prefix):
    """
    Params:
        src_rep: 2d np array, source representations [bs, dim]
        tgt_rep: 2d np array, targeet representations [bs, dim]
        src_sentences: source sentences
        tgt_sentences: target sentenecs
        plot_num: num of sentences to plot(we only plot the first plot_num sentence, but calculate tsne using all the data)
        save_prefix: picture will be saved in save_prefix.png
    """
    # calculate tsne on concat data
    all_rep = np.concatenate((src_rep, tgt_rep), axis=0)
    model = TSNE(n_iter=1000) 
    low_dim_rep = model.fit_transform(all_rep)

    src_plotted_rep = low_dim_rep[:plot_num]
    tgt_plotted_rep = low_dim_rep[len(src_sentences):len(src_sentences) + plot_num]

    for i in range(plot_num):
        x = src_plotted_rep[i, 0]
        y = src_plotted_rep[i, 1]
        plt.scatter(x, y, c='b', marker=".")
        plt.annotate(str(i), xy = (x, y), xytext = (x + 0.1, y + 0.1))

    for i in range(plot_num):
        x = tgt_plotted_rep[i, 0]
        y = tgt_plotted_rep[i, 1]
        plt.scatter(x, y, c='r', marker=".")
        plt.annotate(str(i), xy = (x, y), xytext = (x + 0.1, y + 0.1))
    plt.savefig(save_prefix)
    plt.close()

def test_bilingual_tsne():
    src_rep = np.array([[1, 2], [2, 3]])
    tgt_rep = np.array([[1, 1], [2, 2]])
    src_sentencens = ["12中文", "23"]
    tgt_sentences = ["11", "22"]
    bilingual_tsne_with_sentences(src_rep, tgt_rep, src_sentencens, tgt_sentences, 2, "./2")
    bilingual_tsne_with_sentences(src_rep, tgt_rep, src_sentencens, tgt_sentences, 1, "./1")


if __name__ == "__main__":
   #test_tsne()
   test_bilingual_tsne()
    

