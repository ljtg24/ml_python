import time
import numpy as np
from src.clique import Clique

if __name__ == '__main__':
    start = time.time()
    data_path = './resources/Spiral.csv'
    data = np.loadtxt(data_path, delimiter=',')[:, 0:2]
    # data = np.loadtxt(data_path, delimiter=' ')
    point_nums, dimensions = data.shape
    ud_threshold = 0.01
    m = 10
    clique = Clique(m, ud_threshold, point_nums, dimensions)
    clique.process(data)
    labels = clique._labels
    print(time.time() - start)

    n_clusters_ = clique.n_clusters_

    print("number of estimated clusters : %d" % (n_clusters_))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_+1), colors):
        my_members = labels == k
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()