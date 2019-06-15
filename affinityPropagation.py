import numpy as np
import sys


class AffinityPropagation:
    def __init__(self, preference=None, damping=0.5, max_iter=200, convits=10):
        self.preference = preference
        self.damping = damping
        self.max_iter = max_iter
        self.convits = convits

    @staticmethod
    def get_euclidean_distance(pt1, pt2):
        distance = 0
        zipped_pt = zip(pt1, pt2)
        for item in zipped_pt:
            distance += (item[0] - item[1]) ** 2
        return distance**(1/2)

    #  Similarity matrix
    def initialize_s_matrix(self):
        if self.preference is not None and self.preference >= 0:
            raise Exception('preference is a negative number.')
        self.s_matrix = np.zeros((self.data_len, self.data_len))
        for i in range(self.data_len):
            for j in range(self.data_len):
                self.s_matrix[i, j] = -self.get_euclidean_distance(self.X[i], self.X[j])

        if self.preference is None:
            self.preference = np.median(self.s_matrix)  # np.min(self.s_matrix)
            print("preference:", self.preference)

        for k in range(self.data_len):
            self.s_matrix[k, k] = self.preference

    @staticmethod
    def get_2max(matrix_a_s):
        max_list = list()
        for items in matrix_a_s:
            max_temp = [-sys.maxsize, -sys.maxsize + 1]
            for ele in items:
                if ele > max_temp[0]:
                    max_temp[0] = ele
                elif ele > max_temp[1]:
                    max_temp[1] = ele
            max_list.append(max_temp)
        return max_list

    def update_r_matrix(self):
        temp_r = self.r_matrix.copy()
        matrix_a_s = np.add(self.s_matrix, self.a_matrix)
        a_s_2max = self.get_2max(matrix_a_s)
        del matrix_a_s
        # r_matrix = np.zeros((self.data_len, self.data_len))
        for i in range(self.data_len):
            for j in range(self.data_len):
                self.r_matrix[i, j] = self.s_matrix[i, j] - a_s_2max[i][0] if self.s_matrix[i, j] != a_s_2max[i][0] \
                    else self.s_matrix[i, j] - a_s_2max[i][1]
        self.r_matrix = self.damping * temp_r + (1 - self.damping) * self.r_matrix

    def update_a_matrix(self):
        temp_a = self.a_matrix.copy()
        temp_r_matrix = self.r_matrix.copy()
        temp_r_matrix[temp_r_matrix < 0] = 0  ##
        max_row = np.sum(temp_r_matrix, axis=0)

        for k in range(self.data_len):
            r_k_k = self.r_matrix[k, k]
            for i in range(self.data_len):
                if i != k:
                    temp_r_ele = r_k_k + max_row[k] - temp_r_matrix[i, k] - temp_r_matrix[k, k]
                    self.a_matrix[i, k] = min(temp_r_ele, 0)
                else:
                    self.a_matrix[i, k] = max_row[k] - temp_r_matrix[k, k]
        self.a_matrix = self.damping * temp_a + (1 - self.damping) * self.a_matrix

    def fit(self, X):
        self.X = np.array(X)
        self.data_len, dims = self.X.shape

        self.initialize_s_matrix()
        self.a_matrix = np.zeros((self.data_len, self.data_len))
        self.r_matrix = np.zeros((self.data_len, self.data_len))

        iter_num = 0
        convit_num = 0
        labels = np.zeros((self.data_len, ))
        while convit_num < self.convits and iter_num < self.max_iter:

            self.update_r_matrix()
            self.update_a_matrix()

            matrix_a_r = np.add(self.a_matrix, self.r_matrix)
            print("11", np.flatnonzero(np.diag(matrix_a_r) > 0).__len__())
            new_labels = np.argmax(matrix_a_r, axis=1)
            print(np.unique(new_labels).__len__())
            iter_num += 1

            if (labels == new_labels).all():
                convit_num += 1
            else:
                convit_num = 0
                labels = new_labels

            # print(convit_num, iter_num)

        self.labels_ = new_labels
        self.n_clusters_ = len(set(self.labels_))