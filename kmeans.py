import random
import numpy as np


class KMeans:
    def __init__(self, k, input_data, init='kmeans', kernel=None, iterations=100, threshold=0.01):
        self.k = k
        self.input_data = input_data
        self.iterations = iterations
        self.threshold = threshold
        self.init = init
        self.kernel = kernel

    def get_original_center(self, data_length):
        center_pts = list()
        temp_list = list(range(data_length))
        if self.init == 'kmeans':
            center_pts_ind = random.sample(temp_list, self.k)
            center_pts = self.input_data[center_pts_ind]
        if self.init == 'kmeans++':
            # print('kmeans++')
            center_pts = list()
            center_pts.append(self.input_data[random.choice(temp_list)])
            i = 1
            while i < self.k:
                distance_list = list()
                distance_sum = 0
                for point in self.input_data:
                    distances = [self.get_distance(point, pt2) for pt2 in center_pts]
                    min_distance = min(distances)
                    distance_list.append(min_distance)
                    distance_sum += min_distance
                random_number = random.random()

                prev = 0
                for ind, distance in enumerate(distance_list):
                    cur_probability = prev + distance/distance_sum
                    if cur_probability >= random_number > prev:
                        center_pts.append(self.input_data[ind])
                        break  # find the center point
                    prev = cur_probability
                i += 1
        return center_pts

    @staticmethod
    def get_distance(pt1, pt2):
        distance = 0
        zipped_pt = zip(pt1, pt2)
        for item in zipped_pt:
            distance += (item[0]-item[1]) ** 2
        return distance

    @staticmethod
    def get_min_distance(distances):
        min_distance = distances[0]
        for distance_info in distances:
            if distance_info[1] < min_distance[1]:
                min_distance = distance_info
        return min_distance[0]

    @staticmethod
    def get_center_points(attribution_dict):
        center_points = list()
        for key, value in attribution_dict.items():
            point = np.mean(np.array(value), axis=0)
            center_points.append(point)
        return center_points

    def judge_flag(self, center_pts1, center_pts2):
        flag = False
        for item in zip(center_pts1, center_pts2):
            # if self.get_distance(item[0], item[1]) > self.threshold:
            if self.get_distance(item[0], item[1]) != 0:
                flag = True
                break
        return flag

    def get_labels(self, pt_ind_dict, data_length):
        _labels = np.zeros((data_length, ))
        for key, value in pt_ind_dict.items():
            _labels[value] = key+1

        self.labels_ = _labels

    def run(self):
        data_length = len(self.input_data)
        if self.k > data_length:
            raise Exception('Invalid k')

        center_pts = self.get_original_center(data_length)
        # print(center_pts)
        flag = True
        iter_num = 0
        pt_ind_dict = dict()
        while flag and iter_num < self.iterations:
            attribution_dict = dict()
            pt_ind_dict.clear()
            for pt_ind, point in enumerate(self.input_data):
                distances = [(ind, self.get_distance(point, pt)) for ind, pt in enumerate(center_pts)]
                attribution = self.get_min_distance(distances)
                if attribution in attribution_dict:
                    attribution_dict[attribution].append(point)
                    pt_ind_dict[attribution].append(pt_ind)
                else:
                    attribution_dict[attribution] = [point]
                    pt_ind_dict[attribution] = [pt_ind]
            temp_center_pts = self.get_center_points(attribution_dict)

            flag = self.judge_flag(temp_center_pts, center_pts)
            center_pts = temp_center_pts

            iter_num +=1
        # print('iter_num', iter_num)
        self.get_labels(pt_ind_dict, data_length)
