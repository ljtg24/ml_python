import time
import math
import networkx as nx
import pandas as pd
import numpy as np
import sys

from src.clique import Clique
from src.unit import Unit
from src.subspace import Subspace


def get_position_dict(df_data, m, dims):
    min_per_col = list(df_data.min())
    max_per_col = list(df_data.max())
    # print(max_per_col)
    # print(len(min_per_col), len(max_per_col))
    position_dict = dict()

    if len(min_per_col) == dims and len(max_per_col) == dims:
        for j in range(dims):
            min_temp = min_per_col[j]
            add_temp = (max_per_col[j] - min_temp)/m
            for i in range(m):
                arg1 = min_temp+i*add_temp
                arg2 = arg1 + add_temp
                position_dict[j*m+i] = [j, arg1, arg2]
    return position_dict


def init_unit(m, dims):
    all_unit = []
    for j in range(dims):
        for i in range(m):
            u_no = j*m+i  # notice: the order of dims an m must as same as the dictionary of pisition
            cur_pos = set()
            cur_pos.add(u_no)
            # cur_pos = {i: position_dict[i]}
            unit = Unit(u_no=u_no, cur_pos=cur_pos)
            all_unit.append(unit)
    return all_unit


def get_dense_unit(df_data, candidate_list, position_dict, point_nums, ud_threshold):
    dense_dict = dict()
    for unit in candidate_list:
        position_yield = (position_key for position_key in unit.cur_pos)  # generate a generator of condition
        # select_yield = [key_dim for key_dim in select_condition]
        df_temp = df_data.copy()
        # print(select_yield)
        for key_position_yield in position_yield:
            if df_temp.empty:
                # print(unit.u_no)
                continue  # there is no point in this unit
            else:
                select_condition = position_dict[key_position_yield]
                df_temp = df_temp[(select_condition[1] <= df_temp[select_condition[0]]) & (df_temp[select_condition[0]] < select_condition[2])]
                # print('else', unit.u_no, df.shape)
        unit_point_num = df_temp.shape[0]
        if df_temp.empty is False and unit_point_num / point_nums > ud_threshold:
            unit.point_num = unit_point_num  # update the unit's point numbers while satisfy the condition
            # del df_temp
            dense_dict[unit.u_no] = unit
    return dense_dict


def get_subspace(dense_dict, subspace_dim, position_dict):
    subspace_dict = dict()
    subspace_key = 0
    for unit in dense_dict.values():
        flag = False
        unit_dim_no = unit.get_position_dims(position_dict)
        for key, subspace in subspace_dict.items():
            if subspace.space_dim_no == unit_dim_no:
                subspace.units_no.append(unit.u_no)
                flag = True
        if flag is False:
            new_subspace = Subspace(dim=subspace_dim, space_dim_no=unit_dim_no, units_no=list())
            new_subspace.units_no.append(unit.u_no)
            subspace_dict[subspace_key] = new_subspace
            subspace_key += 1
    return subspace_dict


def mdl(subspace_dict, dense_dict):
    length = len(subspace_dict)
    sj_per_subspace = dict()
    # get coverage(sj) of subspace
    for ind, subspace in subspace_dict.items():
        sj = 0
        unit_in_subspace = [dense_dict[unit_no] for unit_no in subspace.units_no]
        for unit in unit_in_subspace:
            sj += unit.point_num
        sj_per_subspace[ind] = sj
    # Sort dict from large to small based on the coverage size(sj)
    sorted_sj_per_subspace = sorted(sj_per_subspace.items(), key=lambda d: d[1], reverse=True)  # type <list>
    # Compute the minimum value of CL
    CL_min = (sys.maxsize, -1)  # select a biggest integer in python3

    for i in range(1, length):
        selected_list = sorted_sj_per_subspace[0: i]
        pruned_list = sorted_sj_per_subspace[i:]
        u_s = sum(item[1] for item in selected_list)/(i+0.0)
        u_p = sum(item[1] for item in pruned_list)/(length-i+0.0)
        selected_CL = 0
        pruned_CL = 0
        for k, sj in selected_list:
            diff = abs(sj - u_s)
            selected_CL += math.log2(diff) if diff != 0 else 0
        for k, sj in pruned_list:
            diff = abs(sj-u_p)
            pruned_CL += math.log2(diff) if diff != 0 else 0
        CL = math.log2(u_s)+math.log2(u_p)+selected_CL+pruned_CL
        if CL < CL_min[0]:
            CL_min = (CL, i)
    # print('i:', CL_min[1])
    selected_list = sorted_sj_per_subspace[0: CL_min[1]]
    selected_units_no = []
    for sub_no, sj in selected_list:
        selected_units_no.extend(subspace_dict[sub_no].units_no)
    selected_units = [dense_dict[unit_no] for unit_no in selected_units_no]
    return selected_units


# add unique new_pos into candidates
def filter_pos(new_pos, candidates):
    for unit in candidates:
        if new_pos == unit.cur_pos:
            return False
    return True


def join_units(dense_unit_list, position_dict):
    candidates = []
    unit_no = 0
    for u1 in dense_unit_list:
        for u2 in dense_unit_list:
            new_pos = u1.add_dim(u2, position_dict)
            if new_pos and filter_pos(new_pos, candidates):
                candidates.append(Unit(unit_no, cur_pos=new_pos))
                unit_no += 1

    return candidates


def get_cluster(dense_dict, position_dict):
    con_G = nx.Graph()

    # 将dense_dict.keys() 作为节点存入con_GS

    # dense_list = dense_dict.values()
    # length = len(dense_list)
    # for i in range(length):
    #     for j in range(i+1, length):
    for ind1, u1 in dense_dict.items():
        for ind2, u2 in dense_dict.items():
            if ind1 < ind2 and u1.judge_connection(u2, position_dict):
                con_G.add_edge(ind1, ind2)
                # print(ind1, ind2)

    clusters = nx.connected_components(con_G)
    return clusters


def get_growth_units(R, data_dim, growth, dense_dict):
    temp_growth_unit_pos = list()
    for r in R:
        unit = dense_dict[r]
        label_index = -1
        for index, pos in enumerate(unit.cur_pos):
            if pos//10 == data_dim and -1 < pos%10+growth < 10:
                label_index = index
        # not exist the growth dim
        if label_index == -1:
            return None
        else:
            cur_pos = set()
            for index, pos in enumerate(unit.cur_pos):
                if label_index != index:
                    cur_pos.add(pos)
                else:
                    cur_pos.add(pos+growth)
        temp_growth_unit_pos.append(cur_pos)
    return temp_growth_unit_pos


def get_growth_units_no(temp_growth_unit_pos, dense_dict, clu):
    temp_growth_units_no = list()
    for pos in temp_growth_unit_pos:
        flag1 = False
        for u_no in clu:
            if pos == dense_dict[u_no].cur_pos:
                # print(pos, dense_dict[u_no].cur_pos)
                temp_growth_units_no.append(u_no)
                flag1 = True
        if flag1 is False:
            return None
    return temp_growth_units_no


def maximal_cover(clusters, dense_dict, data_dims):
    maximal_coverage = list()
    for clu in clusters:
        R_list = dict()
        for u_no in clu:
            u_no_flag = True
            for R1 in R_list.values():
                if u_no in R1:
                    u_no_flag = False
            # select a id of unit that not in R_list
            if u_no_flag:
                R = list()
                R.append(u_no)
                growth = 1
                # Random selection of one dimension for greed-growth
                for data_dim in range(data_dims):
                    # get
                    temp_growth_unit_pos = get_growth_units(R, data_dim, growth, dense_dict)
                    if temp_growth_unit_pos:
                        # get the id of growth units
                        temp_growth_unit_no = get_growth_units_no(temp_growth_unit_pos, dense_dict, clu)
                        if temp_growth_unit_no:
                            R.extend(temp_growth_unit_no)
                            if growth > 0:
                                growth += 1
                            else:
                                growth += -1
                        elif growth > 0:
                            growth = -1
                        else:
                            pass
                    elif growth > 0:
                        growth = -1
                    else:
                        pass
                # accesses each neighbor unit of R for ascertaining whether R is maximal
                # need implement
                R_list[u_no] = R
        maximal_coverage.append(R_list)
    return maximal_coverage


def points_in_unit(df_data, unit, position_dict):
    position_yield = (position_key for position_key in unit.cur_pos)  # generate a generator of condition
    # select_yield = [key_dim for key_dim in select_condition]
    # print(select_yield)
    df_temp = df_data.copy()
    for key_position_yield in position_yield:
        if df_temp.empty:
            # print(unit.u_no)
            continue  # there is no point in this unit
        else:
            select_condition = position_dict[key_position_yield]
            df_temp = df_temp[(select_condition[1] <= df_temp[select_condition[0]]) & (
                        df_temp[select_condition[0]] < select_condition[2])]
            # print('else', unit.u_no, df.shape)
    return list(df_temp.index)


def assign_points(covers, df_data, points_num, dense_dict, position):
    _labels = np.zeros((points_num, ))
    cluster_id = 1
    for cover in covers:
        for unit_no in cover:
            points_id = points_in_unit(df_data, dense_dict[unit_no], position)
            # print(points_id)
            _labels[points_id] = cluster_id
        cluster_id += 1
    return _labels


if __name__ == '__main__':
    start = time.time()
    data_path = './resources/s3.csv'
    coordinate = np.loadtxt(data_path, delimiter=',')[:, 0:2]
    data = np.loadtxt(data_path, delimiter=',')[:, 0:2]
    shape = data.shape
    print(shape)
    point_nums, dims = shape
    ud_threshold = 0.002
    m = 50
    colums_array = range(dims)
    df = pd.DataFrame(data, columns=colums_array)
    print('start')
    position = get_position_dict(df, m, dims)
    print('position')
    candidate_units = init_unit(m, dims)
    print('candidate_units', len(candidate_units))

    dim = 1
    flag = True
    while dim < dims and flag:
        dense_units = get_dense_unit(df, candidate_units, position, point_nums, ud_threshold)
        subspaces = get_subspace(dense_units, 1, position)
        # the dense units after mdl pruned
        # mdl_dense_units = mdl(subspaces, dense_units)
        mdl_dense_units = dense_units.values()
        # del subspaces
        # del dense_units
        candidate_units = join_units(mdl_dense_units, position)
        print(dim, 'dense_unit', len(dense_units), 'subspace', len(subspaces), 'mdl_dense_units', len(mdl_dense_units),
              'candidate_units', len(candidate_units))
        dim += 1
        flag = True if candidate_units else False
    # print('flag', flag)
    dense_units = get_dense_unit(df, candidate_units, position, point_nums, ud_threshold) if flag else dense_units
    print('dense', len(dense_units))
    # for ind, item in dense_units.items():
    #     print(ind, item.cur_pos)
    connected_unit = get_cluster(dense_units, position)  # generator

    # cluster_num = 1
    # for cluster in connected_unit:
    #     print(cluster_num, len(cluster))
    #     cluster_num += 1
    #     for item in cluster:
    #         print(item, dense_units[item].cur_pos)

    # maximal_covers = maximal_cover(connected_unit, dense_units, dims)
    # print("maximal_cover.")
    # for ma in maximal_covers:
    #     for ind, r in ma.items():
    #         print(ind, r, dense_units[ind].cur_pos)
    labels = assign_points(connected_unit, df, point_nums, dense_units, position)
    print(labels)
    # labels = np.reshape(labels, (point_nums, 1))
    # datas = np.hstack((data, labels))
    print(time.time()-start)
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % (n_clusters_ - 1))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        plt.plot(coordinate[my_members, 0], coordinate[my_members, 1], col + '.')
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()