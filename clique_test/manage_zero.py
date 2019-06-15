import time
import math
import pandas as pd
import numpy as np
import sys
from src.unit import Unit
from src.subspace import Subspace


def get_position_dict(df_data, m, dims):
    min_per_col = list(df_data.min())
    max_per_col = list(df_data.max())
    # print(max_per_col)
    # print(len(min_per_col), len(max_per_col))
    position = dict()

    if len(min_per_col) == dims and len(max_per_col) == dims:
        for j in range(dims):
            min_temp = min_per_col[j]
            add_temp = (max_per_col[j] - min_temp)/m
            for i in range(m):
                arg1 = min_temp+i*add_temp
                arg2 = arg1 + add_temp
                position[i*dims+j] = [arg1, arg2]
    return position


def init_unit(m, dims, position):
    all_unit = []
    for j in range(dims):
        for i in range(m):
            u_no = i*dims+j
            cur_pos = {j: position[u_no]}
            unit = Unit(u_no=u_no, cur_pos=cur_pos)
            all_unit.append(unit)
    return all_unit


def get_dense_unit(df_datas, candidate_units, point_nums, ud):
    dense_unitss = dict()
    for unit in candidate_units:
        select_condition = unit.cur_pos if unit.last_pos == -1 \
            else dict(unit.last_pos.items()+unit.cur_pos.items())
        select_yield = ((key_dim, pos) for key_dim, pos in select_condition.items())  # generate a generator of condition
        # select_yield = [key_dim for key_dim in select_condition]
        df = df_datas.copy()
        # print(select_yield)
        for key_dim_yield, pos_yield in select_yield:
            # print(unit.u_no, df.shape)
            if df.empty:
                # print(unit.u_no)
                continue  # there is no point in this unit
            else:
                # print(key_dim_yield, pos_yield)
                df = df[(pos_yield[0] <= df[key_dim_yield]) & (df[key_dim_yield] < pos_yield[1])]
                # print('else', unit.u_no, df.shape)
        unit_point_num = df.shape[0]
        if df.empty is False and unit_point_num / (point_nums+0.0) > ud:
            unit.point_num = unit_point_num  # update the unit's point numbers while satisfy the condition
            # del df
            dense_unitss[unit.u_no] = unit
    return dense_unitss


def get_subspace(dense_unitss, subspace_dim):
    subspace_dict = dict()
    subspace_key = 0
    for unit in dense_unitss.values():
        flag = False
        unit_position = unit.cur_pos if unit.last_pos == -1 \
            else dict(unit.last_pos.items()+unit.cur_pos.items())
        unit_dim_no = set(unit_position.keys())
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


def get_mdl_dense_units(unit_dict, mdl_dense_units_list):
    return [unit_dict[unit_no] for unit_no in mdl_dense_units_list]


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
    sorted_sj_per_subspace = sorted(sj_per_subspace.items(), key=lambda d:d[1], reverse=True)  # type <list>
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
            selected_CL += math.log2(abs(sj-u_s))
        for k, sj in pruned_list:
            pruned_CL += math.log2(abs(sj-u_p))
        CL = math.log2(u_s)+math.log2(u_p)+selected_CL+pruned_CL
        if CL < CL_min[0]:
            CL_min = (CL, i)
    selected_list = sorted_sj_per_subspace[0: CL_min[1]]
    selected_units_no = []
    selected_units_no = [selected_units_no.extend(subspace_dict[sub_no].units_no) for sub_no, sj in selected_list]
    selected_units = get_mdl_dense_units(dense_dict, selected_units_no)
    return selected_units


def join_units(dense_unit_list):
    candidates = []
    for u1 in dense_unit_list:
        for u2 in dense_unit_list:
            if u1.last_pos == u2.last_pos and u1.cur_pos.keys()[0] < u2.cur_pos.keys()[0]:
                pass

    return candidates


if __name__ == '__main__':
    start = time.time()
    data_path = './resources/face.csv'
    data = np.loadtxt(data_path)
    shape = data.shape
    print(shape)
    point_nums, dims = shape
    ud_threshold = 0.02
    m = 10
    colums_array = range(dims)
    df_data = pd.DataFrame(data, columns=colums_array)
    # print(df_data)
    position = get_position_dict(df_data, m, dims)

    candidate_units = init_unit(m, dims, position)
    print(len(candidate_units))
    dense_units = get_dense_unit(df_data, candidate_units, point_nums, ud_threshold)
    print('dense_unit', len(dense_units))
    subspaces = get_subspace(dense_units, 1)
    print('subspace', len(subspaces))
    # the dense units after mdl pruned
    mdl_dense_units = mdl(subspaces, dense_units)
    del dense_units
    candidate_units = join_units(mdl_dense_units)

    # dim = 1
    # flag = True
    # while dim < dims and flag:
    #     get_dense_unit(df_data, candidate_units, point_nums)
    #     dim += 1
    #     flag = False
    print(time.time()-start)