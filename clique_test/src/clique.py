import math
import sys
import networkx as nx
import pandas as pd
import numpy as np
from src.subspace import Subspace
from src.unit import Unit


class Clique:
    def __init__(self, m, ud, point_nums, dims, pruned=False, max_cover=False):
        # m, divide every dimension to m segments
        # ud, the threshold to judge a unit whether dense
        self.m = m
        self.ud = ud
        self.dims = dims
        self.point_nums = point_nums
        self.min_points = math.ceil(ud*point_nums)
        self.position = None
        self.df = None
        self.pruned = pruned
        self.max_cover = max_cover

    def get_position_dict(self):
        min_per_col = list(self.df.min())
        max_per_col = list(self.df.max())
        # print(max_per_col)
        # print(len(min_per_col), len(max_per_col))
        position_dict = dict()

        if len(min_per_col) == self.dims and len(max_per_col) == self.dims:
            for j in range(self.dims):
                min_temp = min_per_col[j]
                add_temp = (max_per_col[j] - min_temp) / self.m
                for i in range(self.m):
                    arg1 = min_temp + i * add_temp
                    arg2 = arg1 + add_temp
                    position_dict[j * self.m + i] = [j, arg1, arg2]
        self.position = position_dict

    def init_unit(self):
        all_unit = []
        for j in range(self.dims):
            for i in range(self.m):
                u_no = j * self.m + i  # notice: the order of dims an m must as same as the dictionary of pisition
                cur_pos = set()
                cur_pos.add(u_no)
                # cur_pos = {i: position_dict[i]}
                unit = Unit(u_no=u_no, cur_pos=cur_pos)
                all_unit.append(unit)
        return all_unit

    def get_dense_unit(self, candidate_list):
        dense_dict = dict()
        for unit in candidate_list:
            position_yield = (position_key for position_key in unit.cur_pos)  # generate a generator of condition
            # select_yield = [key_dim for key_dim in select_condition]
            df_temp = self.df.copy()
            # print(select_yield)
            for key_position_yield in position_yield:
                if df_temp.empty:
                    # print(unit.u_no)
                    continue  # there is no point in this unit
                else:
                    select_condition = self.position[key_position_yield]
                    df_temp = df_temp[(select_condition[1] <= df_temp[select_condition[0]]) & (
                                df_temp[select_condition[0]] < select_condition[2])]
                    # print('else', unit.u_no, df.shape)
            unit_point_num = df_temp.shape[0]
            if df_temp.empty is False and unit_point_num >= self.min_points:
                unit.point_num = unit_point_num  # update the unit's point numbers while satisfy the condition
                # del df_temp
                dense_dict[unit.u_no] = unit
        return dense_dict

    def get_subspace(self, dense_dict, subspace_dim):
        subspace_dict = dict()
        subspace_key = 0
        for unit in dense_dict.values():
            flag = False
            unit_dim_no = unit.get_position_dims(self.position)
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

    @staticmethod
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
            u_s = sum(item[1] for item in selected_list) / (i + 0.0)
            u_p = sum(item[1] for item in pruned_list) / (length - i + 0.0)
            selected_CL = 0
            pruned_CL = 0
            for k, sj in selected_list:
                diff = abs(sj - u_s)
                selected_CL += math.log2(diff) if diff != 0 else 0
            for k, sj in pruned_list:
                diff = abs(sj - u_p)
                pruned_CL += math.log2(diff) if diff != 0 else 0
            CL = math.log2(u_s) + math.log2(u_p) + selected_CL + pruned_CL
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
    @staticmethod
    def filter_pos(new_pos, candidates):
        for unit in candidates:
            if new_pos == unit.cur_pos:
                return False
        return True

    def join_units(self, dense_unit_list):
        candidates = []
        unit_no = 0
        for u1 in dense_unit_list:
            for u2 in dense_unit_list:
                new_pos = u1.add_dim(u2, self.position)
                if new_pos and self.filter_pos(new_pos, candidates):
                    candidates.append(Unit(unit_no, cur_pos=new_pos))
                    unit_no += 1

        return candidates

    def get_cluster(self, dense_dict):
        con_G = nx.Graph()

        # 将dense_dict.keys() 作为节点存入con_G

        # dense_list = dense_dict.values()
        # length = len(dense_list)
        # for i in range(length):
        #     for j in range(i+1, length):
        for ind1, u1 in dense_dict.items():
            for ind2, u2 in dense_dict.items():
                if ind1 < ind2 and u1.judge_connection(u2, self.position):
                    con_G.add_edge(ind1, ind2)
                    # print(ind1, ind2)

        clusters = nx.connected_components(con_G)
        return clusters

    @staticmethod
    def get_growth_units(R, data_dim, growth, dense_dict):
        temp_growth_unit_pos = list()
        for r in R:
            unit = dense_dict[r]
            label_index = -1
            for index, pos in enumerate(unit.cur_pos):
                if pos // 10 == data_dim and -1 < pos % 10 + growth < 10:
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
                        cur_pos.add(pos + growth)
            temp_growth_unit_pos.append(cur_pos)
        return temp_growth_unit_pos

    @staticmethod
    def get_growth_units_no(temp_growth_unit_pos, dense_dict, clu):
        temp_growth_units_no = list()
        for pos in temp_growth_unit_pos:
            for u_no in clu:
                if pos == dense_dict[u_no].cur_pos:
                    temp_growth_units_no.append(u_no)
                else:
                    return None
        return temp_growth_units_no

    def judge_growth_id(self, g_id, growth_step):
        if (growth_step < 0 and (g_id+growth_step+1) % self.m == 0) or \
                (growth_step > 0 and (g_id+growth_step) % self.m == 0) or (g_id+growth_step) < 0:
            return False
        else:
            return True

    @staticmethod
    def temp_in_unCovered(unCovered, temp_pos, dense_dict):
        for u_no in unCovered:
            if temp_pos == dense_dict[u_no].cur_pos:
                return u_no
        return False

    def maximal_cover(self, clusters, dense_dict):
        maximal_coverage = list()
        for clu in clusters:
            unCovered = set(clu)
            cover = dict()

            for u_no in clu:
                if u_no not in unCovered:
                    continue
                unCovered.remove(u_no)
                region_to_grow = set()
                max_region = set()
                region_to_grow.add(u_no)
                unit_pos = dense_dict[u_no].cur_pos
                for g_id in unit_pos:
                    for growth_step in [-1, 1]:
                        temp_region = set()
                        growth_id = g_id
                        growth_flag = True
                        while growth_flag:
                            if self.judge_growth_id(growth_id, growth_step) is False:
                                growth_flag = False
                                break
                            growth_id += growth_step
                            for grow_u in region_to_grow:
                                grow_u_pos = dense_dict[grow_u].cur_pos
                                temp_pos = grow_u_pos.copy()
                                temp_pos.remove(g_id)
                                temp_pos.add(growth_id)

                                temp_u_no = self.temp_in_unCovered(unCovered, temp_pos, dense_dict)
                                if temp_u_no:
                                    temp_region.add(temp_u_no)
                                else:
                                    growth_flag = False
                            if growth_flag:
                                max_region = max_region.union(temp_region)
                                unCovered = unCovered.difference(temp_region)
                                # for no in temp_region:
                                #     unCovered.remove(no)
                                temp_region.clear()
                    region_to_grow.union(max_region)
                cover[u_no] = region_to_grow
            maximal_coverage.append(cover)
        return maximal_coverage

        # def maximal_cover(self, clusters, dense_dict):
        # maximal_coverage = list()
        # for clu in clusters:
        #     R_list = dict()
        #     for u_no in clu:
        #         u_no_flag = True
        #         for R1 in R_list.values():
        #             if u_no in R1:
        #                 u_no_flag = False
        #         # select a id of unit that not in R_list
        #         if u_no_flag:
        #             R = list()
        #             R.append(u_no)
        #             growth = 1
        #             # Random selection of one dimension for greed-growth
        #             for data_dim in range(data_dims):
        #                 # get
        #                 temp_growth_unit_pos = self.get_growth_units(R, data_dim, growth, dense_dict)
        #                 if temp_growth_unit_pos:
        #                     # get the id of growth units
        #                     temp_growth_unit_no = self.get_growth_units_no(temp_growth_unit_pos, dense_dict, clu)
        #                     if temp_growth_unit_no:
        #                         R.extend(temp_growth_unit_no)
        #                         if growth > 0:
        #                             growth += 1
        #                         else:
        #                             growth += -1
        #                     elif growth > 0:
        #                         growth = -1
        #                     else:
        #                         pass
        #                 elif growth > 0:
        #                     growth = -1
        #                 else:
        #                     pass
        #             # accesses each neighbor unit of R for ascertaining whether R is maximal
        #             # need implement
        #             R_list[u_no] = R
        #     maximal_coverage.append(R_list)
        # return maximal_coverage

    def points_in_unit(self, unit):
        position_yield = (position_key for position_key in unit.cur_pos)  # generate a generator of condition
        # select_yield = [key_dim for key_dim in select_condition]
        # print(select_yield)
        df_temp = self.df.copy()
        for key_position_yield in position_yield:
            if df_temp.empty:
                # print(unit.u_no)
                continue  # there is no point in this unit
            else:
                select_condition = self.position[key_position_yield]
                df_temp = df_temp[(select_condition[1] <= df_temp[select_condition[0]]) & (
                        df_temp[select_condition[0]] < select_condition[2])]
                # print('else', unit.u_no, df.shape)
        return list(df_temp.index)

    def assign_points(self, covers, dense_dict):
        _labels = np.zeros((self.point_nums,))
        cluster_id = 1
        for cover in covers:
            for unit_no in cover:
                points_id = self.points_in_unit(dense_dict[unit_no])
                # print(points_id)
                _labels[points_id] = cluster_id
            cluster_id += 1
        labels_unique = [item for item in np.unique(_labels) if item != 0]
        n_clusters_ = len(labels_unique)

        self._labels = _labels
        self.n_clusters_ = n_clusters_
# return clustering result as list of indexes, for example
# [ [0, 1, 2, 3], [4, 5, 6, 7] ].

    # algorithm implementation
    def process(self, X):
        colums_array = range(self.dims)
        df = pd.DataFrame(X, columns=colums_array)
        # print(df)
        self.df = df
        self.get_position_dict()
        candidate_units = self.init_unit()
        # print('candidate_units', len(candidate_units))
        dim = 1
        flag = True
        print('start while')
        while dim < self.dims and flag:
            dense_units = self.get_dense_unit(candidate_units)
            subspaces = self.get_subspace(dense_units, 1)
            # the dense units after mdl pruned
            if self.pruned:
                mdl_dense_units = self.mdl(subspaces, dense_units)
            else:
                mdl_dense_units = dense_units.values()
            # del subspaces
            # del dense_units
            candidate_units = self.join_units(mdl_dense_units)
            print(dim, 'dense_unit', len(dense_units), 'subspace', len(subspaces), 'mdl_dense_units',
                  len(mdl_dense_units),
                  'candidate_units', len(candidate_units))
            dim += 1
            flag = True if candidate_units else False

        dense_units = self.get_dense_unit(candidate_units) if flag else dense_units
        print('dense', len(dense_units))
        # for ind, item in dense_units.items():
        #     print(ind, item.cur_pos)
        results = self.get_cluster(dense_units)  # generator

        if self.max_cover:
            max_coverage = self.maximal_cover(results, dense_units)
            # for

        self.assign_points(results, dense_units)
        # for ma in maximal_covers:
        #     for ind, r in ma.items():
        #         print(ind, r)
