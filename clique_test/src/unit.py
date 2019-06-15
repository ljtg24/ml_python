class Unit:
    def __init__(self, u_no, cur_pos=None, point_num=0):
        # u_num is the id number of unit,
        # last_pos is the position of unit at k-1 dimension
        # cur_pos is the current position of current dimension
        # point_num is the point numbers in unit
        # self.last_pos = last_pos
        self.u_no = u_no
        self.cur_pos = cur_pos  # set()
        self.point_num = point_num

    def get_cur_pos(self):
        return self.cur_pos

    def get_position_dims(self, position_dict):
        return set(position_dict[pos_no][0] for pos_no in self.cur_pos)

    def add_dim(self, unit, position_dict):
        diff_pos_1 = self.cur_pos.difference(unit.cur_pos)
        diff_pos_2 = unit.cur_pos.difference(self.cur_pos)
        if len(diff_pos_1) == 1 and len(diff_pos_2) == len(diff_pos_1):
            diff_pos_1 = diff_pos_1.pop()
            diff_pos_2 = diff_pos_2.pop()
            if position_dict[diff_pos_1][0] < position_dict[diff_pos_2][0]:
                return self.cur_pos.union(unit.cur_pos)
        return None

    def judge_connection(self, unit, position_dict):
        if self.cur_pos == unit.cur_pos:
            return True
        diff_pos_1 = self.cur_pos.difference(unit.cur_pos)
        diff_pos_2 = unit.cur_pos.difference(self.cur_pos)
        if len(diff_pos_1) == 1 and len(diff_pos_2) == len(diff_pos_1):
            diff_pos_1 = diff_pos_1.pop()
            diff_pos_2 = diff_pos_2.pop()
            # diff_dim_1 = position_dict[diff_pos_1]
            # diff_dim_2 = position_dict[diff_pos_2]
            # 浮点数的精度不同
            # if diff_dim_1[0] == diff_dim_2[0] and (diff_dim_1[1] == diff_dim_2[2] or diff_dim_1[2] == diff_dim_1[1]):
            if abs(diff_pos_2-diff_pos_1) == 1:
                # print(self.u_no, unit.u_no, diff_pos_1, diff_pos_2)
                # print(position_dict[diff_pos_1], position_dict[diff_pos_2])
                return True
        return False
