class Subspace:
    def __init__(self, dim=0, space_dim_no=None, units_no=None):
        self.dim = dim  # 维数
        self.space_dim_no = space_dim_no  # 维数对应的维号 set()
        self.units_no = units_no  # 子空间中的unit编号 list()
