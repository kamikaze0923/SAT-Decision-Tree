import numpy as np
import os
from util.dt import DT
from util.solver import DT_Solver


EXP = 4
MAX_NODES = 9


if __name__ == "__main__":
    data_x = np.load(os.path.join("data", "weather", "x.npy"))
    data_y = np.load(os.path.join("data", "weather", "y.npy"))
    # data_x = np.array([[1,0,1,0],[1,0,0,1],[0,0,1,0],[1,1,0,0],[0,0,0,1],[1,1,1,1],[0,1,1,0],[0,0,1,1]])
    # data_x = data_x[[0,2],:2]
    # data_y = np.array([0,0,1,0,1,0,0,1])
    # data_y = data_y[[0,2]]

    # data_x = np.array([[0,1],[1,1]])
    # data_y = np.array([0,1])

    N_FEATURES = data_x.shape[1]
    USE_AUTHOR_ENCODING = False
    print("MAX_NODES: %d" % MAX_NODES)
    print("N_FEATURES: %d" % N_FEATURES)

    solver = DT_Solver(MAX_NODES, N_FEATURES, (data_x, data_y), USE_AUTHOR_ENCODING)
    solver.init_solver()
    dt = DT(MAX_NODES, N_FEATURES, USE_AUTHOR_ENCODING)
    solution_set = set()
    while True:
        solution = solver.solve_for_one()
        if solution is not None:
            print(solution)
            if solution in solution_set:
                raise Exception("Duplicated Solution!")
            solution_set.add(solution)
            dt.parse_solution(solution)
            dt.validate(data_x, data_y)
            dt.draw(len(solution_set))
        else:
            break









