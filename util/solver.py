from pysmt.shortcuts import *
from pysmt.oracles import get_logic
import numpy as np
from math import ceil, floor


class DT_Solver(object):
    def __init__(self, N, K, data, use_author_encoding=False):
        def add_key(key, syb):
            if syb in self.tree_keys_look_up:
                self.tree_keys.add(key)
            self.all_keys.add(key)

        def init_vars_1d(arr, syb):
            for i in range(1, arr.shape[0]):
                arr[i] = Symbol(syb + "(" + str(i) + ")")
                add_key(arr[i], syb)

        def init_vars_2d(arr, syb):
            if syb != 'p':
                for i in range(1, arr.shape[0]):
                    if syb !='l' and syb != 'r':
                        for j in range(1, arr.shape[1]):
                            arr[i,j] = Symbol(syb + "(" + str(i) + ',' + str(j) + ")")
                            add_key(arr[i,j], syb)
                    else:
                        if syb == 'l':
                            for j in self.LR(i):
                                arr[i,j] = Symbol(syb + "(" + str(i) + ',' + str(j) + ")")
                                add_key(arr[i,j], syb)

                        elif syb == 'r':
                            for j in self.RR(i):
                                arr[i,j] = Symbol(syb + "(" + str(i) + ',' + str(j) + ")")
                                add_key(arr[i,j], syb)
            else:
                for i in range(1, arr.shape[1]):
                    for j in self.LR(i) | self.RR(i):
                        arr[j,i] = Symbol(syb + "(" + str(j) + ',' + str(i) + ")")
                        add_key(arr[j,i], syb)


        def init_vars_additional(arr, syb):
            if syb == 'lmd':
                for i in range(1, arr.shape[1]):
                    for t in range(floor(i/2)+1):
                        arr[t,i] = Symbol(syb + "(" + str(t) + ',' + str(i) + ")")

            elif syb == 'tao':
                for i in range(1, arr.shape[1]):
                    for t in range(i+1):
                        arr[t,i] = Symbol(syb + "(" + str(t) + ',' + str(i) + ")")


        self.N = N
        self.K = K
        self.tree_keys = set()
        self.use_author_encoding = use_author_encoding

        self.tree_keys_look_up = ('l','r','a','c','c0','c1')
        self.all_keys = set()
        self.tree_keys = set()

        self.LR = lambda i: set(filter((lambda y: y % 2 == 0), [j for j in range(i + 1, min(2 * i, N - 1) + 1)]))
        self.RR = lambda i: set(filter((lambda y: y % 2 == 1), [j for j in range(i + 2, min(2 * i + 1, N) + 1)]))

        self.var_v = np.empty(shape=N+1, dtype=object)
        init_vars_1d(self.var_v, 'v')
        self.var_l = np.empty(shape=(N+1, N+1), dtype=object)
        init_vars_2d(self.var_l, 'l')
        self.var_r = np.empty(shape=(N+1, N+1), dtype=object)
        init_vars_2d(self.var_r, 'r')
        self.var_p = np.empty(shape=(N+1, N), dtype=object)
        init_vars_2d(self.var_p, 'p')

        self.var_a = np.empty(shape=(K+1, N+1), dtype=object)
        init_vars_2d(self.var_a, 'a')
        self.var_u = np.empty(shape=(K+1, N+1), dtype=object)
        init_vars_2d(self.var_u, 'u')
        self.var_d0 = np.empty(shape=(K+1, N+1), dtype=object)
        init_vars_2d(self.var_d0, 'd0')
        self.var_d1 = np.empty(shape=(K+1, N+1), dtype=object)
        init_vars_2d(self.var_d1, 'd1')


        if self.use_author_encoding:
            self.var_c = np.empty(shape=N+1, dtype=object)
            init_vars_1d(self.var_c, 'c')
        else:
            self.var_c0 = np.empty(shape=N+1, dtype=object)
            init_vars_1d(self.var_c0, 'c0')
            self.var_c1 = np.empty(shape=N+1, dtype=object)
            init_vars_1d(self.var_c1, 'c1')

        self.var_lmd = np.empty(shape=(floor(N/2)+1,N+1), dtype=object)
        init_vars_additional(self.var_lmd, 'lmd')
        self.var_tao = np.empty(shape=(N+1,N+1), dtype=object)
        init_vars_additional(self.var_tao, 'tao')

        self.p_instance_feature = set()
        self.n_instance_feature = set()
        for x,y in zip(*data):
            if y == 1:
                self.p_instance_feature.add(tuple(x))
            else:
                self.n_instance_feature.add(tuple(x))

        self.solver = None
        self.fomula = None
        self.num_solution = None

    def encode_constraint(self):
        def cons_1(self):
            f = Not(self.var_v[1])
            return [f]

        def cons_2(self):
            all_f = []
            for i in range(1, self.N+1):
                for j in self.LR(i):
                    f = self.var_v[i].Implies(Not(self.var_l[i,j]))
                    all_f.append(f)
            return all_f

        def cons_3(self):
            all_f = []
            for i in range(1, self.N+1):
                for j in self.LR(i):
                    f = Iff(self.var_l[i,j], self.var_r[i,j+1])
                    all_f.append(f)
            return all_f

        def cons_4(self):
            all_f = []
            for i in range(1, self.N+1):
                sum_f = []
                for j in self.LR(i):
                    sum_f.append(self.var_l[i,j])
                f = Not(self.var_v[i]).Implies(ExactlyOne(sum_f))
                all_f.append(f)
            return all_f

        def cons_5(self):
            all_f = []
            for i in range(1, self.N):
                for j in self.LR(i):
                    f = Iff(self.var_p[j,i], self.var_l[i,j])
                    all_f.append(f)
                for j in self.RR(i):
                    f = Iff(self.var_p[j,i], self.var_r[i,j])
                    all_f.append(f)
            return all_f

        def cons_6(self):
            all_f = []
            for j in range(2, self.N+1):
                sum_f = []
                for i in range(floor(j/2), min(j-1,self.N)+1):
                    if j in self.LR(i) | self.RR(i):
                        sum_f.append(self.var_p[j,i])
                f = ExactlyOne(sum_f)
                all_f.append(f)
            return all_f

        def cons_7(self):
            all_f = []
            for r in range(1, self.K+1):
                for j in range(2, self.N+1):
                    all_or = []
                    for i in range(floor(j/2), j):
                        if j in self.LR(i) | self.RR(i):
                            all_or.append(And(self.var_p[j,i], self.var_d0[r,i]))
                            if j in self.RR(i):
                                all_or.append(And(self.var_a[r,i], self.var_r[i,j]))
                    f = Iff(self.var_d0[r,j], Or(all_or))
                    all_f.append(f)
                f = Not(self.var_d0[r,1])
                all_f.append(f)
            return all_f

        def cons_8(self):
            all_f = []
            for r in range(1, self.K+1):
                for j in range(2, self.N+1):
                    all_or = []
                    for i in range(floor(j/2), j):
                        if j in self.LR(i) | self.RR(i):
                            all_or.append(And(self.var_p[j,i], self.var_d1[r,i]))
                            if j in self.LR(i):
                                all_or.append(And(self.var_a[r,i], self.var_l[i,j]))
                    f = Iff(self.var_d1[r,j], Or(all_or))
                    all_f.append(f)
                f = Not(self.var_d1[r,1])
                all_f.append(f)
            return all_f

        def cons_9(self):
            all_f = []
            for r in range(1, self.K+1):
                for j in range(1, self.N+1):
                    all_and = []
                    all_or = []
                    for i in range(floor(j/2), j):
                        if i != 0 and j in self.LR(i) | self.RR(i):
                            all_and.append(And(self.var_u[r,i], self.var_p[j,i]).Implies(Not(self.var_a[r,j])))
                            all_or.append(And(self.var_u[r,i], self.var_p[j,i]))
                    f = And(all_and)
                    all_f.append(f)
                    f = Iff(self.var_u[r,j], Or(self.var_a[r,j], Or(all_or)))
                    all_f.append(f)
            return all_f

        def cons_10(self):
            all_f = []
            for j in range(1, self.N+1):
                sum_f = []
                for r in range(1, self.K+1):
                    sum_f.append(self.var_a[r,j])
                f = Not(self.var_v[j]).Implies(ExactlyOne(sum_f))
                all_f.append(f)
            return all_f

        def cons_11(self):
            all_f = []
            for j in range(1, self.N+1):
                all_and = []
                for r in range(1, self.K+1):
                    all_and.append(Not(self.var_a[r,j]))
                f = self.var_v[j].Implies(And(all_and))
                all_f.append(f)
            return all_f

        def cons_12(self):
            all_f = []
            for feature in self.p_instance_feature:
                for j in range(1, self.N+1):
                    all_or = []
                    for r in range(1, self.K+1):
                        idx = r - 1
                        if feature[idx] == 1:
                            all_or.append(self.var_d1[r,j])
                        elif feature[idx] == 0:
                            all_or.append(self.var_d0[r,j])
                        else:
                            raise Exception("Error")
                    if self.use_author_encoding:
                        f = And(self.var_v[j], Not(self.var_c[j])).Implies(Or(all_or))
                    else:
                        f = And(self.var_v[j], self.var_c0[j]).Implies(Or(all_or))
                    all_f.append(f)
            return all_f

        def cons_13(self):
            all_f = []
            for feature in self.n_instance_feature:
                for j in range(1, self.N+1):
                    all_or = []
                    for r in range(1, self.K+1):
                        idx = r - 1
                        if feature[idx] == 1:
                            all_or.append(self.var_d1[r,j])
                        elif feature[idx] == 0:
                            all_or.append(self.var_d0[r,j])
                        else:
                            raise Exception("Error")
                    if self.use_author_encoding:
                        f = And(self.var_v[j], self.var_c[j]).Implies(Or(all_or))
                    else:
                        f = And(self.var_v[j], self.var_c1[j]).Implies(Or(all_or))
                    all_f.append(f)
            return all_f

        def my_cons_14(self):
            all_f = []
            for j in range(1, self.N + 1):
                all_f.append(Iff(Not(self.var_v[j]), And(Not(self.var_c0[j]), Not(self.var_c1[j]))))
                all_f.append(Iff(self.var_v[j], ExactlyOne(self.var_c0[j], self.var_c1[j])))
            return all_f


        def add_cons_1(self):
            all_f = []
            for i in range(1, self.N+1):
                all_f.append(self.var_lmd[0,i])
                all_f.append(self.var_tao[0,i])
                for t in range(1, floor(i/2)+1):
                    # print("lmd", t, i)
                    if floor((i-1)/2) < t:
                        all_f.append(Iff(self.var_lmd[t,i], Or(self.var_v[i])))
                    else:
                        all_f.append(Iff(self.var_lmd[t,i], Or(self.var_lmd[t,i-1],And(self.var_lmd[t-1,i-1], self.var_v[i]))))
                for t in range(1, i+1):
                    # print("tao", t, i)
                    if i-1 < t:
                        all_f.append(Iff(self.var_tao[t,i], Or(Not(self.var_v[i]))))
                    else:
                        all_f.append(Iff(self.var_tao[t,i], Or(self.var_tao[t,i-1],And(self.var_tao[t-1,i-1], Not(self.var_v[i])))))
            return all_f

        def add_cons_2(self):
            all_f = []
            for i in range(1, self.N+1):
                for t in range(1, floor(i/2)+1):
                    # print("lmd", t, i, 2*(i-t+1), 2*(i-t+1)+1)
                    if 2*(i-t+1) < self.N:
                        all_f.append(self.var_lmd[t,i].Implies(And(Not(self.var_l[i,2*(i-t+1)]), Not(self.var_r[i,2*(i-t+1)+1]))))
            return all_f

        def add_cons_3(self):
            all_f = []
            for i in range(1, self.N+1):
                for t in range(ceil(i/2)+1, i+1):
                    if 2*(t-1) in self.LR(i) and 2*t-1 in self.RR(i):
                        all_f.append(self.var_tao[t,i].Implies(And(Not(self.var_l[i,2*(t-1)]), Not(self.var_r[i,2*t-1]))))
            return all_f

        all_enc = [cons_1, cons_2, cons_3, cons_4, cons_5, cons_6, cons_7, cons_8, cons_9, cons_10,
                   cons_11, cons_12, cons_13, add_cons_1, add_cons_2, add_cons_3]#
        if not self.use_author_encoding:
            all_enc.append(my_cons_14)

        all_f = []
        for func in all_enc:
            print(func)
            fs = func(self)
            all_f.append(And(fs))
        formula = And(all_f)
        return formula

    def init_solver(self):
        self.formula = self.encode_constraint()
        target_logic = get_logic(self.formula)
        print("Target Logic: %s" % target_logic)
        self.solver = Solver(logic=target_logic)
        self.solver.add_assertion(self.formula)
        self.num_solution = 0


    def solve_for_one(self):
        while self.solver.solve():
            partial_model = [EqualsOrIff(k, self.solver.get_value(k)) for k in self.all_keys]
            solution = [(str(k), self.solver.get_value(k).is_true()) for k in self.tree_keys]
            self.solver.add_assertion(Not(And(partial_model)))
            self.num_solution += 1
            # print(self.num_solution)
            return tuple(solution)
        return None





