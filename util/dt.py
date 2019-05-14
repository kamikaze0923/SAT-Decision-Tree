import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt


class DT(object):
    def __init__(self, N, K, use_author_encoding=False):
        self.N = N
        self.K = K
        self.LR = lambda i: set(filter((lambda y: y % 2 == 0), [j for j in range(i + 1, min(2 * i, N - 1) + 1)]))
        self.RR = lambda i: set(filter((lambda y: y % 2 == 1), [j for j in range(i + 2, min(2 * i + 1, N) + 1)]))
        self.l = np.full(shape=(N+1,N+1), fill_value=False, dtype=np.bool)
        self.r = np.full(shape=(N+1,N+1), fill_value=False, dtype=np.bool)
        self.a = np.full(shape=(K+1, N+1), fill_value=False, dtype=np.bool)
        self.use_author_encoding = use_author_encoding
        if self.use_author_encoding:
            self.c = np.full(shape=N+1, fill_value=False, dtype=np.bool)
        else:
            self.c0 = np.full(shape=N+1, fill_value=False, dtype=np.bool)
            self.c1 = np.full(shape=N+1, fill_value=False, dtype=np.bool)

    def create(self, parent=1):
        left_child = []
        for j in self.LR(parent):
            if self.l[parent,j] == True:
                left_child.append(j)
        right_child = []
        for j in self.RR(parent):
            if self.r[parent,j] == True:
                right_child.append(j)
        assert (len(left_child) == 1 and len(right_child) == 1) or (len(left_child) == 0 and len(right_child) == 0)
        if len(left_child) == 0 and len(right_child) == 0:
            if self.use_author_encoding:
                return (parent,(),()), ('1' if self.c[parent] == 1 else '0',(),())
            else:
                assert self.c0[parent] != self.c1[parent]
                return (parent, (), ()), ('1' if self.c1[parent] == 1 else '0', (), ())
        elif len(left_child) == 1 and len(right_child) == 1:
            childLeftInfo = self.create(left_child[0])
            childRightInfo = self.create(right_child[0])
            return (parent, childLeftInfo[0], childRightInfo[0]), (np.argmax(self.a[:,parent]), childLeftInfo[1], childRightInfo[1])
        else:
            raise Exception("Error")

    def print_tree(self):
        print("r:")
        print(np.argwhere(self.r == True))
        print("l:")
        print(np.argwhere(self.l == True))
        print("a:")
        print(np.argwhere(self.a == True))
        print("c:")
        print(np.argwhere(self.c == True))
        tree, ass = self.create(1)
        print(tree, ass)



    def parse_solution(self, solution):
        for var, boolean in solution:
            lit = var[1:var.find("(")]
            idx = var[var.find("(")+1:var.find(")")]
            if self.use_author_encoding:
                lit_look_up = {'l': self.l, 'r': self.r, 'a': self.a, 'c': self.c}
                assert lit in lit_look_up
                if lit == 'c':
                    assert "," not in idx
                    lit_look_up[lit][int(idx)] = boolean
                else:
                    assert "," in idx
                    i, j = idx.split(',')
                    lit_look_up[lit][int(i), int(j)] = boolean
            else:
                lit_look_up = {'l': self.l, 'r': self.r, 'a': self.a, 'c0': self.c0, 'c1': self.c1}
                assert lit in lit_look_up

                if lit == 'c0' or lit == 'c1':
                    assert "," not in idx
                    lit_look_up[lit][int(idx)] = boolean
                else:
                    assert "," in idx
                    i, j = idx.split(',')
                    lit_look_up[lit][int(i), int(j)] = boolean


    def validate(self, data_x, data_y):
        for x,y in zip(data_x, data_y):
            node = 1
            while True:
                node_feature = np.argmax(self.a[:,node])
                check_idx = node_feature - 1
                if x[check_idx] == 0:
                    possible = [j for j in self.LR(node) if self.l[node,j] == True]
                else:
                    possible = [j for j in self.RR(node) if self.r[node, j] == True]
                assert len(possible) == 1
                node = possible[0]
                if self.use_author_encoding:
                    has_feature = [self.a[r,node] for r in range(1, self.K+1)]
                    if sum(has_feature) == 0:
                        if y == 1:
                            assert self.c[node] == True
                        else:
                            assert self.c[node] == False
                        break
                else:
                    if self.c0[node] == True:
                        assert y == 0
                        break
                    elif self.c1[node] == True:
                        assert y == 1
                        break


    def draw(self, cnt):
        tree, ass = self.create()
        nx_tree = nx.Graph()
        nx_tree_label = {}

        def extract_info(tree, ass):
            c, l, r = tree
            c_a, c_l, c_r = ass

            nx_tree.add_node(c)

            if l == ():
                assert r == ()
                nx_tree_label[c] = 'T' if c_a == '1' else 'F'
            else:
                assert r != ()
                nx_tree_label[c] = c_a
                nx_tree.add_edge(c, l[0])
                nx_tree.add_edge(c, r[0])
                extract_info(l, c_l)
                extract_info(r, c_r)
        extract_info(tree, ass)

        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        pos = graphviz_layout(nx_tree, prog='dot')
        nx.draw(nx_tree, pos, with_labels=True, node_color='blue')
        plt.subplot(1,2,2)
        pos = graphviz_layout(nx_tree, prog='dot')
        nx.draw(nx_tree, pos, with_labels=True, labels=nx_tree_label, node_color='red')
        plt.savefig('trees/' + str(cnt) + '.png')
        plt.close()



