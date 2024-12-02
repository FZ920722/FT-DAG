#!/usr/bin/python3
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # #
# Create Time: 2023/9/1317:42
# Fang YJ
# Real-Time Systems Group
# Hunan University HNU
# # # # # # # # # # # # # # # #

from MainHead import *

# GNM DAG生成算法基础穷举（基于边组合的穷举）：
# 可控参数：
#   - m ： DAG的边数量；

# def generate_upper_triangular_matrices(n):
#     for i in range(2 ** (n * (n - 1) // 2)):
#         matrix = []
#         count = 0
#         for row in range(n):
#             matrix.append([0] * n)
#             for col in range(row, n):
#                 if row == col:
#                     matrix[row][col] = 0  # 对角元素始终为 0
#                 elif count == i % 2 ** (row + 1):
#                     matrix[row][col] = 1
#                 count += 1
#         yield matrix


def GNM_Dag_Generator(_n, _md, _mu):
    # S1 所有结点；
    __node_set = set(__i for __i in range(_n))
    # S2 所有边；
    __edge_set = set((__y, __x) for __x in range(_n) for __y in range(__x))
    # S3 满足约束条件的边组合；
    for __mnum in range(_md, _mu + 1):
        if __mnum <= len(__edge_set):
            for __edges in combinations(__edge_set, __mnum):
                __rdag = nx.DiGraph()
                __rdag.add_nodes_from(__node_set)
                __rdag.add_edges_from(__edges)
                yield __rdag

def networkx_gnp(_n, _p):
    while True:
        temp_dag = nx.gnp_random_graph(n=_n, p=_p, directed=True)
        if nx.is_directed_acyclic_graph(temp_dag):
            return temp_dag

__edges_num = 4
for _n in range(3, 100):
    st = time.time()
    # for _mu in range(int(_n * _n / 2)):
        # for _md in range(0, _mu + 1):
            # print(f"{_n}_{_mu}_{_md}")
    # _md, _mu = int(_n * (_n -1)/ 2) - 1, int(_n * (_n -1)/ 2) - 1
    # _md, _mu = int(_n * _n / 4) - 1, int(_n * _n / 4) - 1
    _md, _mu = 0, __edges_num
    # _md, _mu = 0, int(_n * (_n - 1) / 2)
    # _md, _mu = int(_n / 2), int(_n / 2)
    dag_num = 0
    for dagx in GNM_Dag_Generator(_n, _md, _mu):
        # print(dagx.edges())
        dag_num += 1
        assert _md <= dagx.number_of_edges() <= _mu
    et = time.time()
    print(f"n:{_n}\tdagnum{dag_num}\ttime:{et - st:.6f}")
    pd.DataFrame([{'nn':_n, 'dn':dag_num, 'ct': et - st}]).to_csv(f'm_{__edges_num}.csv', index=False,  header=False, mode='a')

        