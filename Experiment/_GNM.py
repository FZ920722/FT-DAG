#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MainHead import *

# GNM DAG生成算法基础穷举（基于边组合的穷举）：
# 可控参数：
#   - m ： DAG的边数量；
def GNM_Dag_Generator(_n, _md, _mu):
    __node_set = set(range(_n))
    __edge_set = set((__y, __x) for __x in range(_n) for __y in range(__x))
    for __mnum in range(_md, _mu + 1):
        if __mnum <= len(__edge_set):
            for __edges in combinations(__edge_set, __mnum):
                __rdag = nx.DiGraph()
                __rdag.add_nodes_from(__node_set)
                __rdag.add_edges_from(__edges)
                yield __rdag


if __name__ == "__main__":
    for _n in range(3, 100):
        st = time.time()
        _md, _mu = 0, int(_n * (_n - 1) / 2)
        dag_num = 0
        for dagx in GNM_Dag_Generator(_n, _md, _mu):
            dag_num += 1
            assert _md <= dagx.number_of_edges() <= _mu
        et = time.time()
        print(f"n:{_n}\tdagnum{dag_num}\ttime:{et - st:.6f}")

