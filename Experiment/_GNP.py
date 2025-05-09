#!/usr/bin/python3
# -*- coding: utf-8 -*-

from MainHead import *

def __dag_gnp(_n, _p):
    __t_dag = nx.DiGraph()
    for __n_x in range(_n):
        __t_dag.add_node(__n_x)
    for __row in range(_n): 
        for __col in range(__row + 1, _n):
            if _p >= random.random():
                __t_dag.add_edge(__row, __col)
    return __t_dag

def networkx_gnp(_n, _p):
    while True:
        temp_dag = nx.gnp_random_graph(n=_n, p=_p, directed=True)
        if nx.is_directed_acyclic_graph(temp_dag):
            return temp_dag

def networkx_gnp(_n, _p):
    while True:
        temp_dag = nx.gnp_random_graph(n=_n, p=_p, directed=True)
        if nx.is_directed_acyclic_graph(temp_dag):
            return temp_dag
