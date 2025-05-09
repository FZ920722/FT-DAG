#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
###################################
主 头文件整理；
###################################
"""

import re
import io
import os
import gc
import sys
import glob
import time
import json
import math
import copy
import mmap
# import dill
import shutil
# import psutil
import pickle
import random
import hashlib
import sqlite3
# import objgraph
import datetime
import requests
import tracemalloc
# import cloudpickle

import numpy as np
import pandas as pd
import pynauty as pn
import pathlib as plx
import networkx as nx
import graphviz as gz
import diskcache as dc

from joblib import Memory
from threading import Thread
from datetime import datetime
from scipy.sparse import csr_matrix
from sympy.utilities.iterables import partitions

# from sympy.combinatorics import Permutation
from random import random, sample, uniform, randint
from mysql.connector import pooling, connect, Error
from scipy.sparse.csgraph import connected_components
from func_timeout import func_timeout, FunctionTimedOut
from collections import defaultdict, Counter, namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
from multiprocessing import cpu_count, Pool, Manager, Event, Process, Queue, queues, current_process, Semaphore, Value
from itertools import chain, accumulate, groupby, product, zip_longest,  permutations, combinations, combinations_with_replacement
# import itertools
# import z3.z3
# from z3.z3 import Int, Sum, If, Solver, Or, IntNumRef, Bool, And, Implies

# from bypy import ByPy
# import networkx as nx
# import mysql.connector
# import pymysql as pymysqlx
# import matplotlib.pyplot as plt


global DAG_ROOT_NAME 
DAG_ROOT_NAME = 'BASIC_DAG_DATA'
def get_dag_root_name():
    global DAG_ROOT_NAME
    return DAG_ROOT_NAME


def __data_file_input(_gtype, _tpath):
    for __file_name in os.listdir(_tpath):
        yield _gtype, __file_name

# ########################################################################
# 1 - DAG 数据IO
# #######################################################################

def __exam_pic_Output(_dag, _fname):
    dot = gz.Digraph()
    dot.attr(rankdir='LR')
    for node_x in _dag.nodes(data=True):
        temp_label = f'{node_x[0]}'
        temp_node_dict = node_x[1]
        dot.node('%s' % node_x[0], temp_label, color='black', shape='box')
    for edge_x in _dag.edges():
        dot.edge('%s' % edge_x[0], '%s' % edge_x[1])
    dot.render(filename= f'{_fname}.png', format="png", view=False)


def Matrix_Encoding(_Matrix):
    return np.packbits(_Matrix)


def Matrix_Decoding(_Compress, _N):
    return np.unpackbits(_Compress)[:_N * _N].reshape((_N, _N)).astype(bool)


def file_name(_n, _dt, _shape, _pid=None):
    __root_addr = f"./{DAG_ROOT_NAME}/Node_{_n}/{_dt}/"
    __hash_hex = hashlib.md5(pickle.dumps(list(_shape))).hexdigest()
    if _pid is None:
        return __root_addr + f"{__hash_hex}.npy"
    else:
        return __root_addr + f"{_pid}_{__hash_hex}.npy"


def file_input(_FileName):
    if os.path.getsize(_FileName) > 0:
        with open(_FileName, 'rb') as f:
            # mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # buffer = io.BytesIO(mmapped_file)
            # buffer = io.BytesIO(f.read())
            # buffer.seek(0)
            while True:
                try:
                    # __ret = np.load(buffer, allow_pickle=True)
                    __ret = np.load(f, allow_pickle=True)
                    yield __ret
                except:
                    break
        # buffer.close()  
        # del buffer


def file_output(_FileName, _DataBuffer):
    with open(_FileName, 'ab') as f:
        f.write(_DataBuffer.getbuffer())
        f.flush()
        os.fsync(f.fileno())
    _DataBuffer.seek(0)
    _DataBuffer.truncate(0)
    # os.sync()
    # os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")


# ########################################################################
# 2 - DAG 数据结构格式转换
# #######################################################################

def Dict_Translate(__sour_dict):    # 将k和v转换， 在equ和iso的部分可使用；
    __ret_dict = defaultdict(list)
    for _k, _v in __sour_dict.items():
        __ret_dict[_v].append(_k)
    return __ret_dict


def EQU_Group(_adj_ad, _adj_rd, _adj_dd):
    __equ_ret, __hash_dict = {}, {}
    for __nid, __nds in _adj_ad.items():
        key = (frozenset(_adj_rd[__nid]), frozenset(__nds))
        if key not in __hash_dict:
            __hash_dict[key] = len(__hash_dict)
        __equ_ret[__nid] = __hash_dict[key]    
    return __equ_ret


def ori_certification_(_adj_ad):
    __pn_tsdag = pn.Graph(len(_adj_ad), directed=True, adjacency_dict=_adj_ad)
    return pn.certificate(__pn_tsdag)


def new_certification(_adj_ad, _adj_rd, _adj_dd):
    # （1） 等价结点合并；
    __equ_list = EQU_Group(_adj_ad, _adj_rd, _adj_dd)               # {nid:eid}
    __requ_list = {_eid: [] for _eid in set(__equ_list.values())}   # {eid:[nid]}
    for _nid, _eid in __equ_list.items():
        __requ_list[_eid].append(_nid)

    if max([len(_nids) for _nids in __requ_list.values()]) == 1:
        __color_data = defaultdict(set)
        for _nid, _sns in _adj_ad.items():
            __color_data[(_adj_dd[_nid], len(_adj_ad[_nid]), len(_adj_rd[_nid]))].add(_nid)
        __pn_tsdag = pn.Graph(len(_adj_ad), directed=True, adjacency_dict=_adj_ad)
        __pn_tsdag.set_vertex_coloring([__color_data[_key] for _key in sorted(__color_data.keys())])
        __certix = pn.certificate(__pn_tsdag)
        return hashlib.md5(__certix).hexdigest()

    else:
        __nequ_color_data = defaultdict(set) # {(dd, ls, lp, nn): eids}
        for _eid, _nids in __requ_list.items():
            __nequ_color_data[(_adj_dd[_nids[0]], len(_adj_ad[_nids[0]]), len(_adj_rd[_nids[0]]), len(_nids))].add(_eid)

        __nadj_aj = {_eid: list(set([__equ_list[_snid] for _snid in _adj_ad[_nids[0]]])) for _eid, _nids in __requ_list.items()}
        __pn_tsdag = pn.Graph(len(__nadj_aj), directed=True, adjacency_dict=__nadj_aj)
        __pn_tsdag.set_vertex_coloring([__nequ_color_data[_key] for _key in sorted(__nequ_color_data.keys())])

        # ################################################################ #
        __certix2 = pn.certificate(__pn_tsdag)
        __ret = pn.autgrp(__pn_tsdag)
        __canonx = pn.canon_label(__pn_tsdag)  # canon_label
        # ################################################################ #
        __iso_group = {__ret[3][__ceid]: [] for __ceid in __canonx}
        for __ceid in __canonx:             # 规范化ID
            __iso_group[__ret[3][__ceid]].append(len(__requ_list[__ceid]))
        __certix1 = pickle.dumps([set(Counter(_igv).items()) for _igv in __iso_group.values()])
        return hashlib.md5(__certix1 + __certix2).hexdigest()


# (1) 邻接表转邻接矩阵；
def adjd_to_adjm(_adj_d):
    __n_num = len(_adj_d)
    __ret_adj_m = np.zeros((__n_num, __n_num), dtype=bool)
    for __pid, __sls in _adj_d.items():
        for __slx in __sls:
            __ret_adj_m[__pid][__slx] = True
    return __ret_adj_m


# (2) 邻接矩阵转邻接表；
def adjm_to_adjd(_adj_m):
    __n_num = _adj_m.shape[0]
    __xy = np.where(_adj_m)
    __ret_adj_d = {__nidx: [] for __nidx in range(__n_num)}
    for _x, _y in zip(__xy[0], __xy[1]):
        __ret_adj_d[_x].append(_y)
    return __ret_adj_d


# (3) 邻接矩阵转networkx；
def adjm_to_dag(_m):
    assert _m.shape[0] == _m.shape[1]
    __ret_dag = nx.DiGraph()
    __n_num = _m.shape[0]
    __xy = np.where(_m)
    __ret_dag.add_nodes_from([__ni for __ni in range(__n_num)])
    __ret_dag.add_edges_from([(_x, _y) for _x, _y in zip(__xy[0], __xy[1])])
    return __ret_dag


# (4) networkx转邻接矩阵
def dag_to_adjm(_dagx):
    __n_num = _dagx.number_of_nodes()
    __ret_adj_m = [[False for _ in range(__n_num)] for _ in range(__n_num)]
    # __ret_adj_m = np.zeros((__n_num, __n_num), dtype=bool)
    for __pid, __sid in _dagx.edges():
        __ret_adj_m[__pid][__sid] = True
    return np.array(__ret_adj_m)



# ########################################################################
# 基础算法；
# #######################################################################
# (1) 基于可达表穷举所有对抗链
def ad_anti(_arr_ad):
    __n_set = set(_arr_ad.keys())
    for __nidx in __n_set:
        __pall_nodes_set = {__n_x:_arr_ad[__n_x] for __n_x in __n_set 
                            if __n_x > __nidx and __n_x not in _arr_ad[__nidx]}
        if len(__pall_nodes_set) == 0: 
            yield (__nidx,)
        else:
            for __s_ans in ad_anti(__pall_nodes_set):
                yield (__nidx,) + tuple(__s_ans)
    yield tuple([])


# (2) 基于邻接表获取祖先节点示例邻接矩阵
def ad_ances(_adj_ad, _nids):
    __pre_nids = set([__tnid for __nidx in _nids for __tnid, __tnsuccs in _adj_ad.items() 
                      if __nidx in __tnsuccs])  # (1) 获取所有nides的前驱；
    if len(__pre_nids) == 0:                    # (2) 如果为空，直接返回；  
        return __pre_nids
    else:                                       # (3) 否则递归，然后合并返回
        return ad_ances(_adj_ad, __pre_nids) | __pre_nids

# (3) 基于邻接表获取各节点的深度；
def ad_to_deep(_dag_ad, _l_num=0):
    __un_source_nodes = set(chain(*_dag_ad.values()))
    __all_nodes = set(_dag_ad.keys())
    __source_nodes = __all_nodes - __un_source_nodes
    __sret = {__snid: _l_num for __snid in __source_nodes} 
    if len(__un_source_nodes) == 0:
        return __sret
    else:
        __oret = ad_to_deep({__nid: __sls for __nid, __sls in _dag_ad.items() 
                             if __nid not in __source_nodes}, _l_num + 1)
        return __sret | __oret

# (4) 邻接表转可达表；
def def_of_node(_nid, _adj_list):
    _ds = [] + _adj_list[_nid]
    for __snid in _adj_list[_nid]:
        _ds += def_of_node(__snid, _adj_list)
    return list(set(_ds))


def adjd_to_arrd(adj_list):
    arr_list = {__nid:  def_of_node(__nid, adj_list) for __nid, __ns in adj_list.items()}
    return arr_list

# (5) 邻接表拆分连通分量；

# (6) 邻接表拆分block


# ########################################################################
# Key parameter analysis & update labelled DAG to configured DAG
# Input: DAG with no attribute parameter
#       assume： There's only one source(sink) node;
# Output: Give the DAG with attribute parameter
# #######################################################################
# """
import networkx as nx
def Dag_Topology_Initial(_CDag):
    # [sorted(generation) for generation in nx.topological_generations(DAG_obj)]
    # [sorted(generation) for generation in nx.topological_generations(nx.DiGraph.reverse(DAG_obj))]
    _CDag.graph["n"] = _CDag.number_of_nodes()
    _CDag.graph["m"] = _CDag.number_of_edges()
    _CDag.graph["e"] = json.dumps(tuple(_CDag.edges()))
    
    # #### (1) 正向 shape #### #
    __RankList = list(nx.topological_generations(_CDag))  # print(f'拓扑分层：{__RankList}')
    # for __RankId, __RankL in enumerate(__RankList, start=1):
    #     for __RankX in __RankL:
    #         _CDag.nodes[__RankX]['d'] = __RankId
    for __d, __n in enumerate(nx.topological_generations(_CDag), start=1):
        for __x in __n:
            _CDag.nodes[__x]['d'] = __d

    __ShList = [len(_RLx) for _RLx in __RankList]  # ['Shape_List'] = sh_list
    # self.parallelism = max([len(rank_x) for rank_x in rank_list])
    _CDag.graph["l"] = len(__ShList)
    _CDag.graph["s"] = json.dumps(tuple(__ShList))
    _CDag.graph["su"] = max(__ShList)
    _CDag.graph["sd"] = min(__ShList)

    # #### 3.antichains #### # list(nx.antichains(DAG_obj, topo_order=None))
    # temp_G1 = nx.transitive_closure_dag(_CDag)
    # temp_G1 = nx.transitive_closure(DAG_obj, reflexive=None)
    # temp_G1 = nx.transitive_reduction(DAG_obj)
    # temp_G3 = nx.maximal_matching(temp_G2)
    # temp_G4 = nx.min_edge_cover(temp_G2)
    # temp_G5 = nx.bipartite.maximum_matching(temp_G2)

    """
    temp_G2 = nx.DiGraph()
    for edge_x in _CDag.edges():
        temp_G2.add_node('p' + str(edge_x[0]), bipartite=0)
        temp_G2.add_node('d' + str(edge_x[1]), bipartite=1)
        temp_G2.add_edge('p' + str(edge_x[0]), 'd' + str(edge_x[1]))
    u = [n for n in temp_G2.nodes if temp_G2.nodes[n]['bipartite'] == 0]
    matching = nx.bipartite.maximum_matching(temp_G2, top_nodes=u)
    _CDag.graph['w'] = _CDag.number_of_nodes() - len(matching) / 2
    """

    # #### 4.1 Degree #### #
    # _CDag.graph['d'] = max([nx.degree(_CDag, __SelfNode) for __SelfNode in _CDag.nodes()])
    _CDag.graph['d'] = max(dict(nx.degree(_CDag)).values())

    # #### 4.2 In-Degree #### #
    # _CDag.graph['id'] = max([_CDag.in_degree(__SelfNode) for __SelfNode in _CDag.nodes()])
    _CDag.graph['id'] = max(dict(_CDag.in_degree()).values())

    # #### 4.3 Out-Degree #### #
    # _CDag.graph['od'] = max([_CDag.out_degree(__SelfNode) for __SelfNode in _CDag.nodes()])
    _CDag.graph['od'] = max(dict(_CDag.out_degree()).values())

    # #### 5. Density of DAG  #### #     2 * nx.density(DAG_obj)    # "Connection_Rate"
    if _CDag.number_of_nodes() == 1:
        _CDag.graph['cr'] = 0
    else:
        _CDag.graph['cr'] = (2 * _CDag.number_of_edges()) / (_CDag.number_of_nodes() * (_CDag.number_of_nodes() - 1))

    # #### 6.Jump level  #### #
    _CDag.graph['jl'] = max([0,] + [_CDag.nodes[ey]['d'] - _CDag.nodes[ex]['d'] for ex, ey in _CDag.edges()])

    # #### 7.Hang level  #### #
    _CDag.graph['hl'] = min([nd['d'] for ni, nd in _CDag.nodes(data=True) if _CDag.out_degree(ni) == 0])

    _CDag.graph['dag'] = pickle.dumps(_CDag)


def SelfCombinations(_datalist, _size):
    """穷举_datalist(可能有重复)的所有长度为_size的"""
    if _size == 1:
        for __x in set(_datalist):
            yield [__x, ]
    else:
        for __x in set(_datalist):
            __rest_data = list(filter(lambda x: x >= __x, _datalist))
            __rest_data.remove(__x)
            for __y in SelfCombinations(__rest_data, _size - 1):
                yield [__x, ] + __y


def __subtract_elements(arr1, arr2):
    # 创建一个副本来存储 arr2 的元素数量
    counts = {}
    for elem in arr2:
        counts[elem] = counts.get(elem, 0) + 1
    result = []  # 用于存储结果的新数组
    for elem in arr1:
        if elem in counts and counts[elem] > 0:
            counts[elem] -= 1  # 减少计数器，表示已经移除一次该元素
        else:
            if len(elem) == 1 and sum(elem) > 1:
                return None
            result.append(elem)  # 如果不在 counts 中或计数为 0，则保留该元素
    return result


def t_o_ret(_tpsh_group, _ppsh_group, __rg_num):
    for __pssh_group in SelfCombinations(_ppsh_group, __rg_num):
        __other_group  = __subtract_elements(_ppsh_group, __pssh_group)
        # print(f"{_psh_group}--{__tpsh_group}--{__other_group}")
        yield __other_group, _tpsh_group + __pssh_group


# (2) 基于DFS的弱联通判定
# def weakly_conn_det(adj_list, _llnum):
#     undirected_adj_list = {}
#     for node in adj_list:
#         if node not in undirected_adj_list:
#             undirected_adj_list[node] = set()
#         for neighbor in adj_list[node]:
#             if neighbor not in undirected_adj_list:
#                 undirected_adj_list[neighbor] = set()
#             undirected_adj_list[node].add(neighbor)
#             undirected_adj_list[neighbor].add(node)
#     # 使用DFS检查连通性
#     visited = set()
#     def dfs(node):
#         stack = [node]
#         while stack:
#             current = stack.pop()
#             if current not in visited:
#                 visited.add(current)
#                 stack.extend(undirected_adj_list[current] - visited)

#     # 从任意一个节点开始DFS
#     start_node = next(iter(undirected_adj_list))
#     dfs(start_node)
#     # WEAKTIME += time.time() - __st
#     return len(visited) == len(undirected_adj_list)


# def MC_Nodes_Combinate(_NNum):
#     """ 基于_NNum穷举所有单连通分量的结点数量组合，不包括单连通情况；生成所有结点数量组合
#     (默认结果从小到大)；   MC DAG的连通分量一定是大于1的"""
#     def __recursion_generate(_rnnum, _mnnum):
#         for __rTNNum in range(1, min(_rnnum, _mnnum) + 1):   # 此次迭代的连通分量结点数量
#             if __rTNNum == _rnnum:
#                 yield [__rTNNum,]
#             elif __rTNNum < _rnnum:
#                 for __occx in __recursion_generate(_rnnum - __rTNNum, __rTNNum):
#                     yield __occx + [__rTNNum, ]
#             else:
#                 assert False
#     for __FNNum in range(1, _NNum):
#         for __OCComb in __recursion_generate(_NNum - __FNNum, __FNNum):
#             yield __OCComb + [__FNNum, ]


""" 计算内存开销：注意会有额外延时
# tracemalloc.start()     # 开始跟踪内存分配
# snapshot = tracemalloc.take_snapshot()  # 获取当前内存分配的快照
# stats = snapshot.statistics('lineno')   # 获取当前和之前内存分配的统计
# print(f"总内存开销: {sum(stat.size for stat in stats) / 1024 / 1024:.2f} MB")   # 计算总的内存开销
# tracemalloc.stop()                                                             # 停止跟踪内存分配
# (1) ***************  内存开销统计  *************** #
# objgraph.show_growth()
"""

# for x in range(3, 100):
#     test = []
#     test_num = 0
#     for y in Shape_Enumerate(x):
#         # print(y)
#         # test.append(y)
#         test_num += 1
#     print(f"node_num:{x},shape_num:{test_num},time_cost:{__et-__st:.2f}")

# def dag_equ_extension(_SubDagM, _RLNMaxNum, _INELabel):
#     """ 拓展_NSubDagx的等价结点 : 尾层节点的可放回组合， 数量为目标数量与当前数量的差；"""
#     global METIME
#     global CERTITIME
#     __st = time.time()
#     __TestLabel = defaultdict(set)
#     __tnnum = _SubDagM.shape[0]                             # 结点数量；
#     __sub_dag_ad = adjm_to_adjd(_SubDagM)                   # 子DAG邻接表；
#     __sub_dag_rad = adjm_to_adjd(_SubDagM.T)                # 子DAG前驱表；
#     __sub_dag_node_num = len(__sub_dag_ad)                  # 子DAG的结点数量；
#     __sun_dag_deep_dick = ad_to_deep(__sub_dag_ad)          # 子DAG的结点深度表；
#     __sub_dag_local_ln = max(__sun_dag_deep_dick.values())  # 前驱子DAG的层数；
#     __local_nids = set([__pad_ni for __pad_ni, __pad_nd in __sun_dag_deep_dick.items() if __pad_nd == __sub_dag_local_ln]) # DAG的尾层结点队列
#     __equ_labels = set([(frozenset(__sub_dag_ad[__nix]), frozenset(__sub_dag_rad[__nix])) for __nix in __local_nids])
#     __equ_labels2 = {__x: __i for __i, __x in enumerate(__equ_labels)}
#     __equ_node_id = defaultdict(set)
#     for __nix in __local_nids:
#         __equ_node_id[__equ_labels2[(frozenset(__sub_dag_ad[__nix]), frozenset(__sub_dag_rad[__nix]))]].add(__nix)
#     for _RLNSNum in range(1, _RLNMaxNum + 1):
#         if (__tnnum - len(__local_nids), len(__local_nids) + _RLNSNum) not in _INELabel:
#             continue
#         __sdag_adj_dict = __sub_dag_ad | {__tnnum + __rnid: [] for __rnid in range(_RLNSNum)}
#         for __cnids in combinations_with_replacement(range(len(__equ_labels)), _RLNSNum):
#             __cnidsc = Counter(__cnids)
#             __cnix = []
#             for __equid, __ennum in __cnidsc.items():
#                 __cnix += random.choices(list(__equ_node_id[__equid]), k=__ennum)
#             __ntdag_adj_dick = {__nid: [] for __nid in range(__sub_dag_node_num + _RLNSNum)}
#             for __tpad_id in range(__tnnum):
#                 __ntdag_adj_dick[__tpad_id] += __sdag_adj_dict[__tpad_id] 
#             for __cnsid, __nnsid in enumerate(__cnix): # 扩展label， 扩展结点ID
#                 for __pnx in __sub_dag_rad[__nnsid]:
#                     __ntdag_adj_dick[__pnx].append(__tnnum + __cnsid)
#             __st = time.time()
#             __pn_tsdag = pn.Graph(__tnnum + _RLNSNum, directed=True, adjacency_dict=__ntdag_adj_dick)
#             __certi_label = pn.certificate(__pn_tsdag)
#             CERTITIME += time.time() - __st
#             if __certi_label not in __TestLabel[_RLNSNum]:
#                 __TestLabel[_RLNSNum].add(__certi_label)
#                 __Ret = adjd_to_adjm(__ntdag_adj_dick)
#                 METIME += time.time() - __st
#                 yield _RLNSNum, __Ret
#                 __st = time.time()
#             METIME += time.time() - __st


# def dag_equ_extension_single(_SubDagM, _RLNSNum):
#     """ 拓展_NSubDagx的等价结点 : 尾层节点的可放回组合， 数量为目标数量与当前数量的差；"""
#     global METIME
#     global CERTITIME
#     __st = time.time()
#     __TestLabel = set()
#     __tnnum = _SubDagM.shape[0]                             # 结点数量；
#     __sub_dag_ad = adjm_to_adjd(_SubDagM)                   # 子DAG邻接表；
#     __sub_dag_rad = adjm_to_adjd(_SubDagM.T)                # 子DAG前驱表；
#     __sub_dag_node_num = len(__sub_dag_ad)                  # 子DAG的结点数量；
#     __sun_dag_deep_dick = ad_to_deep(__sub_dag_ad)          # 子DAG的结点深度表；
#     __sub_dag_local_ln = max(__sun_dag_deep_dick.values())  # 前驱子DAG的层数；
#     __local_nids = set([__pad_ni for __pad_ni, __pad_nd in __sun_dag_deep_dick.items() if __pad_nd == __sub_dag_local_ln]) # DAG的尾层结点队列
#     __sdag_adj_dict = __sub_dag_ad | {__tnnum + __rnid: [] for __rnid in range(_RLNSNum)}
#     __equ_labels = set([(frozenset(__sub_dag_ad[__nix]), frozenset(__sub_dag_rad[__nix])) for __nix in __local_nids])
#     __equ_labels2 = {__x: __i for __i, __x in enumerate(__equ_labels)}
#     __equ_node_id = defaultdict(set)
#     for __nix in __local_nids:
#         __equ_node_id[__equ_labels2[(frozenset(__sub_dag_ad[__nix]), frozenset(__sub_dag_rad[__nix]))]].add(__nix)
#     for __cnids in combinations_with_replacement(range(len(__equ_labels)), _RLNSNum):
#         __cnidsc = Counter(__cnids)
#         __cnix = []
#         for __equid, __ennum in __cnidsc.items():
#             __cnix += random.choices(list(__equ_node_id[__equid]), k=__ennum)
#         __ntdag_adj_dick = {__nid: [] for __nid in range(__sub_dag_node_num + _RLNSNum)}
#         for __tpad_id in range(__tnnum):
#             __ntdag_adj_dick[__tpad_id] += __sdag_adj_dict[__tpad_id] 
#         for __cnsid, __nnsid in enumerate(__cnix): # 扩展label， 扩展结点ID
#             for __pnx in __sub_dag_rad[__nnsid]:
#                 __ntdag_adj_dick[__pnx].append(__tnnum + __cnsid)
#         __pn_tsdag = pn.Graph(__tnnum + _RLNSNum, directed=True, adjacency_dict=__ntdag_adj_dick)
#         __st = time.time()
#         __certi_label = pn.certificate(__pn_tsdag)
#         CERTITIME += time.time() - __st
#         if __certi_label not in __TestLabel:
#             __TestLabel.add(__certi_label)
#             __Ret = adjd_to_adjm(__ntdag_adj_dick)
#             METIME += time.time() - __st
#             yield __Ret
#             __st = time.time()
#     METIME += time.time() - __st


""" 连通分量测试； """
# import time
# import numpy as np
# from scipy.sparse.csgraph import connected_components

# from MainHead import *

# 示例邻接矩阵
"""
adj_matrix = np.array([[0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1],])

# adj_matrix = np.array([[0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0]])

# n = 100
# matrix = np.array([[True, False, True], [False, True, False], [True, True, False]], dtype=bool)
# adj_matrix = np.random.choice([True, False], size=(n, n))
# adj_matrix = np.array([ [False, False, False, False, False, False], 
#                         [False, False, False, False, False, False], 
#                         [False, False, False, False, False, False], 
#                         [False, False, False, False, False, False], 
#                         [False, False, False, False, False, False], 
#                         [False, False, False, False, False, False]])

st = time.time()

n_components, labels = connected_components(adj_matrix, return_labels=True)
print(n_components)
print(labels)
separated_matrices =  [adj_matrix[np.ix_(labels == i, labels == i)] for i in range(n_components)]

et = time.time()

for idx, matrix in enumerate(separated_matrices):
    print(f"Connected Component {idx + 1}:\n{matrix}\n")
print(et - st)
"""



# def reachable_nodes(nid, adj_list, visited=None):
#     if visited is None:
#         visited = set()
#     if nid in visited:
#         return set()
#     visited.add(nid)
#     reachable = set(adj_list.get(nid, []))
#     for neighbor in adj_list.get(nid, []):
#         reachable |= reachable_nodes(neighbor, adj_list, visited)
#     return reachable

# def adj_to_reachable(adj_list):
#     arr_list = {nid: reachable_nodes(nid, adj_list) for nid in adj_list.keys()}
#     return arr_list

# n = 100
# # matrix = np.array([[True, False, True], [False, True, False], [True, True, False]], dtype=bool)
# adj_matrix = np.random.choice([True, False], size=(n, n))
# adj_ad = adjm_to_adjd(adj_matrix)

# st = time.time()
# # x = adj_to_reachable(adj_ad)
# c = adj_matrix @ adj_matrix
# et = time.time()
# print(et - st)


# def reachable_nodes(nid, adj_list, visited=None):
#     if visited is None:
#         visited = set()
#     if nid in visited:
#         return set()
#     visited.add(nid)
#     reachable = set(adj_list.get(nid, []))
#     for neighbor in adj_list.get(nid, []):
#         reachable |= reachable_nodes(neighbor, adj_list, visited)
#     return reachable

# def adj_to_reachable(adj_list):
#     arr_list = {nid: reachable_nodes(nid, adj_list) for nid in adj_list.keys()}
#     return arr_list

# # 示例
# adj_list = {
#     1: [2, 3],
#     2: [4],
#     3: [],
#     4: [3],
# }

# reachable_table = adj_to_reachable(adj_list)
# print(reachable_table)





""" 哈希计算 与 数据压缩"""
# import sys
# import hashlib
# import numpy as np

# def save_array_with_hash(array, path):
#     # 计算文件内容的哈希值
#     hash_object = hashlib.md5(array.tobytes())
#     hash_hex = hash_object.hexdigest()
#     short_filename = f"{hash_hex[:8]}.npy"  # 使用哈希值的前8位作为短文件名
#     np.save(f"{path}/{short_filename}", array)

# # 示例
# array = np.array([1, 2, 3, 4])
# save_array_with_hash(array, 'data')

# if np.array_equal(matrix, restored):
#     print(123)
# print(sys.getsizeof(matrix))
# print(sys.getsizeof(compressed))

# (2) 方法2
# mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
# return mmap_array[start:end]
# buffer = io.BytesIO(mmapped_file)
# while True:
#     try:
#         __ret = np.load(buffer, allow_pickle=True)
#         yield __ret
#     except EOFError:
#         break
#     except Exception as e:
#         # print(f"Error processing file: {e}")
#         break
# # with open(__fn, 'rb') as f:
#     # for __Ret in file_input(__fn, sum(_shape)):
#     # yield __Ret
#     # __dag_num += 1
#     # (1) 方法1 
#     """
#     while True:
#         try:
#             __ret = np.load(f, allow_pickle=True)
#             yield __ret
#         except:
#             break
#     """


""" 数据压缩 """
# print(calculate_n_from_m(10))
# 示例布尔矩阵（非标准上三角矩阵）
# bool_matrix = np.array([[False, True,  False, True,  True],
#                         [False, True,  False, True,  True],
#                         [False, False, True,  True,  True],
#                         [False, False, False, False, True],
#                         [False, False, False, False, False]])
# print(sys.getsizeof(bool_matrix))
# print(sys.getsizeof(np.packbits(bool_matrix)))
# n = bool_matrix.shape[0]
# print("重新排序后的矩阵:")
# cm = np.packbits(compressed)
# print(sys.getsizeof(cm))
# ccc = int(n * (n - 1) / 2)
# new_compressed = np.unpackbits(cm)[:ccc].reshape(ccc).astype(bool)

# from collections import defaultdict, Counter
# from itertools import chain, accumulate, groupby

# def __data_file_input(_gtype, _tpath):
#     # global connection_pool
#     for __file_name in os.listdir(_tpath):
#         yield _gtype, __file_name#, connection_pool
# from Src.MainHead import *

# n = 5
# # adj_matrix = np.random.choice([True, False], size=(n, n))

# while True:
#     temp_dag = nx.gnp_random_graph(n=n, p=0.4, directed=True)
#     if nx.is_directed_acyclic_graph(temp_dag):
#         break
# adj_matrix = nx.to_numpy_array(temp_dag, dtype=bool)
# print(adj_matrix)
# _Compress =  Matrix_Encoding(adj_matrix)
# new_matrix = Matrix_Decoding(_Compress, n)
# print(new_matrix)


# 1. 位图压缩（Bitmasking or Bitmap Compression）
# def compress_bool_matrix(matrix):
#     return np.packbits(matrix)

# # 解压函数：将位图还原为布尔矩阵
# def decompress_bool_matrix(bitmask, shape):
#     # return np.unpackbits(bitmask)[:np.prod(shape)].reshape(shape)
#     return np.unpackbits(bitmask)[:np.prod(shape)].reshape(shape).astype(bool)
# # 示例
# n = 100
# # matrix = np.array([[True, False, True], [False, True, False], [True, True, False]], dtype=bool)
# matrix = np.random.choice([True, False], size=(n, n))
# compressed = compress_bool_matrix(matrix)
# restored = decompress_bool_matrix(compressed, matrix.shape)

# print("Original Matrix:")
# print(matrix)
# print("Compressed:")
# print(compressed)
# print("Restored Matrix:")
# print(restored)


# __xy = np.where(_m)
# __ret_dag.add_nodes_from([__ni for __ni in range(__n_num)])
# __ret_dag.add_edges_from([(_x, _y) for _x, _y in zip(__xy[0], __xy[1])])
# return __ret_dag



# def shape_component_group(_shape):
#     """
#     for __sc_list in number_combine(_shape[0], (2, min(_shape[0],  _shape[1] + 1)), (1, _shape[0])): 
#         __group_num = len(__sc_list)
#         # case 1 shape[1] >= 分组的数量， 用原始方法；
#         if _shape[1] >= len(__sc_list): 
#             for __retx in mc_generate([(__x, ) for __x in __sc_list], _shape[1:]):
#                 yield __retx
#         # case 2 shape[1] < 分组的数量， 最多分 shape[1] + 1组，多出来的一组是离散图；
#         else:
#             for _sd_g in set(__sc_list):
#                 __index = __sc_list.index(_sd_g)
#                 __ctt = [_x for _x in __sc_list[:__index] + __sc_list[__index + 1:]]
#                 for __retx in mc_generate([tuple([__x, ]) for __x in __ctt], _shape[1:]):
#                     yield __retx + [tuple([1, ]) for _ in range(__sc_list[__index])]
#     """
#     for __sc_list in number_combine(_shape[0], (2, min(_shape[0],  _shape[1] + 1)), (1, _shape[0])): 
#         __group_num = len(__sc_list)
#         # case 1 shape[1] >= 分组的数量， 用原始方法；
#         if _shape[1] >= len(__sc_list): 
#             for __retx in mc_generate([(__x, ) for __x in __sc_list], _shape[1:]):
#                 yield __retx
#         # case 2 shape[1] < 分组的数量， 最多分 shape[1] + 1组，多出来的一组是离散图；
#         else:
#             for _sd_g in set(__sc_list):
#                 __index = __sc_list.index(_sd_g)
#                 __ctt = [_x for _x in __sc_list[:__index] + __sc_list[__index + 1:]]
#                 for __retx in mc_generate([tuple([__x, ]) for __x in __ctt], _shape[1:]):
#                     yield __retx + [tuple([1, ]) for _ in range(__sc_list[__index])]


# def mc_generate(_psh_group, _succ_shape):
#     __pc_group_num = len(_psh_group)  # 前驱shape的连通分量数量
#     for __group_num in range(1, min(__pc_group_num, _succ_shape[0]) + 1):
#         for __tpsh_group in SelfCombinations(_psh_group, __group_num):
#             __other_group = _psh_group.copy()   # 没有被选中的分量定义
#             for __x1 in __tpsh_group:
#                 __other_group.remove(__x1)
#             if __group_num == 1:                # (*) 如果只分了一组，后续shape就全分过去
#                 __ret1 = [__tpsh_group[0] + tuple(_succ_shape), ] + __other_group
#                 yield __ret1
#             elif __group_num > 1:               # (*) 分了多组，后续shape就全分过去
#                 for __sc_list in combination_generator(_succ_shape[0], __group_num):
#                     assert len(__sc_list) == len(__tpsh_group)
#                     __tset = set()
#                     for __nnp in permutation_generator(__sc_list):
#                         __src_sh_group = [__pcsh + tuple([__succ_nn]) for __pcsh, __succ_nn in zip(__tpsh_group, __nnp)]
#                         __tl = frozenset(Counter(__src_sh_group).items())
#                         if __tl not in  __tset:
#                             __tset.add(__tl)
#                             if len(_succ_shape) == 1:
#                                 yield __src_sh_group + __other_group
#                             elif len(_succ_shape) > 1:
#                                 for __retx in mc_generate(__src_sh_group, _succ_shape[1:]):
#                                     yield __retx + __other_group
#                             else:
#                                 assert False
#             else:
#                 assert False

