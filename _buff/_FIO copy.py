#!/usr/bin/python3
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # #
# Real-Time Systems Group
# Hunan University HNU
# Fang YJ
# # # # # # # # # # # # # # # #

from MainHead import *

# FAN-IN-FAN-OUT DAG生成算法基础穷举（基于点迭代的穷举）：
# 可控参数：
#   - id ： 最大入度；
#   - od ： 最大出度；
#   - l  ： 最大长度；
def AntiChain(_sdag, _rnd, _rnu, _ns):
    if 1 <= _rnd <= _rnu:
        for __ni in _ns:
            __redata = set((__ni,))

            if _rnd == 1:
                yield __redata
        
            if 1 < _rnu:
                __sub_nodes = set(__sni for __sni in _ns if __sni > __ni) \
                    - nx.descendants(_sdag, __ni) - nx.ancestors(_sdag, __ni)
                for __rai in AntiChain(_sdag, max(_rnd - 1, 1), _rnu - 1, __sub_nodes):
                    yield __redata | __rai

def FanInFanOut(_nnum, _id=float('Inf'), _od=float('Inf'), _ld=float('Inf')):
    if _nnum > 0 and _ld > 0:
        if _ld == 1:
            __rdag = nx.DiGraph()
            __rdag.add_nodes_from([(_i, {'d': 1}) for _i in range(_nnum)])    
            yield __rdag
        for __sub_dag in FanInFanOut(_nnum - 1, _id, _od, max(1, _ld - 1)):
            __sub_dd = max([__snd['d'] for __sni, __snd in __sub_dag.nodes(data=True)])
            __rn_i = __sub_dag.number_of_nodes()
            __temp_pndoes = [__tni for __tni, __tnd in __sub_dag.nodes(data=True)]
            for __pns in nx.antichains(__sub_dag):
                if len(__pns) <= _id and max([0,] + [__sub_dag.out_degree(__pns_x) for __pns_x in __pns]) < _od:
                    __rdag = nx.DiGraph(__sub_dag)
                    __max_d = 1 if len(__pns) == 0 else max([__rdag.nodes[__pns_x]['d'] for __pns_x in __pns]) + 1
                    __rdag.add_nodes_from([(__rn_i, {'d':__max_d})])
                    __rdag.add_edges_from([(__pns_x, __rn_i) for __pns_x in __pns])
                    if max(__sub_dd, __max_d) >= _ld:
                        yield __rdag
        # 结点提取；
        # 判定是否满足条件
    # __nid = _sdag.number_of_nodes()
    # __sub_nodes = set(_i for _i in _sdag.nodes() if _sdag.out_degree(_i) < _od and _sdag.nodes[_i]['d'] < _l)
    # for __pns in chain([set()], AntiChain(_sdag, 1, _id, set(_i for _i in _sdag.nodes() if _sdag.out_degree(_i) < _od and True))):
    # for __pns in chain([set()], AntiChain(_sdag, 1, _id, __sub_nodes)):
        # __ndag = nx.DiGraph(_sdag)
        # __ndag.add_node(__nid)
        # __ndag.add_nodes_from([(__nid, {'d': 1 if len(__pns) == 0 else max([_sdag.nodes[__pns_x]['d'] for __pns_x in __pns]) + 1})]) 
        # __ndag.add_edges_from([(__px, __nid) for __px in __pns]) 
        # if _nnum == 1:
            # yield __ndag
        # else:
            # for __nrdag in FanInFanOut(__ndag, _nnum - 1, _id, _od, _l):
                # yield __nrdag


# def FanInFanOut(_sdag, _nnum, _id=float('Inf'), _od=float('Inf'), _ld=float('Inf')):
#     __nid = _sdag.number_of_nodes()
#     __sub_nodes = set(_i for _i in _sdag.nodes() if _sdag.out_degree(_i) < _od and _sdag.nodes[_i]['d'] < _l)
#     # for __pns in chain([set()], AntiChain(_sdag, 1, _id, set(_i for _i in _sdag.nodes() if _sdag.out_degree(_i) < _od and True))):
#     for __pns in chain([set()], AntiChain(_sdag, 1, _id, __sub_nodes)):
#         __ndag = nx.DiGraph(_sdag)
#         # __ndag.add_node(__nid)
#         __ndag.add_nodes_from([(__nid, {'d': 1 if len(__pns) == 0 else max([_sdag.nodes[__pns_x]['d'] for __pns_x in __pns]) + 1})]) 
#         __ndag.add_edges_from([(__px, __nid) for __px in __pns]) 
#         if _nnum == 1:
#             yield __ndag
#         else:
#             for __nrdag in FanInFanOut(__ndag, _nnum - 1, _id, _od, _l):
#                 yield __nrdag

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


if __name__ == "__main__":
    id, od = 1, 2
    for nnum in range(3, 100):
        st = time.time()

        ld = max(1, nnum - 2)
        dag_num = 0
        for rdag in FanInFanOut(nnum, id, od, ld):
            # __exam_pic_Output(rdag, f'{nnum}_{dag_num}')
            dag_num += 1
        et = time.time()

        print(f"node_num:{nnum}\t--{et-st:.12f}\t--{dag_num}")
