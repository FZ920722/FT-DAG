#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MainHead import *

def NSG(_n: int, _lr: tuple, _sr: tuple, _mr: tuple, _id:int, _od:int, _Xs:int=0):
    """ 尾结点数小者优先；"""
    if 1 <= _lr[0] <= _lr[1] and 1 <= _sr[0] <= _sr[1] and _lr[0] * _sr[0] <= _n <= _lr[1] * _sr[1]:
        __Xld, __Xlu = _sr[0], _sr[1]
        if 0 < _od and 0 < _id:
            __Xld = max(__Xld, math.ceil(_Xs/_od))
            # S1. 递归生成长shape：
            if 1 < _lr[1]:
                for __Xp, __Xl in map(lambda __x: (_n - __x, __x), range(__Xld, __Xlu + 1)):
                    __l_md, __l_mu = __Xl,                  min(_id, __Xp - (_lr[0] - 1) + 1) * __Xl
                    __p_md, __p_mu = max(0, _lr[0] - 2),    int(pow(__Xp, 2) / 4)
                    if __Xp * _od >= __Xl and __l_md  + __p_md <= _mr[1] and _mr[0] <= __p_mu + __l_mu:
                        __nlr = (max(1, _lr[0] - 1), _lr[1] - 1)
                        __nsr = (_sr[0],  min(_sr[1], __Xp))
                        __nmr = (max(0, _mr[0] - __l_mu), _mr[1] - __l_md)
                        for __s_p in NSG(__Xp, __nlr, __nsr, __nmr, _id, _od, __Xl):
                            yield __s_p + (__Xl, )
        # S2. 不在递归，直接返回本层结点数量作为shape的第一层；
        if _lr[0] <= 1 <= _lr[1] and __Xld <= _n <= __Xlu and _mr[0] <= 0 <= _mr[1]:
            yield (_n, )

def TempShape(_n, _max_s, _ld, _lu, _sd, _su, _id, _od):
# def NSG(_n: int, _lr: tuple, _sr: tuple, _mr: tuple, _id:int, _od:int, _Xs:int=0):
    if 1 <= _max_s <= _n:
        if _n == _max_s:
            yield (_n, )
        else:
            __rn = _n - _max_s
            for __shx in NSG(__rn, (max(_ld - 1, 1), _lu - 1), 
                                (_sd, _su), 
                                (0, int(pow(__rn, 2) / 4)), _id, _od):
                for _rsh in set([__shx[:__index] + (_max_s, ) + __shx[__index:] for __index in range(len(__shx) + 1)]):
                    yield  _rsh


def lbl_node_conn(_pd, _lns:int, _ln:int, _jl:int):
    __nid = _pd.number_of_nodes()
    # S1 anti穷举；
    for __pd_ai in nx.antichains(_pd.subgraph([__i for __i, __d in _pd.nodes(data=True) if _ln - _jl <= __d['d'] < _ln])):
        if _ln - 1 in set(_pd.nodes[__pdax]['d'] for __pdax in __pd_ai):
            # S2 加入结点；连接；
            __nd = nx.DiGraph(_pd)
            __nd.add_nodes_from([(__nid, {'d': _ln}),])
            __nd.add_edges_from([(__pday, __nid) for __pday in __pd_ai])

            # S3 如果lns为1直接返回，否则减一递归；
            if _lns == 1:
                yield __nd
            elif _lns > 1:
                for __xd in lbl_node_conn(__nd, _lns - 1, _ln, _jl):
                    yield __xd
            else:
                assert False

def LBL_Dag_Generator(_pd, _os, _ln:int, _jl:int=float('inf')):
    for __npd in lbl_node_conn(_pd, _os[0], _ln, _jl):
        if len(_os) == 1:
            yield __npd
        else:
            for __rpd in LBL_Dag_Generator(__npd, _os[1:], _ln + 1, _jl):
                yield __rpd

# 测试代码
if __name__ == "__main__":
    _lx = 3  
    for nnum in range(3, 50):
        st = time.time()
        rdag_num = 0
        shape_num = 0

        _max_s = nnum - _lx + 1
        _ldx, _lux = _lx, _lx
        sd, su = 1, nnum
        for __s in TempShape(nnum, _max_s, _ldx, _lux , sd, su, nnum - 1, nnum - 1):            
            sdag = nx.DiGraph()
            sdag.add_nodes_from([(__i, {'d': 1}) for __i in range(__s[0])])
            if len(__s) == 1:
                rdag = nx.DiGraph(sdag)
                rdag_num += 1

            elif len(__s) > 1:
                for rdag in LBL_Dag_Generator(sdag, __s[1:], 2):

                    rdag_num += 1
            else:
                assert False
        et = time.time()
        print(f"{nnum}_\t{shape_num}_\t{et - st :.6f}_\t{rdag_num}")


# __nnum - 2 # math.ceil(__nnum * 3/ 4)
# if _ldx <= len(__s) <= _ldx:
# for shape in shape_generation(nnum, (1, nnum), (1, nnum)):
# shape_num += 1
# if not _pl_nodes.isdisjoint(_pdag_anti_nodes):
#     pn_list.append(_pdag_anti_nodes)
