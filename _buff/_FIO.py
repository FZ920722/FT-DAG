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

# G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (4, 5), (4, 6)])
def FanInFanOut(_sdag, _nnum, _id=float('Inf'), _od=float('Inf'), _l=float('Inf')):
    __nid = _sdag.number_of_nodes()
    # 子DAG的深度不能超过l - 1
    __sub_nodes = set(_i for _i in _sdag.nodes() if _sdag.out_degree(_i) < _od and _sdag.nodes[_i]['d'] < _l)
    # for __pns in chain([set()], AntiChain(_sdag, 1, _id, set(_i for _i in _sdag.nodes() if _sdag.out_degree(_i) < _od and True))):
    for __pns in chain([set()], AntiChain(_sdag, 1, _id, __sub_nodes)):
        __ndag = nx.DiGraph(_sdag)
        # __ndag.add_node(__nid)
        __ndag.add_nodes_from([(__nid, {'d': 1 if len(__pns) == 0 else max([_sdag.nodes[__pns_x]['d'] for __pns_x in __pns]) + 1})]) 
        __ndag.add_edges_from([(__px, __nid) for __px in __pns]) 
        if _nnum == 1:
            yield __ndag
        else:
            for __nrdag in FanInFanOut(__ndag, _nnum - 1, _id, _od, _l):
                yield __nrdag


if __name__ == "__main__":
    id, od = 1, 2

    for nnum in range(2, 20):
        ld = nnum - 2 # math.ceil(math.log2(nnum + 1)) + 1
        st = time.time()
        dag_num = 0
        gx = nx.DiGraph()
        gx.add_nodes_from([(0, {'d': 1,})])

        for rdag in FanInFanOut(gx, nnum - 1, id, od, 4):
        # for rdag in FanInFanOut(gx, nnum - 1,  int(pow(nnum, 2) / 4) - 1,  int(pow(nnum, 2) / 4) - 1):
            # __id = max(dict(rdag.in_degree()).values())
            # __od = max(dict(rdag.out_degree()).values())
            # assert __id <= id and __od <= od
            dag_num += 1
        et = time.time()
        print(f"node_num:{nnum}\t--{et-st:.12f}\t--{dag_num}")

        pd.DataFrame([{'nn':nnum, 'dn':dag_num, 'ct': et - st}]).to_csv(f'FIO_id_{id}_od_{od}.csv', index=False,  header=False, mode='a')
