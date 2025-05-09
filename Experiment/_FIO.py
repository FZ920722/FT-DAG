#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MainHead import *


def FanInFanOut(_nnum, _id=float('Inf'), _od=float('Inf'), _ld=1, _lu=float('Inf')):
    if 0 < _ld <= _lu and _ld <= _nnum:
        if _ld == 1:
            __rdag = nx.DiGraph()
            __rdag.add_nodes_from([(_i, {'d': 1}) for _i in range(_nnum)])
            yield __rdag
        for __sub_dag in FanInFanOut(_nnum - 1, _id, _od, max(1, _ld - 1)):
            __rn_i = __sub_dag.number_of_nodes()
            for __pns in nx.antichains(__sub_dag):
                if len(__pns) <= _id and max([__sub_dag.out_degree(__pns_x) for __pns_x in __pns], default=0) < _od:
                    __max_d = max([__rdag.nodes[__pns_x]['d'] for __pns_x in __pns], default=0) + 1
                    __rdag = nx.DiGraph(__sub_dag)
                    __rdag.add_nodes_from([(__rn_i, {'d':__max_d})])
                    __rdag.add_edges_from([(__pns_x, __rn_i) for __pns_x in __pns])
                    if _ld <= max([__snd['d'] for __sni, __snd in __rdag.nodes(data=True)]):
                        yield __rdag


if __name__ == "__main__":
    _id, _od = 1, 2
    for nnum in range(3, 100):
        st = time.time()
        _ld = max(1, nnum - 4)
        dag_num = 0
        for rdag in FanInFanOut(nnum, _id, _od, _ld):
            dag_num += 1
        et = time.time()
        print(f"node_num:{nnum}\t--{et-st:.5f}\t--{dag_num}")
