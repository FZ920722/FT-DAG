#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainHead import *
global TIMES, TIME_SCS, TIME_MS, TIME_MC, DBCONN, TCONN, TCURS, FTable, NNUM
TIMES, TIME_SCS, TIME_MS, TIME_MC = 0, 0, 0, 0
N_N, N_SCS, N_MS, N_MC = 0, 0, 0, 0

# sys.setrecursionlimit(2000)
def BSG(_n: int, _Xp:int=1):
    assert _n > 0
    _Xs = 2 * _Xp
    if _n <= _Xs:
        yield (_n,)
    else:
        for _ss in BSG(_n - _Xs, _Xs):
            yield (_Xs,) + _ss


def CSG(_n: int, _Xp:int=1):
    assert _n > 0
    _Xs = _Xp + 1
    if _n <= _Xs:
        yield (_n,)
    else:
        for _ss in CSG(_n - _Xs, _Xs):
            yield (_Xs,) + _ss


def NSG(_n: int, _lr: tuple, _sr: tuple, _mr: tuple, _id:int, _od:int, _Xs:int=0):
    """ 尾结点数小者优先；"""
    if 1 <= _lr[0] <= _lr[1] and 1 <= _sr[0] <= _sr[1] and _lr[0] * _sr[0] <= _n <= _lr[1] * _sr[1]:
        __Xld, __Xlu = _sr[0], _sr[1]
        if 0 < _od and 0 < _id:
            __Xld = max(__Xld, math.ceil(_Xs/_od))
            # S1. 递归生成长shape：
            if 1 < _lr[1]:
                for __Xp, __Xl in map(lambda __x: (_n - __x, __x), range(__Xld, __Xlu + 1)):
                    __l_md, __l_mu = __Xl,               min(_id, __Xp - (_lr[0] - 1) + 1) * __Xl
                    __p_md, __p_mu = max(0, _lr[0] - 2), int(pow(__Xp, 2) / 4)
                    if __Xp * _od >= __Xl and __l_md  + __p_md <= _mr[1] and _mr[0] <= __p_mu + __l_mu:
                        __nlr = (max(1, _lr[0] - 1),        _lr[1] - 1)
                        __nsr = (_sr[0],                    min(_sr[1], __Xp))
                        __nmr = (max(0, _mr[0] - __l_mu),   _mr[1] - __l_md)
                        for __s_p in NSG(__Xp, __nlr, __nsr, __nmr, _id, _od, __Xl):
                            yield __s_p + (__Xl, )
        # S2. 不在递归，直接返回本层结点数量作为shape的第一层；
        if _mr[0] <= 0 <= _mr[1] and _lr[0] <= 1 <= _lr[1] and __Xld <= _n <= __Xlu:
            yield (_n, )

def TempShape(_n, _max_s, _ld, _lu, _sd, _su, _id, _od):
    assert _n >= _max_s
    if _n == _max_s:
        yield (_n, )
    else:
        __rn = _n - _max_s
        for __shx in NSG(__rn, (max(_ld - 1, 1), _lu - 1), 
                               (_sd, _su), 
                               (0, int(pow(__rn, 2) / 4)), _id, _od):
            for _rsh in set([__shx[:__index] + (_max_s, ) + __shx[__index:] for __index in range(len(__shx) + 1)]):
                yield  _rsh

# ###########################################
def NullDAG(_n:int):    # 零图；
    """ 743.4 - 4.4 """
    __rdag = nx.DiGraph()
    __rdag.add_nodes_from([(_i, {'d': 1, 'Equ': 0, 'P': set(), 'S': set(), 'A': set(), 'D': set()}) for _i in range(_n)])
    __rdag.graph |= {'s': (_n, ), 'c': f"({'|'.join('t' * _n)})", 'dt': 'MC' if _n > 1 else 'SCS', 'jl': 0, 'hl': 1, 'm': 0, 'w': _n, 'id': 0, 'od': 0}
    return __rdag


def ChainDAG(_n:int):    # 零图；
    """ 743.4 - 4.4 """
    __rdag = nx.DiGraph()
    __rdag.add_nodes_from([(_i, {'d': 1 + _i, 'Equ': _i, 
         'P': {_i - 1,}              if 0 < _i      else set(),
         'A': set(range(_i))         if 0 < _i      else set(), 
         'S': {_i + 1,}              if _i < _n - 1 else set(),   
         'D': set(range(_i + 1, _n)) if _i < _n - 1 else set()
         }) for _i in range(_n)])
    __rdag.add_edges_from([(_i - 1, _i) for _i in range(1, _n)])
    __rdag.graph |= {'s': tuple([1 for _ in range(_n)]), 
                     'c': f"({'-'.join('t' * _n)})", 
                     'dt': 'MS' if _n > 1 else 'SCS', 
                     'jl': 1 if _n > 1 else 0, 
                     'id': 1 if _n > 1 else 0, 
                     'od': 1 if _n > 1 else 0, 
                     'hl': _n, 'm': _n - 1, 'w': 1}
    return __rdag

def DagWidth(_cdag, _ans:set=set()):
    """ 默认使用霍普克洛夫特-卡普算法计算最大匹配 -- 52.463330 -- 6.19 """
    _ons = _cdag.nodes() - _ans
    __dn, __tg, __us, __te = len(_ons), nx.DiGraph(), list(), list()
    for __pi, __o_d in map(lambda __x: (f"p{__x}", _cdag.nodes[__x]['D'] - _ans), _ons):
        if len(__o_d) > 0:
            __us.append(__pi)
            for __si in map(lambda __y: f"s{__y}", __o_d):
                __te.append( (__pi, __si) )
    __tg = nx.DiGraph(__te)
    __ret = __dn - int(len(nx.algorithms.bipartite.matching.hopcroft_karp_matching(__tg, __us)) / 2)
    return __ret

def NCDete(_s:tuple, _id:int, _od:int):
    for __i in range(1, len(_s)):
        if _s[__i - 1] * _od < _s[__i]:
            return False
    return True

def MaxJl(_s:tuple):
    __max_jl, __l = 0, len(_s)
    if __l > 1:
        for __in, __sx in enumerate(_s):
            if __sx > 1:
                __max_jl = __l - __in - 1
                break
        __max_jl = max(1, __max_jl)
    return __max_jl

def MinHl(_s:tuple):
    __min_hl, __l = 1, len(_s)
    for __sx in _s:
        if __sx > 1:
            break
        __min_hl += 1
    return min(__l, __min_hl)
    
def NCDA(_tdag):
    return {"c":  _tdag.graph['c'],   "m": _tdag.graph['m'],   "w": _tdag.graph['w'],    "id":  _tdag.graph['id'],
            "od":  _tdag.graph['od'], "jl": _tdag.graph['jl'], "hl":  _tdag.graph['hl'], "dag": pickle.dumps(_tdag)}

# L : 0 ing写数据； # L : 1 可读数据；  # L : 2 空数据；
def __DB_Data(_shash, _s:tuple, _id:int, _od:int, _jl:int, _hl:int, _md:int, _mu:int, _wd:int, _wu:int, _dt:str):
    global FTable, TCURS
    __GDGen, __tname = None, _dt + _shash
    if _dt == 'SCS':
        __GDGen = SCSGen(_s, _id, _od, _jl, _hl, _md, _mu, _wd, _wu)
    elif _dt == 'MC':
        __GDGen = MCGen(_s, _id, _od, _jl, _hl, _md, _mu, _wd, _wu)
    elif _dt == 'MS':
        __GDGen = MSGen(_s, _id, _od, _jl, _hl, _md, _mu, _wd, _wu)
    else:
        assert False

    TCURS.execute(f'SELECT * FROM {FTable} WHERE ts = "{__tname}"')
    __db_data = TCURS.fetchall()

    if len(__db_data) > 0:
        __red = dict(__db_data[0])
        # (1) 包含关系，且数据违被锁定，则利用数据库数据返回
        __query = f'SELECT dag FROM {__tname} WHERE od <= {_od} AND id <= {_id} AND jl <= {_jl} AND hl >= {_hl} AND m >= {_md} AND {_mu} >= m AND w >= {_wd} AND {_wu} >= w'
        if __red['wd'] <= _wd <= _wu <=__red['wu'] and __red['id'] >= _id and __red['jl'] >= _jl and \
            __red['md'] <= _md <= _mu <=__red['mu'] and __red['od'] >= _od and __red['hl'] <= _hl and __red['L'] == 1:
            if __red['L'] != 2:
                TCURS.execute(__query)
                for __DbDag in TCURS.fetchall():
                    yield pickle.loads(__DbDag[0])
        # (2) 其他，与数据库无关
        else:
            # if __red['L'] != 2:
            for __rd in __GDGen:
                yield __rd
    else:           # (3) 没有数据则新建表；给一个锁位L = 0, 录入数据后打开， 新建的时候锁上，录入后打开 L = 1；
        TCURS.execute(f"INSERT INTO {FTable} (ts, wd, wu, md, mu, jl, hl, id, od, L) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (__tname, _wd, _wu, _md, _mu, _jl, _hl, _id, _od, 0))
        __data_buff = []
        for __rd in __GDGen:
            yield __rd
            __data_buff.append(NCDA(__rd))

        if len(__data_buff) > 0:
            TCURS.execute(f'''CREATE TABLE  {__tname} (dag BLOB NOT NULL, c BLOB PRIMARY KEY,  w INTEGER NOT NULL,  
                            m INTEGER NOT NULL, jl INTEGER NOT NULL, hl INTEGER NOT NULL, od INTEGER NOT NULL,id INTEGER NOT NULL);''')
            TCURS.executemany(f""" INSERT INTO {__tname} (c, m, w, id, od, jl, hl, dag) VALUES (?, ?, ?, ?, ?, ?, ?, ?) """, 
                                [(__dx['c'], __dx['m'], __dx['w'], __dx['id'], __dx['od'], __dx['jl'], __dx['hl'], __dx['dag'],) for __dx in __data_buff])
            TCURS.execute(f'UPDATE {FTable}  SET L = 1 WHERE ts = "{__tname}"')
        else:
            TCURS.execute(f'UPDATE {FTable}  SET L = 2 WHERE ts = "{__tname}"')
    __GDGen.close()


def NC(_s:tuple, _id:int, _od:int, _jl:int, _hl:int, _md:int, _mu:int, _wd:int, _wu:int, _ms:bool=True, _mc:bool=True):
    global TIME_SCS, TIME_MS, TIME_MC, N_SCS, N_MS, N_MC
    # (1) 基于shape的 参数配置：
    __n, __l, __nid, __nod  = sum(_s), len(_s), 0, 0
    __njl,  __nhl           = min(MaxJl(_s), _jl),  max(MinHl(_s), _hl)
    __nwd,  __nwu           = max(max(_s),   _wd),  min(__n - __l + 1, _wu)

    # iod不能超过宽度；宽度可以大于iod
    if __l > 1:
        __nid = min((__n - _s[-1]) - (__l - 2) + 1, _id, __nwu) # min(__sn - __ss[-1] - __sl + 2, _id)
        __nod = min((__n - _s[0]) - (__l - 2) + 1, _od, __nwu)  # min(__sn - __ss[0] - __sl + 2, _od)

    __nmd = max(_md, __n - _s[0])
    __nmu = min(_mu, int(pow(__n, 2) / 4), __nid * (__n - _s[0]), __nod * (__n - _s[-1]))

    if NCDete(_s, __nid, __nod) and 0 <= __nid and 0 <= __nod and 0 <= __njl and\
              0 <= __nmd <= __nmu and 1 <= __nwd <= __nwu and 1 <= __nhl <= __l:    # 基础条件
        __shash = f"s{hashlib.md5(pickle.dumps(_s)).hexdigest()}"

        """ (1) SCS DAG 生成 """
        # if NNUM == __n:
        scst = time.time()
        if (__n == 1) or (__n > 3 and _s[0] > 1 and __l > 1 and __nid > 1):
            for __rd in __DB_Data(__shash, _s, __nid, __nod, __njl, __nhl, __nmd, __nmu, __nwd, __nwu, 'SCS'):
                yield __rd

        # if NNUM == __n:
        TIME_SCS += time.time() - scst
            
        """ (2) MC DAG 生成 """
        # if NNUM == __n:
        mct = time.time()
        if _mc and _s[0] > 1:
            for __rd in __DB_Data(__shash, _s, __nid, __nod, __njl, __nhl, __nmd, __nmu, __nwd, __nwu, 'MC'):
                yield __rd

        # if NNUM == __n:
        TIME_MC += time.time() - mct

        """ (3) MS DAG 生成 """
        # if NNUM == __n:
        mst = time.time()
        if _ms and __l > 1:
            for __rd in __DB_Data(__shash, _s, __nid, __nod, __njl, __nhl, __nmd, __nmu, __nwd, __nwu, 'MS'):
                yield __rd
        # if NNUM == __n:
        TIME_MS += time.time() - mst

""" (1) SCS-DAG """
def CCD(_tdag, _ln:int, _id:int, _od:int, _jl:int, _hl:int):
    """ 多连通分量检测、约束条件判定 以及 各连通分量的可连接结点； """  # -53   0.59
    # (1) 分量的必连结点； m:必连结点[l-jl, ht)  ：(l - jl <= 深度 < _hl) & (出度 == 0)；
    # (2) 分量的可选结点； n:可选结点；          ：_hl <= 深度 <= _pln
    # C1. t-dag必须存在至少一个p结点：           ：(d == _ln - 1) & (od < _od);
    if len([_i for _i, _d in _tdag.nodes(data=True) if _d['d'] == _ln - 1 and len(_d['S']) < _od]) > 0:
        __min_pn, __rdg = 0, tuple()
        for __scs_cx in nx.weakly_connected_components(_tdag):
            __cx_d, __cx_ms, __cx_ns, __cx_md = dict(), set(), set(), set()
            for __ni, __nd in map(lambda __x: (__x, _tdag.nodes[__x]), __scs_cx):
                # S2 检查各分量中结点的出度小于od的分量并归类；
                # C2 保证 所有结点的出度必须不超过od，深度不小于min(_hl, _ln - _jl)；
                if len(__nd['S']) > _od and __nd['d'] < min(_hl, _ln - _jl):
                    return False
                elif __nd['d'] <= _ln - 1:
                    __cx_d[__ni] = {'d': __nd['d'], 'Equ': __nd['Equ']} | {_tx: set() | __nd[_tx] for _tx in ['P', 'S', 'A', 'D']}
                    # t1 必连结点；
                    if len(__nd['S']) == 0 and __nd['d'] < _hl: 
                        __cx_ms.add(__ni)
                        __cx_md |= __nd['D'] | __nd['A']
                    # t2 可连连结点；
                    elif len(__nd['S']) < _od and _ln - _jl <= __nd['d']:
                        __cx_ns.add(__ni)

            __min_pn += max(1, len(__cx_ms))    # 必连结点的数量

            # C2. 分量数 <= _id; and 各分量必须有可连接的结点；
            if len(__cx_d) > 0 and __min_pn <= _id:
                __rdg += ( (tuple([(__mni, __cx_d[__mni]) for __mni in __cx_ms]), 
                            tuple([(__nni, __cx_d[__nni]) for __nni in  __cx_ns - __cx_md])), )
            else:
                return False
        return __rdg
    return False

def NPlGen(_pdg, _pln:int, _min_pn:int,  _max_pn:int, _pn:bool=False):
    """ 循环获取前驱结点-5.17  - 0.06 """
    if 1 <= _min_pn <= _max_pn:
        __smns, __snns = _pdg[0]
        __rpn_min = sum([max(1, len(__y)) for __y, _ in _pdg[1:]])
        __rpn_max = sum([len(__y) + len(__z) for __y, __z in _pdg[1:]])
        __tpn_max = _max_pn - __rpn_min - len(__smns)
        __tpn_min = max(1, _min_pn - __rpn_max - len(__smns))
        __anti_gen = NxAnti(sorted(__snns, key=lambda x: x[0]), __tpn_min, __tpn_max)

        if len(__smns) > 0:     # 有mns 则下限为1 否则为0；
            __anti_gen = chain(__anti_gen, [tuple()])
        for __rpnx in map(lambda __x: __x + __smns, __anti_gen):
            __rpnn, __pnlb =  len(__rpnx), (_pln in set([__rn['d'] for __ri, __rn in __rpnx])) or _pn
            if len(_pdg) == 1:   # 不递归
                if __pnlb:
                    yield __rpnx
            else:                # 递归
                for __spn in NPlGen(_pdg[1:], _pln, max(__rpn_min, _min_pn - __rpnn), min(__rpn_max, _max_pn - __rpnn), __pnlb):
                    yield __spn + __rpnx

def SCSGen(_s:tuple, _id:int, _od:int, _jl:int, _hl:int, _md:int, _mu:int, _wd:int, _wu:int):
    """ """
    # S0. 关键参数初始化；
    __n, __l = sum(_s), len(_s)

    # (1) Trivial DAG
    if _hl == __n == 1 and _md <= 0 <= _mu and _wd <= 1 <= _wu:
        yield NullDAG(__n)

    # (2) Other DAG; 除了 T-DAG ；l和s[0]必须大于1；
    if __l > 1 and _hl > 0 and _s[0] > 1 and _id > 0 and _od > 0 and _jl > 0:
        __ss = _s[:-1] if _s[-1] == 1 else _s[:-1] + (_s[-1] - 1, )
        __sn, __sl, __cbuff = sum(__ss),                        len(__ss),      set()
        __smd, __smu        = _md - _id,                        _mu - 1  
        __swd, __swu        = _wd - 1 if _s[-1] > 1 else _wd,   _wu

        # S1. 生成满足条件的子DAG；
        for __sscs in NC(__ss, _id, _od, _jl, min(_hl, __l - _jl), __smd, __smu, __swd, __swu, _ms=False, _mc=True if _id > 1 else False):

            # S2. 子DAG约束条件判定与连通分量分组: (( m:必连结点 [l-jl, ht) ), ( n:可选结点 ),)；
            if __cng:= CCD(__sscs, __l, _id, _od, _jl, _hl):

                __scss_nodes, __scss_edges = tuple(__sscs.nodes(data=True)), tuple(__sscs.edges())

                # S3. 基于满足条件的子DAG，进一步穷举所有连接前驱(后期优化) 
                __v_sink = {__ni: __nd for __ni, __nd in __scss_nodes if len(__nd['S']) == 0}
                __vsi_set = set(__v_sink.keys())
                __ss_cn = sum([max(1, len(__mns)) for __mns, _ in __cng])       # 必连结点的数量：
                __ss_w, __ss_en = __sscs.graph['w'], __sscs.number_of_edges()   # sscs的边数量：
                for __pns in NPlGen(__cng, __l - 1, max(__ss_cn, _md - __ss_en), min(_id, _mu - __ss_en)):

                    # S4. 生成结果验证
                    __pni_set = set(__pnsx[0] for __pnsx in __pns)
                    if __pni_set != __vsi_set:
                        __pi_ance = __pni_set | set().union(*[__pid[1]['A'] for __pid in __pns])
                        __scs_w = max(__ss_w, DagWidth(__sscs, __pi_ance) + 1) if _s[-1] > 1 else __ss_w
                        if _wd <= __scs_w <= _wu:
                            __new_edges = __scss_edges + tuple((__pi, __sn) for __pi in __pni_set)
                            __new_nodes, __X_er, __V_er, __E_er, __E_Ad = tuple(), dict(), list(), dict(), dict() 
                            for __nni, __nnd in __scss_nodes + ((__sn, {'d': __l, 'Equ': None, 'P': __pni_set, 'S': set(), 'A': __pi_ance, 'D': set()}),):
                                __temp_nd =  {'d': __nnd['d'], 'Equ': __nnd['Equ']} | {_tx: set() | __nnd[_tx] for _tx in ['P', 'S', 'A', 'D']}
                                if __nni in __pi_ance:
                                    __temp_nd['D'].add(__sn)    # S-4.1 祖先的后代加入ni
                                    if __nni in __pni_set:      # S-4.2 其中前驱的后继加入ni
                                        __temp_nd['S'].add(__sn)
                                __nel = (frozenset(__temp_nd['P']), frozenset(__temp_nd['S']))
                                if __nel not in __V_er:
                                    __V_er.append(__nel)
                                __temp_nd['Equ'] = __V_er.index(__nel)
                                __new_nodes += ( (__nni, __temp_nd), )
                                # S1 等价约简
                                __E_er[__nni] = __temp_nd['Equ']
                                if __E_er[__nni] not in __X_er:
                                    __X_er[__E_er[__nni]] = {'n': 0, 'd': __temp_nd['d'], 'id': len(__temp_nd['P']), 'od': len(__temp_nd['S'])}
                                    __E_Ad[__E_er[__nni]] = set()
                                __X_er[__E_er[__nni]]['n'] += 1

                            __D_er, __E_Dn = defaultdict(set), len(__E_Ad)
                            for __Xi, __Xd in __X_er.items():
                                __D_er[(__Xd['n'], __Xd['d'], __Xd['id'], __Xd['od'])].add(__Xi)
                            # S2. 等价约简的caonnel图（color目测还有优化空间）
                            for __nx, __ny in __new_edges:
                                __E_Ad[__E_er[__nx]].add(__E_er[__ny])

                            __E_Pn = pn.Graph(__E_Dn, directed=True, adjacency_dict=__E_Ad)
                            __E_Pn.set_vertex_coloring([__D_er[__k] for __k in sorted(__D_er.keys())])
                            __Cal = pn.canon_label(__E_Pn)

                            # S3 加入等价结点数量的certifion计算；
                            __E_Ct = np.zeros((__E_Dn, __E_Dn), dtype=int)
                            for __ep, __ess in __E_Ad.items():
                                __E_Ct[__Cal.index(__ep), __Cal.index(__ep)] = __X_er[__ep]['n']
                                for __es in __ess:
                                    __E_Ct[__Cal.index(__ep), __Cal.index(__es)] = 1
                            __cx = hashlib.md5(__E_Ct.tobytes()).hexdigest()

                            # S5. 生成数据整理输出
                            if __cx not in __cbuff:
                                __cbuff.add(__cx)
                                __scs = nx.DiGraph()
                                __scs.add_nodes_from(__new_nodes)                            
                                __scs.add_edges_from(__new_edges)
                                __scs.graph = {'c': __cx, 's': _s, 'w':__scs_w, 'dt': 'SCS', 'm': __sscs.graph['m'] + len(__pns),
                                               'od': max([__sscs.graph['od'],] + [len(__pnd['S']) + 1 for __pni, __pnd in __pns]),
                                               'jl': max([__sscs.graph['jl'],] + [__l - __pnd['d'] for __pni, __pnd in __pns]),
                                               'id': max(__sscs.graph['id'], len(__pns)), 'hl': min([__v_sink[__rvsi]['d'] for __rvsi in __vsi_set - __pni_set])}
                                yield __scs


def NxAnti(_ns:tuple, _nd:int, _nu:int):    
    if  1 <= _nd <= _nu:
        __equ_buff = set()
        for __nid in _ns:
            if __nid[1]['Equ'] not in __equ_buff:
                __equ_buff.add(__nid[1]['Equ'])
                if _nd == 1:
                    yield (__nid,)
                if _nu > 1:
                    __rns = tuple([__sid for __sid in _ns if __sid[0] > __nid[0] and __sid[0] not in set(__nid[1]['A'] | __nid[1]['D'])])
                    for __s_an in NxAnti(__rns, max(1, _nd - 1), _nu - 1):
                        yield (__nid,) + __s_an


""" (2) MC-DAG """
def MCGen(_s:tuple, _id:int, _od:int, _jl:int, _hl:int, _md:int, _mu:int, _wd:int, _wu:int):
    """ """
    __n, __l = sum(_s), len(_s)
    if _hl == __l == 1 and _md <= 0 <= _mu and _wd <= __n <= _wu: # (1) 零图DAG_MC-DAG
        yield NullDAG(__n)

    if 1 < __l and 1 <= _id and 1 <= _od and  1 <= _jl:
        """ S1 shape 分解； """
        for __lgn in range(_s[0] if _hl > 1 else 1, _s[0] + 1):   # lg的总数；根据hl计算l的上限 min(_s[:_hl]) # _s[0] - __lgn ：平凡shape的数量；
            __ogn = _s[0] - __lgn
            for __tlg_d in number_combine(__lgn, (1 if _s[0] > __lgn else 2, min(_s[1], __lgn, min(_s[:_hl]))), (1, __lgn)):
                __tlg = tuple(((__gd, ), __gn) for __gd, __gn in Counter(__tlg_d).items())
                for __lg in LG(__tlg, _s[1:], _od, _hl):
                    __mc_sg = tuple(_x for _x, _i in __lg for _ in range(_i))
                    # """ S2 根据sg 分解，加入边与宽的条件约束，判定是否满足约束条件并更新； """
                    """ S3 DAG """
                    __mc_wd, __mc_wu = sum([max(__mcs) for __mcs in __mc_sg]), sum([sum(__mcs) - len(__mcs) + 1 for __mcs in __mc_sg])
                    __mc_md, __mc_mu = sum([sum(__mcs) - __mcs[0] for __mcs in __mc_sg]), sum([int(pow(sum(__mcs), 2) / 4) for __mcs in __mc_sg])
                    for __rdag_mc in RG(tuple(Counter(__mc_sg).items()), _id, _od, _jl, _hl, max(__mc_md, _md), min(__mc_mu, _mu), 
                                                                                             max(__mc_wd, _wd - __ogn), min(_wu - __ogn, __mc_wu)):
                        ret = EM(__rdag_mc + (NullDAG(__ogn), ))    # ret = EM(__rdag_mc)
                        yield ret

def number_combine(_NNum: int, _LRange: tuple, _SRange: tuple): 
    """ 基于NNUM的组合穷举 """
    if 1 <= _LRange[0] <= _LRange[1] and 1 <= _SRange[0] <= _SRange[1] and _SRange[0] * _LRange[0] <= _NNum <= _SRange[1] * _LRange[1]:
        for __gxn in range(max(1, _SRange[0]), min(_SRange[1], _NNum - (_LRange[0] - 1) * _SRange[0]) + 1):
            __grn = _NNum - __gxn
            if __grn == 0:
                yield (__gxn, )
            else:
                for __sub_comb in number_combine(__grn, (max(_LRange[0] - 1, 1), _LRange[1] - 1), (__gxn, _SRange[1])): 
                    yield __sub_comb + (__gxn, )

def LS(_sg, _sn):
    """ lo 分组"""
    __tsd, __tsn = _sg[0]
    for __tlsn in range(min(_sn, __tsn) + 1): 
        __rlsg = ((__tsd, __tlsn),) if __tlsn > 0 else tuple()
        __rnsg = ((__tsd, __tsn - __tlsn),) if __tsn > __tlsn else tuple()
        if __tlsn == _sn or len(_sg) == 1:
            yield __rlsg, __rnsg + _sg[1:]
        else:
            for __srlsg, __srnsg in LS(_sg[1:], _sn - __tlsn):
                yield __rlsg + __srlsg, __rnsg + __srnsg

def NA(_sg, _an:int, _od:int):
    """ 把an个结点分给_sg的每个组，每个shape至少1个, 最多分_od * __tsd[-1]； """
    __min_an = sum(__n for __s, __n in _sg)
    if __min_an == _an:
        yield tuple((__sx + (1, ), __sn) for __sx, __sn in _sg)
    elif __min_an < _an:
        __tsd, __tsn = _sg[0]
        if len(_sg) == 1:       # 不再递归，_an全部分发
            for __lg_data in number_combine(_an, (__tsn, __tsn), (1, min(_an, _od * __tsd[-1]))):
                yield tuple((__tsd + (__nn, ), __sn) for __nn, __sn in Counter(__lg_data).items())
        else:                   # 继续递归
            for __t_an in range(__tsn, _an - (__min_an - __tsn) + 1):
                for __lg_data in number_combine(__t_an, (__tsn, __tsn), (1, min(__t_an, _od * __tsd[-1]))):
                    __t_sg = tuple((__tsd + (__nn, ), __sn) for __nn, __sn in Counter(__lg_data).items())
                    for _new_sg in NA(_sg[1:], _an - __t_an, _od):
                        yield _new_sg + __t_sg
    else:
        assert False


def LG(_psg, _s:tuple, _od:int, _hl:int):
    """ 将_s0的结点加入_psg中，不再重新分组合并 IO都是分组格式的（shape, num） """
    for __newsg in NA(_psg, _s[0], _od):    # (1) s0结点分配,每个组必须有一个后继;
        if len(_s) == 1:                    # (2) s数据处理完成直接返回；
            yield __newsg

        else:                               # (3) s数据还有剩余，继续递归；
            # __tl =       # 当前shape的长度；
            # (3.1) __newsg的L/O分组,返回__lsg, __nsg; lsg的范围[1, min(len(__newsg), _s[1])]，剩下的给__nsg，可以为空；
            if len(__newsg[0][0])  < _hl:
                for __tu_sg in LG(__newsg, _s[1:], _od, _hl):
                    yield __tu_sg
            else:
                for __lsg, __nsg in LS(__newsg, min(sum(__nsn for __nss ,__nsn in __newsg), _s[1])):
                    if len(__lsg) > 0:
                        if sum(__lsn for __lss, __lsn in __lsg) == 1:
                            yield ((__lsg[0][0] + _s[1:], 1),) + __nsg
                        else:
                            for __sub_sg in LG(__lsg, _s[1:], _od, _hl):
                                yield __sub_sg + __nsg

def RG(_sg:tuple, _id:int, _od:int, _jl:int, _hl:int, _md:int, _mu:int, _wd:int, _wu:int):
    """ """
    __sgmd, __sgmu, __sgwd, __sgwu = 0, 0, 0, 0
    for __sgs, __sgn in _sg:
        __sgsn, __sgsl = sum(__sgs), len(__sgs)
        __sgwd += max(__sgs) * __sgn
        __sgwu += (__sgsn - __sgsl + 1) * __sgn
        __sgmd += (__sgsn - __sgs[0]) * __sgn
        __sgmu += min([int(pow(__sgsn, 2) / 4), (__sgsn - __sgs[-1]) * _od, (__sgsn - __sgs[0]) * _id]) * __sgn

    __nmd, __nmu, __nwd, __nwu = max(__sgmd, _md), min(__sgmu, _mu), max(__sgwd, _wd), min(__sgwu, _wu)
    # __njl,  __nhl   = min(MaxJl(_s), _jl), max(MinHl(_s), _hl)    # iod不能超过宽度；宽度可以大于iod

    if __nmd <= __nmu and __nwd <= __nwu:
        __lgs, __lgn = _sg[0]
        __lgsn, lgsl = sum(__lgs), len(__lgs)
        __lgwd, __lgwu = max(__lgs), __lgsn - lgsl + 1
        __lgmd, __lgmu = sum(__lgs[1:]), min(int(pow(__lgsn, 2) / 4), (__lgsn - __lgs[-1]) * _od, (__lgsn - __lgs[0]) * _id)
        __rgmd, __rgmu = __sgmd - __lgmd * __lgn, __sgmu - __lgmu * __lgn
        __rgwd, __rgwu = __sgwd - __lgwd * __lgn, __sgwu - __lgwu * __lgn

        __TncGen = NC(__lgs, _id, _od, _jl, _hl, int((_md - __rgmu) / __lgn), _mu - __rgmd,   
                      int((_wd - __rgwu)/__lgn),  _wu - __rgwd, _ms=True, _mc=False)

        for __ldagg in combinations_with_replacement(__TncGen, __lgn):
            __tmn, __twn = sum(__ldx.number_of_edges() for __ldx in __ldagg), sum(__ldx.graph['w'] for __ldx in __ldagg)

            if len(_sg) > 1:
                for __odag in RG(_sg[1:], _id, _od, _jl, _hl, _md - __tmn, _mu - __tmn, _wd - __twn, _wu - __twn):
                    yield __ldagg + __odag
            else:
                if _md <= __tmn <= _mu and _wd <= __twn <= _wu:
                    yield __ldagg


""" (3) MS-DAG ：主生成部分可分为: p-segement & l-segment """
def MSGen(_s:tuple, _id:int, _od:int, _jl:int, _hl:int, _md:int, _mu:int, _wd:int, _wu:int):
    global TIMES
    st = time.time()    
    __n, __l = sum(_s), len(_s)

    if __n == __l == _hl and _md <= __n - 1 <= _mu and _wd <= __n - 1 <= _wu:
        yield ChainDAG(__n)
    else:
        for __ls_ln in range(1, len(_s)): # ls层数：从小到大；
            __ps_s, __ls_s = _s[:__ls_ln], _s[__ls_ln:]
            __pn,  __ln,  __pl,  __ll   = sum(__ps_s), sum(__ls_s), len(__ps_s), len(__ls_s)
            # (1) pl 不能是MS-DAG 根图一定是MS DAG: S[0] == 1 AND L > 1 逆否命题：S[0] > 1 or L == 1 
            if (1 < __ps_s[0] or __pl == 1) and (__ls_s[0] <= _od) and (__ps_s[-1] <= _id):
                __lmd, __lmu = __ln - __ls_s[0], int(pow(__ln, 2) / 4)
                __pwd, __lwd, __pwu, __lwu  = max(__ps_s), max(__ls_s), __pn - __pl + 1, __ln - __ll + 1
                # (2.1) lg的w是否能达标，可以则上限不超标即可，否则pg必须达标；
                __npmd, __npmu = _md - __lmu - __pwu * __ls_s[0], _mu - __lmd - __ps_s[-1] * __ls_s[0]
                __npwd = __pwd if _wd <= __lwu and __lwd <= _wu else max(_wd, __pwd)
                for __ps_dg in NC(__ps_s, _id, _od, _jl, __pl - _jl + 1, __npmd, __npmu,
                                __npwd, _wu, _ms=False, _mc=True if _id > 1 else False):  # p-segment
                    __psd_sn = len({__i for __i, __d in __ps_dg.nodes(data=True) if len(__d['S']) == 0})                 
                    if __psd_sn <= _id:
                        __nm_num = __psd_sn * __ls_s[0] + __ps_dg.number_of_edges()
                        # (2) pg达标则上限不超标即可，否则lg必须达标(即上下限严格约束)；
                        __nlmd, __nlmu = _md - __nm_num, _mu - __nm_num
                        __nlwd = __lwd if _wd <= __ps_dg.graph['w'] <= _wu else max(_wd, __lwd)
                        for __ls_dg in NC(__ls_s, _id, _od, _jl, max(1, _hl - __ls_ln), __nlmd, __nlmu, __nlwd, _wu, _ms=True, _mc=True):   # l-segment
                            __rpl = FC(__ps_dg, __ls_dg)
                            if _wd <= __rpl.graph['w'] <= _wu:
                                TIMES += time.time() - st
                                yield __rpl
                                st = time.time()

def EM(_sg:tuple):
    """ # -52.274558 -- 1.91 """
    __rd, __rdns, __rdes, __equ_nl, __rm, __rw = nx.DiGraph(), list(), list(), [None, ], 0, 0 
    __rs, __rc, __rjl, __rhl, __rid, __rod = list(), list(), list(), list(), list(), list()

    for __ti, __td in enumerate(_sg):
        __rn = len(__rdns)              # (1) sd 结点id 更新；
        __td_nd = {__i : __i + __rn for __i in __td.nodes()}
        __tdns = [(__td_nd[__i], {'d': __d['d'], 'Equ': __d['Equ']} | {_tx:  set(__td_nd[__xi] for __xi in __d[_tx]) for _tx in ['P', 'S', 'A', 'D']})  for __i, __d in __td.nodes(data=True)]
        __rdes += [(__td_nd[__ex], __td_nd[__ey]) for __ex, __ey in __td.edges()]
        if len(__tdns) == 1:            # (2) sd 结点的 'Equ' 更新；
            __tdns[0][1]['Equ'] = 0     # 平凡分量结点equ恒为0
        else:
            for __tdni, __tdnd in __tdns:
                __nel = f"{__ti}_{__tdnd['Equ']}"
                if __nel not in __equ_nl:
                    __equ_nl.append(__nel)
                __tdnd['Equ'] = __equ_nl.index(__nel)
        __rdns += __tdns
        # (*) DAG的新边结点的d/id/od更新(无)
        # ### graph 数据更新 ### #
        __rs.append(__td.graph['s']),   __rc.append(__td.graph['c']),   __rjl.append(__td.graph['jl'])
        __rhl.append(__td.graph['hl']), __rid.append(__td.graph['id']), __rod.append(__td.graph['od'])
        __rw += __td.graph['w'];        __rm += __td.graph['m']

    __rd.add_nodes_from(__rdns)
    __rd.add_edges_from(__rdes)
    __rd.graph |= {'s': tuple(sum([0 if __x == None else __x for __x in __rx]) for __rx in zip_longest(*__rs)), 'dt': 'MC', 
                   'c': f"({'|'.join(sorted(__rc))})", 'jl': max(__rjl), 'hl': min(__rhl), 'w': __rw, 'm': __rm, 'id': max(__rid), 'od': max(__rod)}
    return __rd

def FC(_pd, _sd):
    """ 54.037139 -- 3.15 """
    __td, __pn, __pl, __equ_nl = nx.DiGraph(), _pd.number_of_nodes(), len(_pd.graph['s']), list()
    __sd_nd = {__i : __i + __pn for __i in _sd.nodes()}
    __sdns = [(__sd_nd[__i], {'d': __d['d'], 'Equ': __d['Equ']} | {_tx: set(__sd_nd[__xi] for __xi in __d[_tx]) for _tx in ['P', 'S', 'A', 'D']})  for __i, __d in _sd.nodes(data=True)]
    __pdns = [(__i,          {'d': __d['d'], 'Equ': __d['Equ']} | {_tx: set() | __d[_tx] for _tx in ['P', 'S', 'A', 'D']})                         for __i, __d in _pd.nodes(data=True)]
    __sdes = [(__sd_nd[__ex], __sd_nd[__ey]) for __ex, __ey in _sd.edges()]
    __pdes = [(__ex, __ey)                   for __ex, __ey in _pd.edges()]

    # (1) sd 结点id 与 新边 更新；
    __pd_Vsink = set(__i for __i, __d in __pdns if len(__d['S']) == 0)
    __sd_Vsrc  = set(__i for __i, __d in __sdns if len(__d['P']) == 0)
    __tdns = __sdns + __pdns
    # (2) pd结点的D - pd-Vsink的S和D & sd结点的P - sd-Vsrc的A和P更新 & sd结点的 'd' 和 'Equ' 更新；
    __pdn, __sdn = set(__px[0] for __px in __pdns), set(__sd_nd.values())
    for __ni, __nd in __tdns:
        if __ni < __pn: # pd
            __nel = f"p_{__nd['Equ']}"     
            __nd['D'] |= __sdn
            if __ni in __pd_Vsink:
                __nd['S'] |= __sd_Vsrc
        else:           # sd
            __nel = f"s_{__nd['Equ']}"
            __nd['A'] |= __pdn
            if __ni in __sd_Vsrc:
                __nd['P'] |= __pd_Vsink
            __nd['d'] += __pl

        if __nel not in __equ_nl:
            __equ_nl.append(__nel)
        __nd['Equ'] = __equ_nl.index(__nel)

    __td.add_nodes_from(__tdns)
    __td.add_edges_from(__sdes + __pdes + list(product(__pd_Vsink, __sd_Vsrc)))
    __td.graph |= {'s': _pd.graph['s'] + _sd.graph['s'], 'c': f'{_pd.graph['c']}-{_sd.graph['c']}', 
                   'w': max(_pd.graph['w'], _sd.graph['w']), 'hl': _sd.graph['hl'] + __pl, 'dt': 'MS', 
                   'jl': max(_pd.graph['jl'], _sd.graph['jl'], __pl + 1 - _pd.graph['hl']),
                   'm': _pd.graph['m'] + _sd.graph['m'] + len(__pd_Vsink) * len(__sd_Vsrc),
                   'id': max(_pd.graph['id'], _sd.graph['id'], len(__pd_Vsink)), 
                   'od': max(_pd.graph['od'], _sd.graph['od'], len(__sd_Vsrc))}
    return __td




if __name__ == "__main__":
    RootPath = os.path.join(os.getcwd(), 'BASIC_DAG_DATA')
    print(f"Current time:{datetime.now()}--CPU_NUM:{cpu_count()}\n")
    print(f"Root path:{RootPath}")
    FADDR, FTable = './TCDATA.db', 'Total'
    ROOT_DB_ADDR, Table_Name  = './BDAG.db', 'DagData'
    # if os.path.exists(ROOT_DB_ADDR):
    #     os.remove(ROOT_DB_ADDR) 
    target_connection = sqlite3.connect(ROOT_DB_ADDR)
    target_cursor = target_connection.cursor()
    if os.path.exists(FADDR):
        os.remove(FADDR)  
    TCONN = sqlite3.connect(FADDR)    
    TCONN.row_factory = sqlite3.Row
    TCURS = TCONN.cursor()
    TCURS.execute(f"DROP TABLE IF EXISTS {FTable};")
    TCURS.execute(f'''CREATE TABLE IF NOT EXISTS {FTable} (ts BLOB PRIMARY KEY, wd INTEGER NOT NULL,
                        wu INTEGER NOT NULL, md INTEGER NOT NULL, mu INTEGER NOT NULL, jl INTEGER NOT NULL,
                        hl INTEGER NOT NULL, id INTEGER NOT NULL, od INTEGER NOT NULL, L  INTEGER NOT NULL);''')
    # """ 验证算法：
    for __nnum in range(3, 100):
        new_data = defaultdict(int)
        # print("################################")

        # 实验 4：
        _x = 3
        _hlx, _jlx = 1, max(0, __nnum - 1)
        _idx, _odx = __nnum - 1, __nnum - 1
        _mdx, _mux = 0, int(pow(__nnum, 2) / 4)
        _wdx, _wux = _x, _x
        _sdx, _sux = 1, __nnum 
        _ldx, _lux = __nnum - _x + 1, __nnum - _x + 1

        dag_num, shape_num = 0, 0
        # print(f"n:{__nnum}, id:{_idx}, od:{_odx}, jl:{_jlx}, hl:{_hlx}, md:{_mdx}, mu:{_mux}, wd:{_wdx}, wu:{_wux}")
        st = time.time()
        for __s in NSG(__nnum, (_ldx, _lux), (_sdx, _sux), (_mdx, _mux), _idx, _odx):
            shape_num += 1        
            for __gx in NC(__s, _idx, _odx, min(MaxJl(__s), _jlx), _hlx, _mdx, _mux, _wdx, _wux, _ms=True, _mc=True):
                dag_num += 1

        et = time.time()
        print(f"算法生成_{__nnum}-----{shape_num}----{dag_num}--{et - st:.6f}----{(et - st) / max(1, dag_num):.6f}")

    TCURS.close()
    target_cursor.close()
    target_connection.commit()
    target_connection.close()

