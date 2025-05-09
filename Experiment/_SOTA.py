#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainHead import *
# from MainFrame import *

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


# # # # # # # # # # # # # # # #
# (1) combination
# A004250
# 1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575, 1958,
# 2436, 3010, 3718, 4565, 5604, 6842, 8349, 10143, 12310, 14883, 17977, 21637, 26015, 31185, 37338, 44583, 53174, 63261,
# 75175, 89134, 105558, 124754, 147273, 173525
# # # # # # # # # # # # # # # #
def combination_exhaustion(node_num, shape_min=0, shape_max=float('Inf')):
    ret_list = []
    for local_n in range(max(shape_min, 1), min(node_num + 1, shape_max)):
        input_node_num = node_num - local_n
        if input_node_num == 0:
            ret_list.append((node_num,))
        else:
            sat_list = combination_exhaustion(input_node_num, local_n, shape_max)
            for sat_list_x in sat_list:
                ret_list.append((local_n,) + sat_list_x)
    return ret_list

# # # # # # # # # # # # # # # #
# (2) permutation
# 输入 combination不同的组合；
# 根据组合合成不同的排序；
# 不同的组合一定无法得到同样的排列；
# # # # # # # # # # # # # # # #
def permutation_exhaustion(combination_list):
    ret_list = []
    for combination_x in combination_list:
        ret_list += list(set(permutations(combination_x, len(combination_x))))
    return ret_list


# # # # # # # # # # # # # # # #
# (3) connection
# # # # # # # # # # # # # # # #
"""
# def combination_exhaustion_free(n, l, sh_min, sh_max):  # 不能于shmin shmax 相等
#     # elif l == n == 0:
#     #     ret_list = [()]
#     assert n > 0
#     assert l > 0
#     ret_list = []
#     if l == 1 and sh_min < n < sh_max:
#         ret_list = [(n,)]
#     else:
#         for source_node_num in range(sh_min + 1, sh_max):   # (sh_min, sh_max)
#             input_node_num = n - source_node_num
#             input_level_num = l - 1
#             if sh_min * input_level_num < input_node_num < sh_max * input_level_num:
#                 sat_list = combination_exhaustion_free(n-source_node_num, input_level_num, source_node_num-1, sh_max)
#                 for sat_list_x in sat_list:
#                     ret_list.append((source_node_num,) + sat_list_x)
#     return ret_list

def combination_exhaustion(n, l, sh_min, sh_max):
    assert l > 2
    assert n > 3
    assert sh_max >= sh_min
    ret_combination_exhaustion_list = []
    if sh_min == sh_max:
        if (l - 2) * sh_min == n - 2:
            ret_combination_exhaustion_list = [(1, 1) + tuple([sh_min for _ in range(l - 2)])]
    else:
        assert l > 3
        if sh_min + 1 == sh_max:    # 没有随机层
            if sh_min * (l -2) <= n - 2 <= sh_max *  (l -2):
                num_sh_max = n - 2 - sh_min * (l -2)
                num_sh_min = l - 2 - num_sh_max
                ret_combination_exhaustion_list = [(1, 1) + tuple([sh_min for _ in range(num_sh_min)]) +
                                                            tuple([sh_max for _ in range(num_sh_max)])]
        elif sh_max > sh_min + 1:
            for num_sh_max in range(1, l - 1):  # [1, l-2]
                for num_sh_min in range(1, l - 1 - num_sh_max):      # [1, l-2-num_sh_max]
                    res_level_num = l - 2 - num_sh_max - num_sh_min  # 剩下的随机层,
                    num_rnode_min = res_level_num * sh_min + 1
                    num_rnode_max = res_level_num * sh_max - 1
                    res_node_num = n - 2 - num_sh_min * sh_min - num_sh_max * sh_max
                    # 可取的选择区间: res_node_num \in (num_rnode_min, num_rnode_max)
                    if num_rnode_min <= res_node_num <= num_rnode_max:
                        temp_shape_num_list = combination_exhaustion_free(res_node_num, res_level_num, sh_min, sh_max)
                        for temp_shape_num_x in temp_shape_num_list:
                            ret_combination_exhaustion_list.append(temp_shape_num_x + (1, 1) +
                                                                   tuple([sh_max for _ in range(num_sh_max)]) +
                                                                   tuple([sh_min for _ in range(num_sh_min)]))
        else:
            os.system('error')
    return ret_combination_exhaustion_list

"""

def Algorithm_input(Algorithm, Algorithm_param_dict):
    dag_list = []
    if Algorithm == 'MINE_NEW':
        dag_list = __gen_mine_new(Algorithm_param_dict)
    else:
        pass
    return dag_list


def exam_pic_show(dag_x, node_num, title):
    dot = gz.Digraph()
    dot.attr(rankdir='LR')
    for node_x in dag_x.nodes(data=True):
        temp_label = 'Node_ID:{0}'.format(str(node_x[0]))
        dot.node('%s' % node_x[0], temp_label, color='black')
    for edge_x in dag_x.edges():
        dot.edge('%s' % edge_x[0], '%s' % edge_x[1])
    # dot.view('./test.png')
    address = f'./generator_test/{node_num}/'
    os.makedirs(address, mode=0o777, exist_ok=True)
    # dot.view(address + f'{title}')
    dot.render(address + f'{title}', view=False)


def gen_mine_new(shape_num_list):
    # for shape_num_list in all_shape_list:
    shape_list = shape_list_trance(shape_num_list)
    dag_list = [nx.DiGraph()]
    for level_id, self_node_list in enumerate(shape_list):
        if (level_id + 1) == len(shape_list):
            for dag_x in dag_list:
                p_nodes = [nodex for nodex in dag_x.nodes() if len(list(dag_x.successors(nodex))) == 0]
                dag_x.add_nodes_from([(self_node_list[0], {'level_num': level_id})])
                for p_node_x in p_nodes:
                    dag_x.add_edge(p_node_x, self_node_list[0])
        else:
            temp_dag_list = []
            for dag_x in dag_list:
                for __rdx in shape_dag_generator(dag_x, self_node_list, level_id):
                    temp_dag_list.append(__rdx)
            dag_list = temp_dag_list
    for ret_dagx in dag_list:
        yield ret_dagx


# #### DAG generator new mine 算法  #### #
def __gen_mine_new(Param_Dict):
    node_num = Param_Dict['Node_Num']
    all_shape_list = shape_enumator(node_num, [1])
    ret_dag_list = []
    for shape_num_list in all_shape_list:
        shape_list = shape_list_trance(shape_num_list)
        dag_list = [nx.DiGraph()]
        for level_id, self_node_list in enumerate(shape_list):
            if (level_id + 1) == len(shape_list):
                for dag_x in dag_list:
                    p_nodes = [nodex for nodex in dag_x.nodes() if len(list(dag_x.successors(nodex))) == 0]
                    dag_x.add_nodes_from([(self_node_list[0], {'level_num': level_id})])

                    for p_node_x in p_nodes:
                        dag_x.add_edge(p_node_x, self_node_list[0])
            else:
                temp_dag_list = []
                for dag_x in dag_list:
                    temp_dag_list += shape_dag_generator(dag_x, self_node_list, level_id)
                dag_list = temp_dag_list
        ret_dag_list += dag_list
    return ret_dag_list

def shape_enumator(node_num, last_shape_num_list):
    reset_node_num = node_num - sum(last_shape_num_list)
    assert reset_node_num > 0
    if reset_node_num == 1:
        temp_new_last_shape_num_list = copy.deepcopy(last_shape_num_list)
        temp_new_last_shape_num_list.append(1)
        return [temp_new_last_shape_num_list]
    else:
        ret_list = []
        for slevel_node_num in range(1, reset_node_num):
            temp_new_last_shape_num_list = copy.deepcopy(last_shape_num_list)
            temp_new_last_shape_num_list.append(slevel_node_num)
            ret_list += shape_enumator(node_num, temp_new_last_shape_num_list)
        return ret_list

def shape_list_trance(shape_num_list):
    node_num = sum(shape_num_list)
    node_id_list = list(range(node_num))
    ret_shape_list = []
    for shape_num_x in shape_num_list:
        ret_shape_list.append(node_id_list[:shape_num_x])
        del node_id_list[:shape_num_x]
    return ret_shape_list

def shape_dag_generator(dag_x, self_node_list, level_num):
    for level_id in range(level_num):
        if level_id == level_num - 1:
            up_same_level_node_iso_label_comput(dag_x, level_id)        # 按层计算 up_iso_label(只算最后一层，其他的之前算过了)
        down_same_level_node_iso_label_comput(dag_x, level_id)          # 按层计算 down_iso_label(所有层都更新一遍)
    # (2) 加新边 pnode_list_enumerate
    # (2.1) 前驱sink点集合  sink_node_list; 2.前驱内点集合  inter_node_list
    total_label_dict = {}
    for node_x in dag_x.nodes(data=True):
        node_x_label = (node_x[1]['level_num'], node_x[1]['up_iso_label'],node_x[1]['down_iso_label'])
        if node_x_label in total_label_dict:
            total_label_dict[node_x_label].append(node_x[0])
        else:
            total_label_dict[node_x_label] = [node_x[0]]

    last_level_node_list = [node_x[0] for node_x in dag_x.nodes(data=True) if node_x[1]['level_num'] == level_num - 1]
    # inte_level_node_list = [(node_x[0]['level_num'], node_x[0]['up_iso_label'],node_x[0]['down_iso_label'])
    #                             for node_x in dag_x.nodes(data=True) if node_x[1]['level_num'] < level_num - 1]

    # (2.3) 穷举所有可行连接前驱 pnode_list_enumerate [(p1,p2,p3...),()], 至少1个sink——node,
    last_level_node_id_enumerate_list = []
    for sn_num in range(len(last_level_node_list)):
        temp_id_enumerate_list_1 = list(combinations(last_level_node_list, sn_num + 1))
        temp_label_enumerate_list = list(set([tuple([(dag_x.nodes[temp_id_x]['level_num'], dag_x.nodes[temp_id_x]['up_iso_label'],dag_x.nodes[temp_id_x]['down_iso_label']) for temp_id_x in temp_id_list])
                                            for temp_id_list in temp_id_enumerate_list_1]))
        temp_id_enumerate_list_2 = []
        for temp_label_list in temp_label_enumerate_list:
            temp_total_label_dict = copy.deepcopy(total_label_dict)
            temp_id_enumerate_list_2.append([temp_total_label_dict[temp_label_x].pop(0) for temp_label_x in temp_label_list])
        last_level_node_id_enumerate_list += temp_id_enumerate_list_2

    # last_level_node_enumerate_list = []
    # for last_level_node_enumerate_list in last_level_node_enumerate_list:
    # inter_level_node_enumerate_list = []
    # for sn_num in range(len(inte_level_node_list) + 1):
    #     inter_level_node_enumerate_list += list(set(itertools.combinations(inte_level_node_list, sn_num)))

    # (2.4) sink的对抗链
    pnode_list_enumerate = []
    for last_level_node_enumerate_x in last_level_node_id_enumerate_list:
        # 1) 样例DAG 删除sink node的所有祖先；
        sample_dag = copy.deepcopy(dag_x)
        rem_set = set(last_level_node_list)
        for last_level_node_x in last_level_node_enumerate_x:
            rem_set.update(nx.ancestors(sample_dag, last_level_node_x))
        sample_dag.remove_nodes_from(rem_set)
        # (2) 获取剩下DAG的对抗链子；
        pred_node_opt_list = list(nx.antichains(sample_dag, topo_order=None))
        for pred_node_opt_x in pred_node_opt_list:
            pred_node_opt_x += last_level_node_enumerate_x
        pnode_list_enumerate += pred_node_opt_list
    # (1) 加新结点
    temp_dag_x = copy.deepcopy(dag_x)
    # 添加向上同构label，添加level_num
    # temp_dag_x.add_nodes_from(self_node_list)
    temp_dag_x.add_nodes_from([(self_node_x, {'level_num':level_num}) for self_node_x in self_node_list])
    # (2.4) 根据本层结点搭配组合 edge_pnode_ret_list      [ [1:[p11,p12,p13],2:[p21,p22,p23],3:[p31,p32,p33]], [1:[p11,p12,p13],2:[p21,p22,p23],3:[p31,p32,p33]] ]
    if len(pnode_list_enumerate) == 0:
        yield temp_dag_x
    else:
        edge_p_list = list(combinations_with_replacement(pnode_list_enumerate, len(self_node_list))) # 从可行解法中抽取 len(sn)个
        for edge_p_list_x in edge_p_list:
            temp_dag_list_x = copy.deepcopy(temp_dag_x)
            for self_node_id, edges_p_x in enumerate(edge_p_list_x):
                for edge_p_x in edges_p_x:
                    temp_dag_list_x.add_edge(edge_p_x, self_node_list[self_node_id])
            yield temp_dag_list_x


def up_same_level_node_iso_label_comput(dag_x, level_id):
    self_level_node_list = [node_x[0] for node_x in dag_x.nodes(data=True) if node_x[1]['level_num'] == level_id]
    up_iso_node_list = [[self_level_node_list.pop()]]
    for node_x in self_level_node_list:
        t_step = True
        sn_subg = dag_x.subgraph(list(nx.ancestors(dag_x, node_x)) + [node_x])
        for node_id_list in up_iso_node_list:
            tsn_subg = dag_x.subgraph(list(nx.ancestors(dag_x, node_id_list[0])) + [node_id_list[0]] )
            if nx.isomorphism.GraphMatcher(sn_subg, tsn_subg).is_isomorphic():
                node_id_list.append(node_x)            # 如果同构，直接加入label对应的list中
                t_step = False
                break
        if t_step:
            up_iso_node_list.append([node_x])
    # print(sink_node_up_iso_list)
    for up_iso_label, node_list in enumerate(up_iso_node_list):
        for node_x in node_list:
            dag_x.nodes[node_x]['up_iso_label'] = up_iso_label



def down_same_level_node_iso_label_comput(dag_x, level_id):
    self_level_node_list = [node_x[0] for node_x in dag_x.nodes(data=True) if node_x[1]['level_num'] == level_id]
    down_iso_node_list = [[self_level_node_list.pop(0)]]
    for node_x in self_level_node_list:
        t_step = True
        sn_subg = dag_x.subgraph(list(dag_x.successors(node_x)) + [node_x])

        for node_id_list in down_iso_node_list:
            tsn_subg = dag_x.subgraph(list(dag_x.successors(node_id_list[0])) + [node_id_list[0]] )
            if nx.isomorphism.GraphMatcher(sn_subg, tsn_subg).is_isomorphic():
                node_id_list.append(node_x)            # 如果同构，直接加入label对应的list中
                t_step = False
                break
        if t_step:
            down_iso_node_list.append([node_x])
    for down_iso_label, node_list in enumerate(down_iso_node_list):
        for node_x in node_list:
            dag_x.nodes[node_x]['down_iso_label'] = down_iso_label


def width_of_dag(_cdag):
    temp_G2 = nx.DiGraph()  # 创建一个二分图
    for edge_x in _cdag.edges():
        temp_G2.add_node('p' + str(edge_x[0]), bipartite=0)
        temp_G2.add_node('d' + str(edge_x[1]), bipartite=1)
        temp_G2.add_edge('p' + str(edge_x[0]), 'd' + str(edge_x[1]))
    u = [n for n in temp_G2.nodes if temp_G2.nodes[n]['bipartite'] == 0]
    matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(temp_G2, top_nodes=u)
    return _cdag.number_of_nodes() - int(len(matching) / 2)
    # matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(temp_G2, top_nodes=['p0', 'p1', 'p4'])
    # matching = nx.bipartite.maximum_matching(temp_G2, top_nodes=u)
    # matching = nx.bipartite.hopcroft_karp_matching(G, {()})
    # maximum_matching()
    # _cdag.graph['w']


if __name__ == "__main__":

    for nnum in range(3, 10):  # node_num = n + 2
        s_num, dag_num = 0, 0
        cc = defaultdict(int)
        st = time.time()
        _x = 3
        _wdx, _wux = _x, _x
        ld, lu = nnum - _x + 1, nnum - _x + 1
        sd, su = 1, min(nnum, _wux)
        for __s in NSG(nnum, (ld, lu), (sd, su), (0, int(pow(nnum, 2) / 4)), nnum - 1, nnum - 1):                                 
            assert sd <= max(__s) <= su, ld <= len(__s) <= lu
            __s = (1,) + __s + (1,)
            s_num += 1

            for dagx in gen_mine_new(__s):

                cdagx = nx.transitive_closure_dag(dagx)  
                w_dag = width_of_dag(cdagx)
                # w_dag = max([len(__x) for __x in nx.antichains(dagx)])
                if _wdx <= w_dag <= _wux:
                    dag_num += 1
                    cc[max(__s)] += 1

        et = time.time()
        print(cc)
        print(f"{nnum}_\t{s_num}_\t{dag_num}_\t{et - st :.8f}")

# G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (4, 5), (4, 6)])
# print( G.graph['w'])
# #########################################################################
# node_num:3_ time1:0.0_ time2:0.0_list1-length = 1; list2-length = 1; list3-length = 1
# #########################################################################
# node_num:4_ time1:0.0_ time2:0.0_list1-length = 2; list2-length = 2; list3-length = 2
# #########################################################################
# node_num:5_ time1:0.0_ time2:0.002043008804321289_list1-length = 3; list2-length = 4; list3-length = 5
# #########################################################################
# node_num:6_ time1:0.0_ time2:0.0065233707427978516_list1-length = 5; list2-length = 8; list3-length = 15
# #########################################################################
# node_num:7_ time1:0.0_ time2:0.019937515258789062_list1-length = 7; list2-length = 16; list3-length = 55
# #########################################################################
# node_num:8_ time1:0.0_ time2:0.08110785484313965_list1-length = 11; list2-length = 32; list3-length = 252
# #########################################################################
# node_num:9_ time1:0.0_ time2:0.44913721084594727_list1-length = 15; list2-length = 64; list3-length = 1464
# #########################################################################
# node_num:10_ time1:0.0020112991333007812_ time2:2.9970197677612305_list1-length = 22; list2-length = 128; list3-length = 10859
# #########################################################################
# node_num:11_ time1:0.018213748931884766_ time2:26.864338874816895_list1-length = 30; list2-length = 256; list3-length = 103141
# #########################################################################
# node_num:12_ time1:0.1936030387878418_ time2:505.52682876586914_list1-length = 42; list2-length = 512; list3-length = 1256764
