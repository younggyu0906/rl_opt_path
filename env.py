import os
import time
import math
import copy
import random
import requests
from queue import PriorityQueue

import folium
import numpy as np
import pandas as pd
import networkx as nx
from haversine import haversine


class Env:
    def __init__(self, data_dir, accident_api=None, is_train=False):
        self.stop_nodes = [
            4130022800, 4130026600, 4130023200, 4130023000, 4130026500, 4130022900, 4130105600, 4130109800
        ]
        self.stop_links = [
            4130191300, 4130191400, 4130148300, 4130148400, 4130126900, 4130126300, 
        ]
        self._DEBUG = False

        self.data_dir = data_dir
        self.accident_api = accident_api

        self.is_train = is_train

        self.G = self._create_graph()
        
        if self.is_train:
            random.seed(time.time())
        else:
            random.seed(12345)
        
        self.max_num_neighbors = self._get_max_num_neighbors()
        self.num_features = 2
        self.max_search_depth = 5
        
        self._set_accident()
        
        self.neighbor_map = None

        self.prev_node = None
        self.current_node = None
        
        self.start_node = None
        self.goal_node = None

        self.max_dist = 0
        
        self.num_step = 0
        self.num_episode = 0
        self.history = []
    
    def get_state_shape(self):
        return (self.max_search_depth, self.max_num_neighbors * self.num_features)

    def get_action_size(self):
        return self.max_num_neighbors

    def _create_graph(self):
        df_sejong_nodes = pd.read_csv(
            os.path.join(self.data_dir, 'AUTO_AREA_NODE.csv'), encoding='cp949')

        sejong_node_ids = df_sejong_nodes['NODE_ID'].to_list()

        np_link_info = np.load(os.path.join(self.data_dir, 'LINK_INFO.npy'))
        df_link = pd.DataFrame(np_link_info)
        df_link.columns = [
            "LINK_ID", "F_NODE", "T_NODE", "LANES",
            "ROAD_RANK", "ROAD_TYPE", "ROAD_NO", "ROAD_NAME",
            "ROAD_USE", "MULTI_LINK", "CONNECT", "MAX_SPD",
            "REST_VEH", "REST_W", "REST_H", "LENGTH",
            "VERTEX_CNT", "REMARK"
        ]

        np_node_info = np.load(os.path.join(self.data_dir, 'NODE_INFO.npy'))
        df_node_info = pd.DataFrame(np_node_info)
        df_node_info.columns = [
            "NODE_ID","NODE_TYPE","NODE_NAME","TURN_P","REMARK"
        ]
        np_node_GIS = np.load(os.path.join(self.data_dir, 'NODE_GIS.npy'))
        df_node_GIS = pd.DataFrame(np_node_GIS)
        df_node_GIS.columns = ["NODE_ID","LONGITUDE","LATITUDE","ELEVATION"]
        df_node = pd.concat((df_node_info,df_node_GIS[["LONGITUDE","LATITUDE","ELEVATION"]]), axis=1)

        G = nx.DiGraph()

        # node
        for (i, row) in df_node.iterrows():
            if int(row['NODE_ID']) in sejong_node_ids:
                if int(row['NODE_ID']) in self.stop_nodes:
                    continue
                G.add_node(int(row['NODE_ID']), coordinate=(row['LONGITUDE'], row['LATITUDE']))

        # link
        for (i, row) in df_link.iterrows():
            if int(row['F_NODE']) in G.nodes() and int(row['T_NODE']) in G.nodes():
                if int(row['LINK_ID']) in self.stop_links:
                    continue
                G.add_edge(
                    int(row['F_NODE']), int(row['T_NODE']),
                    link_id=int(row['LINK_ID']),
                    max_spd=row['MAX_SPD'],
                    length=row['LENGTH']
                )
        
        num_nodes = len(G.nodes())
        while True:
            node_list = list(G.nodes())
            for node in node_list:
                if len([n for n in G.neighbors(node)]) < 2:
                    G.remove_node(node)
            if len(G.nodes()) == num_nodes:
                break
            num_nodes = len(G.nodes())
        
        if self._DEBUG:
            print('nodes: ', len(G.nodes()))
            print('edges: ', len(G.edges()))
        
        return G

    def _get_max_num_neighbors(self):
        max_num_neighbors = 0
        for n in self.G.nodes():
            num_neighbors = len([tmp for tmp in self.G.neighbors(n)])
            if num_neighbors > max_num_neighbors:
                max_num_neighbors = num_neighbors
        return max_num_neighbors
    
    def _set_accident(self):
        if self.accident_api is None or self.is_train:
            weights = (70, 15, 8, 4, 3)

            # set accident
            accident_ranks = [0, 1, 2, 3, 4]
            for edge in self.G.edges():
                self.G.edges[edge]['accident'] = random.choices(accident_ranks, weights=weights)[0]
        else:
#             params = {
#                 'locationName': '세종',
#                 'numOfRows': 100,
#                 'pageNo': 1,
#                 'dataType': 'JSON',
#                 'details': 'TRUE',
#                 'startX': 127.2575497,      # 127.12,
#                 'endX': 127.3446409,        # 127.4,
#                 'startY': 36.4603768,       # 36.4,
#                 'endY': 36.5116487,         # 36.45,
#                 'vertex': 'TRUE'
#             }
#             accident_ranks = [0, 1, 2, 3, 4]
            
#             response = requests.get(self.accident_api, params=params, timeout=10)
#             acc_link_map = response.json()

            import json
            accident_ranks = [0, 1, 2, 3, 4]
            with open(os.path.join(self.data_dir, 'acc_pred_example.json'), 'r') as f:
                acc_link_map = json.load(f)

            for edge in self.G.edges():
                self.G.edges[edge]['accident'] = 0

            edge_ids = [self.G.edges[edge]['link_id'] for edge in self.G.edges()]
            edges = [edge for edge in self.G.edges()]
            for item in acc_link_map['items']:
                if item['linkId'] in edge_ids:
                    for edge in self.G.edges():
                        if self.G.edges[edge]['link_id'] == item['linkId']:
                            self.G.edges[edge]['accident'] = item['rank']
                            break

    def _set_neighbor_map(self):
        neighbor_map = {}
        for node in self.G.nodes():
            neighbor_list = [n for n in self.G.neighbors(node)]
            if self.max_num_neighbors - len(neighbor_list) > 0:
                for i in range(self.max_num_neighbors - len(neighbor_list)):
                    neighbor_list.append(-1)
                    
            random.shuffle(neighbor_list)
            
            neighbor_map[node] = neighbor_list
        return neighbor_map
    
    def _set_start_goal_node(self):
        def get_next_node(node, history):
            next_node = random.choice([n for n in self.G.neighbors(node)])
            tmp = 0
            while True:
                if tmp > 10:
                    next_node = None
                if next_node not in history:
                    break
                else:
                    next_node = random.choice([n for n in self.G.neighbors(node)])
                tmp += 1
            return next_node

        nodes = [node for node in self.G.nodes()]
        start_node = random.choice(nodes)
        history = [start_node]

        if self.is_train:
            # 목표노드
            # 에피소드마다 start_node와의 거리를 증가시켜서 선택
            curr_node = start_node
            if self.num_episode > 100:
                goal_node = random.choice(nodes)
            else:
                next_node = None
                for i in range(self.num_episode // 20 + 1):
                    next_node = get_next_node(curr_node, history)
                    if next_node is None:
                        break
                    history.append(next_node)
                    curr_node = next_node
                if next_node is None:
                    next_node = random.choice(nodes)
                goal_node = next_node

            if start_node == goal_node:
                while start_node != goal_node:
                    goal_node = random.choice(nodes)
        else:
            goal_node = random.choice(nodes)
            if start_node == goal_node:
                while start_node != goal_node:
                    goal_node = random.choice(nodes)
        
        self.start_node = start_node
        self.goal_node = goal_node

    def _get_neighbors(self, current_node, history=None):
        if history is None:
            history = copy.deepcopy(self.history)
        neighbor = copy.deepcopy(self.neighbor_map[current_node])
        for i in range(len(neighbor)):
            if neighbor[i] in history:
                neighbor[i] = -1
        return neighbor


    def _get_state(self, current_node):
        def get_feature(current_node, next_node):
            if current_node == -1 or next_node == -1:
                return [2, 2]
            else:
                # 다음 노드에서 목표 노드까지의 직선 거리
                _dist = haversine(
                    self.G.nodes[next_node]['coordinate'],
                    self.G.nodes[self.goal_node]['coordinate'],
                    unit = 'm')

                _norm_dist = _dist / self.max_dist
                _norm_dist = _norm_dist if _norm_dist < 2 else 2
                
                _norm_acc = self.G.edges[(current_node, next_node)]['accident'] / 4
                return [_norm_dist, _norm_acc]

        if current_node == -1:
            state = np.ones(self.get_state_shape())
            state = state + 1
            return state

        # 우선순위 큐
        search_queue = PriorityQueue()
        state_map = {}
        test = {}

        neighbor = self._get_neighbors(current_node)
        for next_node in neighbor:
            state_map[next_node] = []
            test[next_node] = []
            feature = get_feature(current_node, next_node)
            search_queue.put((feature[0], (next_node, next_node, [feature], [next_node])))

        history = copy.deepcopy(self.history)
        while True:
            if search_queue.qsize() == 0:
                break
            item = search_queue.get()
            value = item[0]
            r_node_id = item[1][0]
            c_node_id = item[1][1]
            features = copy.deepcopy(item[1][2])
            _test = item[1][3]

            # 종료조건
            if c_node_id == self.goal_node:
                state_map[r_node_id] = features
                test[r_node_id] = _test
                if len(state_map[r_node_id]) < self.max_search_depth:
                    for _ in range(self.max_search_depth - len(state_map[r_node_id])):
                        state_map[r_node_id].append([0 for _ in range(self.num_features)])
                        test[r_node_id].append('g')
                break
            elif len(features) > self.max_search_depth:
                break

            if c_node_id == -1:
                continue
            if len(state_map[r_node_id]) == len(features):
                if value < sum([sum(f) for f in state_map[r_node_id]]):
                    continue
            # update state map
            state_map[r_node_id] = features
            test[r_node_id] = _test
            
            # 방문 노드를 history에 append
            history.append(c_node_id)
            # 다음 방문 가능 노드를 우선순위 큐에 삽입
            next_neighbor = self._get_neighbors(c_node_id, history)
            for next_node in next_neighbor:
                n_features = copy.deepcopy(features)
                n_test = copy.deepcopy(_test)
                n_feature = get_feature(c_node_id, next_node)
                n_value = value + n_feature[0]
                n_features.append(n_feature)
                n_test.append(next_node)
                search_queue.put((n_value, (r_node_id, next_node, n_features, n_test)))

        for key in state_map:
            if len(state_map[key]) < self.max_search_depth:
                for _ in range(self.max_search_depth - len(state_map[key])):
                    state_map[key].append([1 for _ in range(self.num_features)])

                for _ in range(self.max_search_depth - len(test[key])):
                    test[key].append('p')

        state = []
        test_state = []
        for i in range(self.max_search_depth):
            tmp = []
            tmp_test = []
            for key in neighbor:
                tmp.extend(state_map[key][i])
                tmp_test.append(test[key][i])
            state.append(tmp)
            test_state.append(tmp_test)
            
        state = np.array(state)
        test_state = np.array(test_state)
        return state

    def _reward(self, action):
        neigebor = self._get_neighbors(self.current_node)
        next_node = neigebor[action]
        a = self.G.edges[(self.current_node, next_node)]['accident'] / 4
        reward = a * a / 2
        return -reward

    def reset(self, start_node=None, goal_node=None):
        if self.is_train:
            self._set_accident()
        
        if start_node is not None and goal_node is not None:
            self.start_node = start_node
            self.goal_node = goal_node
        else:
            is_setting = False
            while True:
                if is_setting:
                    break
                try:
                    self._set_start_goal_node()
                    is_setting = True
                except KeyError:
                    self.key_error.append(str(KeyError))
            

        self.neighbor_map = self._set_neighbor_map()

        # max_dist 초기화
        self.max_dist = haversine(
            self.G.nodes[self.start_node]['coordinate'],
            self.G.nodes[self.goal_node]['coordinate'],
            unit = 'm')
        if self.max_dist == 0:
            self.max_dist = 1

        self.history = [self.start_node]
        self.prev_node = None
        self.current_node = self.start_node
        self.num_step = 0
        state = self._get_state(self.current_node)
        
        self.num_episode += 1
        return state
    
    def step(self, action):
        neigebor = self._get_neighbors(self.current_node)
        next_node = neigebor[action]
        
        if next_node == -1:
            if self.num_step > 100:
                done = True
                if self._DEBUG:
                    print('[ENV] MAX STEP.. {}'.format(self.num_step))
            else:
                done = False if self.is_train else True
                if self._DEBUG:
                    print('[ENV] DUMMY NODE..')
            reward = -2
            # self.prev_node = self.current_node
            # self.current_node = next_node
            
        elif next_node == self.goal_node:
            reward = 1
            self.prev_node = self.current_node
            self.current_node = next_node
            done = True
            if self._DEBUG:
                print('[ENV] GOAL!')
        else:
            if self.num_step > 100:
                done = True
                if self._DEBUG:
                    print('[ENV] MAX STEP.. {}'.format(self.num_step))
            else:
                done = False
            reward = self._reward(action)
            self.prev_node = self.current_node
            self.current_node = next_node

        tmp = False
        for n in neigebor:
            if not n == -1:
                tmp = True
                break
        if not tmp:
            reward = -2
            self.prev_node = self.current_node
            self.current_node = next_node
            done = True
            if self._DEBUG:
                print('[ENV] DUMMY NODE..!!!')
        
        self.history.append(self.current_node)
        state = self._get_state(self.current_node)
        self.num_step += 1
        return state, reward, done
