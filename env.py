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

from graph_builder import GraphBuilder


class Env:
    def __init__(self, data_dir, is_train=False):
        self._DEBUG = False

        self.is_train = is_train
        if self.is_train:
            seed = time.time()
            self.is_random = True
        else:
            seed = 12345
            self.is_random = False

#         ### TEST ########################
#         self.is_random = True
#         #################################

        self.gb = GraphBuilder(data_dir=data_dir, seed=seed)#, accident_api=accident_api)
        self.G = self.gb.build_graph()
        
        self.G = self.gb.set_accident(self.G, is_random=self.is_random)
        self.G = self.gb.set_speed(self.G, is_random=self.is_random)

        # random.seed(seed)

        self.max_num_neighbors = self._get_max_num_neighbors()
        self.num_features = 5       # accident, time, length, start_to_current, current_to_goal
        self.max_search_depth = 10
        
        self.neighbor_map = None

        self.max_dist = 7435
        self.max_length = self.get_max_length()
        self.min_length = self.get_min_length()
        self.max_time = self.get_max_time()
        self.min_time = self.get_min_time()
        
        self.prev_node = None
        self.current_node = None
        self.start_node = None
        self.goal_node = None
        self.num_step = 0
        self.num_episode = 0
        self.history = []
    
    def get_state_shape(self):
        return (self.max_search_depth, self.max_num_neighbors * self.num_features)

    def get_action_size(self):
        return self.max_num_neighbors

    def get_max_length(self):
        max_length = 0
        for edge in self.G.edges():
            length = self.G.edges[edge]['length']
            if length > max_length:
                max_length = length
        return max_length

    def get_min_length(self):
        min_length = 999999999999
        for edge in self.G.edges():
            length = self.G.edges[edge]['length']
            if length < min_length:
                min_length = length
        return min_length
                
    def get_max_time(self):
        max_time = 0
        for edge in self.G.edges():
            length = self.G.edges[edge]['length']
            speed = self.G.edges[edge]['speed']
            time = length / speed
            if time > max_time:
                max_time = time
        return max_time

    def get_min_time(self):
        min_time = 999999999999
        for edge in self.G.edges():
            length = self.G.edges[edge]['length']
            speed = self.G.edges[edge]['speed']
            time = length / speed
            if time < min_time:
                min_time = time
        return min_time

    def _get_max_num_neighbors(self):
        max_num_neighbors = 0
        for n in self.G.nodes():
            num_neighbors = len([tmp for tmp in self.G.neighbors(n)])
            if num_neighbors > max_num_neighbors:
                max_num_neighbors = num_neighbors
        return max_num_neighbors

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

        # if self.is_train:
        #     # 목표노드
        #     # 에피소드마다 start_node와의 거리를 증가시켜서 선택
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
        # else:
        #     goal_node = random.choice(nodes)
        #     if start_node == goal_node:
        #         while start_node != goal_node:
        #             goal_node = random.choice(nodes)
        
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

    def _get_cost(self, current_node, next_node):
        # if current_node == -1 or next_node == -1:
        #     return 99999999
        accident = self.G.edges[(current_node, next_node)]['accident']
        length = self.G.edges[(current_node, next_node)]['length']
        speed = self.G.edges[(current_node, next_node)]['speed']
        time = length / speed

        norm_accident = accident / 4
        norm_accident = norm_accident ** 2
        norm_length = (length - self.min_length) / (self.max_length - self.min_length)
        norm_time = (time - self.min_time) / (self.max_time - self.min_time)

        return norm_accident * 0.6 + norm_length * 1.2 + norm_time * 1.2

    # g = 현재 노드에서 출발 지점까지의 총 cost
    def _get_g(self, history):
        sum_cost = 0
        for i in range(len(history) - 1):
            cost = self._get_cost(history[i], history[i + 1])
            sum_cost += cost
        return sum_cost

    # 현재 노드에서 목적지까지의 추정 cost
    def _get_h(self, current_node, goal_node):
        # if current_node == -1:
        #     return 99999999
        goal_dist = haversine(
            self.G.nodes[current_node]['coordinate'],
            self.G.nodes[goal_node]['coordinate'],
            unit = 'm')
        return (goal_dist - self.min_length) / (self.max_length - self.min_length)

    # accident, time, length, start_to_current, current_to_goal
    def _get_state(self, current_node):
        def get_feature(current_node, next_node, history):
            if current_node == -1 or next_node == -1:
                return [2 for _ in range(self.num_features)]
            else:
                _norm_acc = self.G.edges[(current_node, next_node)]['accident'] / 4

                length = self.G.edges[(current_node, next_node)]['length']
                speed = self.G.edges[(current_node, next_node)]['speed']
                time = length / speed
                _norm_time = (time - self.min_time) / (self.max_time - self.min_time)

                _norm_length = (length - self.min_length) / (self.max_length - self.min_length)

                g = self._get_g(history)
                _norm_g = g / len(history)*3
                
                h = self._get_h(current_node, self.goal_node)
                _norm_h = h / self.max_dist
                return [_norm_acc, _norm_time, _norm_length, _norm_g, _norm_h]

        if current_node == -1:
            state = np.ones(self.get_state_shape())
            state = state + 1
            return state

        # 우선순위 큐
        queue = PriorityQueue()
        state_map = {}
        visited = copy.deepcopy(self.history)

        neighbor = self._get_neighbors(current_node)
        for next_node in neighbor:
            state_map[next_node] = []
            if next_node == -1:
                continue
            history = copy.deepcopy(visited)
            history.append(next_node)

            g = self._get_g(history)
            h = self._get_h(next_node, self.goal_node)
            f = g + h

            feature = get_feature(current_node, next_node, history[:-1])
            queue.put((f, (next_node, [feature], history)))

        # history = copy.deepcopy(self.history)
        while True:
            if queue.qsize() == 0:
                break
            item = queue.get()
            f = item[0]
            root_node = item[1][0]
            features = copy.deepcopy(item[1][1])
            history = item[1][2]
            current_node = history[-1]

            # 종료조건
            if current_node == self.goal_node:
                state_map[root_node] = features
                if len(state_map[root_node]) < self.max_search_depth:
                    for _ in range(self.max_search_depth - len(state_map[root_node])):
                        state_map[root_node].append([0 for _ in range(self.num_features)])
                break
            elif len(features) > self.max_search_depth:
                break

            if current_node == -1:
                continue

            state_map[root_node] = features

            visited.append(current_node)

            # 다음 방문 가능 노드를 우선순위 큐에 삽입
            next_neighbor = self._get_neighbors(current_node, visited)
            for next_node in next_neighbor:
                if next_node in visited or next_node == -1:
                    continue
                next_history = copy.deepcopy(history)
                next_history.append(next_node)

                ng = self._get_g(next_history)
                nh = self._get_h(next_node, self.goal_node)
                nf = ng + nh

                n_feature = get_feature(current_node, next_node, next_history[:-1])
                n_features = copy.deepcopy(features)
                n_features.append(n_feature)

                queue.put((nf, (root_node, n_features, next_history)))

        for key in state_map:
            if len(state_map[key]) < self.max_search_depth:
                for _ in range(self.max_search_depth - len(state_map[key])):
                    state_map[key].append([2 for _ in range(self.num_features)])

        state = []
        for i in range(self.max_search_depth):
            tmp = []
            for key in neighbor:
                tmp.extend(state_map[key][i])
            state.append(tmp)
            
        state = np.array(state)
        return state

    def _reward(self, action):
        neigebor = self._get_neighbors(self.current_node)
        next_node = neigebor[action]

        accident = self.G.edges[(self.current_node, next_node)]['accident']
        length = self.G.edges[(self.current_node, next_node)]['length']
        speed = self.G.edges[(self.current_node, next_node)]['speed']
        time = length / speed

        norm_accident = accident / 4
        norm_accident = norm_accident# ** 2
        norm_length = (length - self.min_length) / (self.max_length - self.min_length)
        norm_time = (time - self.min_time) / (self.max_time - self.min_time)

        h = self._get_h(next_node, self.goal_node)
        _norm_h = h / self.max_dist

        return -((norm_accident * 0.8 + norm_length * 1.2 + norm_time * 1.2) / 3 + _norm_h)/2

    def reset(self, start_node=None, goal_node=None):
        if self.is_train:
            self.G = self.gb.set_accident(self.G, is_random=self.is_random)
            self.G = self.gb.set_speed(self.G, is_random=self.is_random)
            self.max_time = self.get_max_time()
            self.min_time = self.get_min_time()
        
        if start_node is not None and goal_node is not None:
            self.start_node = start_node
            self.goal_node = goal_node
        else:
            self._set_start_goal_node()

        self.neighbor_map = self._set_neighbor_map()

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
                done = False# if self.is_train else True
                if self._DEBUG:
                    print('[ENV] DUMMY NODE..')
            reward = -4
            # self.prev_node = self.current_node
            # self.current_node = next_node
            
        elif next_node == self.goal_node:
            reward = 2
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
            reward = -4
            self.prev_node = self.current_node
            self.current_node = next_node
            done = True
            if self._DEBUG:
                print('[ENV] DUMMY NODE..!!!')
        
        if self.current_node not in self.history:
            self.history.append(self.current_node)
        state = self._get_state(self.current_node)
        self.num_step += 1
        return state, reward, done
