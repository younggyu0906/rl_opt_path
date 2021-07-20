import os
import sys
import copy
import pylab
import folium
import numpy as np
import pandas as pd
import tensorflow as tf
from queue import PriorityQueue
from haversine import haversine

from env import Env
from agent import DQNAgent


class OptimalPathFinder:
    def __init__(self):
        self.env = None
        self.agent = None

        self.state_shape = None
        self.action_size = None

        # if not os.path.exists('./results'):
        #     os.makedirs('./results')
        # if not os.path.exists('./results/models'):
        #     os.makedirs('./results/models')
        # if not os.path.exists('./results/graphs'):
        #     os.makedirs('./results/graphs')
        # if not os.path.exists('./results/maps'):
        #     os.makedirs('./results/maps')

    # environment 생성 (train, get_optimal_path 함수 호출 전에 사용)
    ## data_dir: 노드&링크 데이터 디렉토리
    ## accident_api: 사고 위험지역 예측 API URL (http://10.100.20.61:{port}/link)
    ## is_train: 학습할 경우 True, 최적 경로를 예측할 경우 False로 설정
    def load_env(self, data_dir, is_train=False):
        self.env = Env(data_dir=data_dir, is_train=is_train)

        self.state_shape = self.env.get_state_shape()
        self.action_size = self.env.get_action_size()

    # 새로운 모델(Agent)를 생성 (train 함수 호출 전에 사용)
    def build_model(self, model_name, **dqn_hyperparameters):
        if self.env is None:
            raise ValueError('\'env\' is None. Please call \'load_env()\' first.')
        self.agent = DQNAgent(
            model_name=model_name,
            state_shape=self.state_shape,
            action_size=self.action_size,
            load_model=False,
            discount_factor=dqn_hyperparameters['discount_factor'] if 'discount_factor' in dqn_hyperparameters else 0.99,
            learning_rate=dqn_hyperparameters['learning_rate'] if 'learning_rate' in dqn_hyperparameters else 0.001,
            epsilon=dqn_hyperparameters['epsilon'] if 'epsilon' in dqn_hyperparameters else 1.0,
            epsilon_decay=dqn_hyperparameters['epsilon_decay'] if 'epsilon_decay' in dqn_hyperparameters else 0.999,
            epsilon_min=dqn_hyperparameters['epsilon_min'] if 'epsilon_min' in dqn_hyperparameters else 0.001,
            batch_size=dqn_hyperparameters['batch_size'] if 'batch_size' in dqn_hyperparameters else 256,
            train_start=dqn_hyperparameters['train_start'] if 'train_start' in dqn_hyperparameters else 1000,
            memory_size=dqn_hyperparameters['memory_size'] if 'memory_size' in dqn_hyperparameters else 5000)

        # 모델 구조 저장
        model = tf.keras.utils.plot_model(
            self.agent.model, './data/results/plot_models/{}_model.png'.format(model_name), show_shapes=True)
        target_model = tf.keras.utils.plot_model(
            self.agent.target_model, './data/results/plot_models/{}_target_model.png'.format(model_name), show_shapes=True)
        return model, target_model

    # 학습된 모델(Agent) 를 load
    def load_model(self, model_name):
        if self.env is None:
            raise ValueError('\'env\' is None. Please call \'load_env()\' first.')
        self.agent = DQNAgent(
            model_name=model_name,
            state_shape=self.state_shape,
            action_size=self.action_size,
            epsilon=0,
            load_model=True)

    # DQN Agent 학습
    ## model_name: 모델 이름
    ## num_episode: 학습할 에피소드 수
    ## dqn_hyperparameters: DQN 하이퍼파라미터
    def train(self, model_name, num_episode=5000):
        if self.agent is None:
            raise ValueError('\'agent\' is None. Please call \'build_model()\' first.')
        scores, episodes, is_goals = [], [], []

        for e in range(num_episode):
            done = False
            score = 0
            step = 0

            # env 초기화
            state = self.env.reset()
            state = np.reshape(state, (1, self.state_shape[0], self.state_shape[1]))

            while not done:
                step += 1

                # 현재 상태로 행동을 선택
                action = self.agent.get_action(state)

                # 선택한 행동으로 환경에서 한 타임스텝 진행
                next_state, reward, done = self.env.step(action)
                next_state = np.reshape(next_state, (1, self.state_shape[0], self.state_shape[1]))

                # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
                self.agent.append_sample(state, action, reward, next_state, done)
                # 매 타임스텝마다 학습
                if len(self.agent.memory) >= self.agent.train_start:
                    self.agent.train_model()

                score += reward
                state = next_state

                if done:
                    is_goal = True if reward == 1 else False
                    is_goals.append(is_goal)

                    # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                    self.agent.update_target_model()

                    # 에피소드마다 학습 결과 출력
                    scores.append(score)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./data/results/graphs/score_{}.png".format(model_name))

                    print('episode: {0:4d}\t\tscore: {1:.3f}\t\tmemory length: {2:4d}\t\tepsilon: {3:.3f}'.format(
                        e, score, len(self.agent.memory), self.agent.epsilon
                    ))

                    # 종료조건
                    exit_num = 10
                    is_exit = True
                    if len(is_goals) < exit_num:
                        is_exit = False
                    else:
                        for is_goal in is_goals[-exit_num:]:
                            if not is_goal:
                                is_exit = False
                                break
                    if is_exit and e > 200 and np.mean(scores[-min(exit_num, len(scores)):]) > 0.5:
                        self.agent.model.save_weights("./data/results/models/{}.h5".format(model_name))
                        sys.exit()
                    if e % 50 == 0:
                        self.agent.model.save_weights("./data/results/models/{}_{}.h5".format(model_name, e))
        self.env = None
        self.agent = None

        self.state_shape = None
        self.action_size = None

    # 최적 경로 탐색
    ## shuttle: 셔틀의 현재 위치(노드id)
    ## passenger: 승객의 현재 위치(노드id)
    ## goal: 목표 위치(노드id)
    def get_optimal_path(self, shuttle, passenger, goal):
        if self.agent is None:
            raise ValueError('\'agent\' is None. Please call \'load_model()\' first.')

        def get_cost(current_node, next_node, goal_node, max_dist):
            if max_dist == 0:
                max_dist = 1
            _dist = haversine(
                self.env.G.nodes[next_node]['coordinate'],
                self.env.G.nodes[goal_node]['coordinate'],
                unit = 'm')

            _norm_dist = _dist / max_dist
            _norm_dist = _norm_dist if _norm_dist < 2 else 2
            
            _norm_acc = self.env.G.edges[(current_node, next_node)]['accident'] / 4
            return _norm_dist + _norm_acc

        def get_path(start_node, goal_node):
            state = self.env.reset(start_node=start_node, goal_node=goal_node)
            state = np.reshape(state, (1, self.state_shape[0], self.state_shape[1]))

            done = False
            while not done:
                # 현재 상태로 행동을 선택
                action = self.agent.get_action(state)

                # 선택한 행동으로 환경에서 한 타임스텝 진행
                next_state, reward, done = self.env.step(action)
                next_state = np.reshape(next_state, (1, self.state_shape[0], self.state_shape[1]))

                state = next_state
    
                if done:
                    break
            if False:#reward == 1:
                return self.env.history
            else:
                def _get_cost(current_node, next_node):
                    # if current_node == -1 or next_node == -1:
                    #     return 99999999
                    accident = self.env.G.edges[(current_node, next_node)]['accident']
                    length = self.env.G.edges[(current_node, next_node)]['length']
                    speed = self.env.G.edges[(current_node, next_node)]['speed']
                    time = length / speed

                    norm_accident = accident / 4
                    norm_accident = norm_accident ** 2
                    norm_length = (length - self.env.min_length) / (self.env.max_length - self.env.min_length)
                    norm_time = (time - self.env.min_time) / (self.env.max_time - self.env.min_time)

                    return norm_accident * 0.6 + norm_length * 1.2 + norm_time * 1.2

                # g = 현재 노드에서 출발 지점까지의 총 cost
                def _get_g(history):
                    sum_cost = 0
                    for i in range(len(history) - 1):
                        cost = _get_cost(history[i], history[i + 1])
                        sum_cost += cost
                    return sum_cost

                # 현재 노드에서 목적지까지의 추정 cost
                def _get_h(current_node, goal_node):
                    # if current_node == -1:
                    #     return 99999999
                    goal_dist = haversine(
                        self.env.G.nodes[current_node]['coordinate'],
                        self.env.G.nodes[goal_node]['coordinate'],
                        unit = 'm')
                    return (goal_dist - self.env.min_length) / (self.env.max_length - self.env.min_length)

                optimal_history = []
                visited = []
                G = self.env.G

                max_dist = haversine(
                    G.nodes[start_node]['coordinate'],
                    G.nodes[goal_node]['coordinate'],
                    unit = 'm')

                queue = PriorityQueue()

                visited.append(start_node)
                for next_node in G.neighbors(start_node):
                    # h = get_cost(current_node=start_node, next_node=next_node, goal_node=goal_node, max_dist=max_dist)
                    history = [start_node, next_node]

                    g = _get_g(history)
                    h = _get_h(next_node, self.env.goal_node)
                    f = g + h
                    queue.put((h, history))

                while True:
                    if queue.qsize() == 0:
                        break
                    item = queue.get()
                    h = item[0]
                    history = item[1]

                    current_node = history[-1]
                    visited.append(current_node)

                    # 종료조건
                    if current_node == goal_node:
                        optimal_history = history
                        break

                    for next_node in G.neighbors(current_node):
                        if next_node in visited:
                            continue
                        # next_h = get_cost(current_node=current_node, next_node=next_node, goal_node=goal_node, max_dist=max_dist)
                        next_history = copy.deepcopy(history)
                        next_history.append(next_node)
                        
                        ng = _get_g(next_history)
                        nh = _get_h(next_node, self.env.goal_node)
                        nf = ng + nh
                        
                        queue.put((nf, next_history))
                return optimal_history
        history_1 = get_path(shuttle, passenger)
        history_2 = get_path(passenger, goal)
        return history_1, history_2

    def render(self, model_name, data_dir, shuttle, passenger, goal, history_1, history_2, map_type="2D"):
        G = self.env.G
        sejong_coord = [36.4831117, 127.2920639]

        # Map 종류 Base, gray, midnight, Hybrid, Satellite
        map_real_image = 'http://api.vworld.kr/req/wmts/1.0.0/D62FBAB1-07EC-3CC5-8F48-033735C2BF9A/Satellite/{z}/{y}/{x}.jpeg'
        map_2D_image = 'http://api.vworld.kr/req/wmts/1.0.0/D62FBAB1-07EC-3CC5-8F48-033735C2BF9A/midnight/{z}/{y}/{x}.png'

        if map_type == "REAL" :
            map_title = map_real_image
        else :
            map_title = map_2D_image

        li_color = ["lightgreen", "green", "yellow", "orange", "red"]
        shuttle_color = 'purple'
        passenger_color = 'cyan'
        goal_color = 'blue'

        # data load
        np_link_vertex = np.load(os.path.join(data_dir, 'LINK_VERTEX.npy'))[:,:4]
        
        df_link = pd.DataFrame(np.load(os.path.join(data_dir, 'link.npy')))
        df_link.columns = np.load(os.path.join(data_dir, 'LINK_COLUMNS.npy'))

        df_node = pd.DataFrame(np.load(os.path.join(data_dir, 'node.npy')))
        df_node.columns = np.load(os.path.join(data_dir, 'NODE_COLUMNS.npy'))

        # 맵 생성
        m = folium.Map(location=sejong_coord,
            zoom_start=14,
            tiles=map_title,#'tiles/{z}/{x}/{y}.png',
            attr='sejong')

        np_link_data = np.array(df_link)
        np_node_data = np.array(df_node)

        nodes = [node for node in G.nodes()]
        edges = [G.edges[edge]['link_id'] for edge in G.edges()]

        # 자율주행 구역 링크
        for edge in G.edges():
            # 링크 좌표 탐색
            # 링크아이디와 매칭되는 데이터구간을 좌표 데이터셋에서 탐색하여 링크 좌표 불러옴)

            link_id = G.edges[edge]['link_id']

            np_idx = np.where(np_link_vertex[:,0]==link_id)[0]

            min_idx = np_idx.min()
            max_idx = np_idx.max()
            
            np_target_link_coords_set = np_link_vertex[min_idx:max_idx+1,2:4]
            
            np_coords_set = np.zeros((len(np_target_link_coords_set),2))

            np_coords_set[:,0] = np_target_link_coords_set[:,1]
            np_coords_set[:,1] = np_target_link_coords_set[:,0]
            
            folium.PolyLine(
                locations=np_coords_set,
                color=li_color[G.edges[edge]['accident']],
                popup = '{}/{}/{}'.format(int(G.edges[edge]['length']), G.edges[edge]['speed'], G.edges[edge]['accident']),
                line_cap='round'
            ).add_to(m)

        # 경로1
        for i in range(len(history_1) - 1):
            if history_1[i] == -1 or history_1[i + 1] == -1:
                break
            try:
                link_id = G.edges[(history_1[i], history_1[i + 1])]['link_id']
            except KeyError:
                continue
            np_idx = np.where(np_link_vertex[:,0]==link_id)[0]

            min_idx = np_idx.min()
            max_idx = np_idx.max()
            
            np_target_link_coords_set = np_link_vertex[min_idx:max_idx+1,2:4]
            
            np_coords_set = np.zeros((len(np_target_link_coords_set),2))

            np_coords_set[:,0] = np_target_link_coords_set[:,1]
            np_coords_set[:,1] = np_target_link_coords_set[:,0]
            
            edge = (history_1[i], history_1[i + 1])
            folium.PolyLine(
                locations=np_coords_set,
                color=shuttle_color,
                popup = '{}/{}/{}'.format(int(G.edges[edge]['length']), G.edges[edge]['speed'], G.edges[edge]['accident']),
                line_cap='round',
                dash_array='5'
            ).add_to(m)

        # 경로2
        for i in range(len(history_2) - 1):
            if history_2[i] == -1 or history_2[i + 1] == -1:
                break
            try:
                link_id = G.edges[(history_2[i], history_2[i + 1])]['link_id']
            except KeyError:
                continue
            np_idx = np.where(np_link_vertex[:,0]==link_id)[0]

            min_idx = np_idx.min()
            max_idx = np_idx.max()
            
            np_target_link_coords_set = np_link_vertex[min_idx:max_idx+1,2:4]
            
            np_coords_set = np.zeros((len(np_target_link_coords_set),2))

            np_coords_set[:,0] = np_target_link_coords_set[:,1]
            np_coords_set[:,1] = np_target_link_coords_set[:,0]
            
            edge = (history_2[i], history_2[i + 1])
            folium.PolyLine(
                locations=np_coords_set,
                color=goal_color,
                popup = '{}/{}/{}'.format(int(G.edges[edge]['length']), G.edges[edge]['speed'], G.edges[edge]['accident']),
                line_cap='round',
                dash_array='5'
            ).add_to(m)

        # 노드
        for node in nodes:
            coords = [G.nodes[node]['coordinate'][1], G.nodes[node]['coordinate'][0]]
            folium.CircleMarker(
                    location = coords,
                        radius=3,
                        color = 'lightgreen',
                        popup = node,
                        fill = True,
                        fill_color = 'lightgreen'
            ).add_to(m)

        # 경로 노드 1
        for (i, node) in enumerate(history_1):
            if node == -1:
                break
            coords = (G.nodes[node]['coordinate'][1], G.nodes[node]['coordinate'][0])
            folium.CircleMarker(
                location = coords,
                radius=5,
                color = shuttle_color,
                popup = node,
                # popup = 'Shuttle_to_Passenger_{}  <node_id: {}>'.format(i, node),
                # popup = '{}/{}/{}'.format(int(G.edges[edge]['length']), G.edges[edge]['speed'], G.edges[edge]['accident']),
                fill = True,
                fill_color = shuttle_color
            ).add_to(m)

        # 경로 노드 2
        for (i, node) in enumerate(history_2):
            if node == -1:
                break
            coords = (G.nodes[node]['coordinate'][1], G.nodes[node]['coordinate'][0])
            folium.CircleMarker(
                location = coords,
                radius=5,
                color = goal_color,
                popup = node,
                # popup = 'Passenger_to_Goal_{}  <node_id: {}>'.format(i, node),
                # popup = '{}/{}/{}'.format(int(G.edges[edge]['length']), G.edges[edge]['speed'], G.edges[edge]['accident']),
                fill = True,
                fill_color = goal_color
            ).add_to(m)

        # 셔틀 위치
        coords = (G.nodes[shuttle]['coordinate'][1], G.nodes[shuttle]['coordinate'][0])
        folium.CircleMarker(
            location = coords,
            radius=7,
            color = shuttle_color,
            popup = 'Shuttle  <node_id: {}>'.format(shuttle),
            fill = True,
            fill_color = shuttle_color
        ).add_to(m)

        # 승객 위치
        coords = (G.nodes[passenger]['coordinate'][1], G.nodes[passenger]['coordinate'][0])
        folium.CircleMarker(
            location = coords,
            radius=7,
            color = passenger_color,
            popup = 'Passenger  <node_id: {}>'.format(passenger),
            fill = True,
            fill_color = passenger_color
        ).add_to(m)

        # 목적지
        coords = (G.nodes[goal]['coordinate'][1], G.nodes[goal]['coordinate'][0])
        folium.CircleMarker(
            location = coords,
            radius=7,
            color = goal_color,
            popup = 'Goal  <node_id: {}>'.format(goal),
            fill = True,
            fill_color = goal_color
        ).add_to(m)
            
        m.save(os.path.join('./data/results/maps', '{}.html'.format(model_name)))#, map_type)))

        return m

    def test_render(self, data_dir):
        G = self.env.G
        sejong_coord = [36.4831117, 127.2920639]

        # Map 종류 Base, gray, midnight, Hybrid, Satellite
        map_real_image = 'http://api.vworld.kr/req/wmts/1.0.0/D62FBAB1-07EC-3CC5-8F48-033735C2BF9A/Satellite/{z}/{y}/{x}.jpeg'
        map_2D_image = 'http://api.vworld.kr/req/wmts/1.0.0/D62FBAB1-07EC-3CC5-8F48-033735C2BF9A/midnight/{z}/{y}/{x}.png'

        if map_type == "REAL" :
            map_title = map_real_image
        else :
            map_title = map_2D_image

        li_color = ["lightgreen", "green", "yellow", "orange", "red"]
        shuttle_color = 'purple'
        passenger_color = 'cyan'
        goal_color = 'blue'

        # data load
        np_link_vertex = np.load(os.path.join(data_dir, 'LINK_VERTEX.npy'))[:,:4]
        
        df_link = pd.DataFrame(np.load(os.path.join(data_dir, 'link.npy')))
        df_link.columns = np.load(os.path.join(data_dir, 'LINK_COLUMNS.npy'))

        df_node = pd.DataFrame(np.load(os.path.join(data_dir, 'node.npy')))
        df_node.columns = np.load(os.path.join(data_dir, 'NODE_COLUMNS.npy'))

        # 맵 생성
        m = folium.Map(location=sejong_coord,
            zoom_start=14,
            tiles=map_title,#'tiles/{z}/{x}/{y}.png',
            attr='sejong_test')

        np_link_data = np.array(df_link)
        np_node_data = np.array(df_node)

        nodes = [node for node in G.nodes()]
        edges = [G.edges[edge]['link_id'] for edge in G.edges()]

        # 자율주행 구역 링크
        for edge in G.edges():
            # 링크 좌표 탐색
            # 링크아이디와 매칭되는 데이터구간을 좌표 데이터셋에서 탐색하여 링크 좌표 불러옴)

            link_id = G.edges[edge]['link_id']

            np_idx = np.where(np_link_vertex[:,0]==link_id)[0]

            min_idx = np_idx.min()
            max_idx = np_idx.max()
            
            np_target_link_coords_set = np_link_vertex[min_idx:max_idx+1,2:4]
            
            np_coords_set = np.zeros((len(np_target_link_coords_set),2))

            np_coords_set[:,0] = np_target_link_coords_set[:,1]
            np_coords_set[:,1] = np_target_link_coords_set[:,0]
            
            folium.PolyLine(
                locations=np_coords_set,
                color=li_color[G.edges[edge]['accident']],
                popup = '{}/{}/{}'.format(int(G.edges[edge]['length']), G.edges[edge]['speed'], G.edges[edge]['accident']),
                line_cap='round'
            ).add_to(m)

        # 노드
        for node in nodes:
            coords = [G.nodes[node]['coordinate'][1], G.nodes[node]['coordinate'][0]]
            folium.CircleMarker(
                    location = coords,
                        radius=3,
                        color = 'lightgreen',
                        popup = node,
                        fill = True,
                        fill_color = 'lightgreen'
            ).add_to(m)

        return m