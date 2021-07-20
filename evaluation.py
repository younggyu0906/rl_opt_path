import copy
from queue import PriorityQueue

import numpy as np
from haversine import haversine

def get_sum_dist(G, history):
    sum_dist = 0

    for i in range(len(history) - 1):
        dist = G.edges[(history[i], history[i + 1])]['length']
        sum_dist += dist

    return sum_dist

def eval_dist(dist_rl, dist_a_star):
    return dist_rl / dist_a_star

def get_sum_time(G, history):
    sum_time = 0

    for i in range(len(history) - 1):
        dist = G.edges[(history[i], history[i + 1])]['length']
        speed = G.edges[(history[i], history[i + 1])]['speed']
        sum_time += dist/speed

    return sum_time

def eval_time(time_rl, time_a_star):
    return time_rl / time_a_star

def eval_acc(G, history):
    sum_0 = 0
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    sum_4 = 0

    for i in range(len(history) - 1):
        acc = G.edges[(history[i], history[i + 1])]['accident']
        if acc == 0:
            sum_0 += 1
        elif acc == 1:
            sum_1 += 1
        elif acc == 2:
            sum_2 += 1
        elif acc == 3:
            sum_3 += 1
        elif acc == 4:
            sum_4 += 1
    tmp = sum_0 + sum_1 + sum_2 + sum_3 + sum_4
    if not tmp == len(history) - 1:
        print(len(history), tmp, sum_0, sum_1, sum_2, sum_3, sum_4)
        raise ValueError()

    return sum_0, sum_1, sum_2, sum_3, sum_4

# 성능 비교를 위한 A* Search 알고리즘
def a_star(G, start_node, goal_node, min_length, max_length, min_time, max_time, option='accident'):
    if option == 'accident':
        cost_weights = [0.2, 0.4, 0.4]
    elif option == 'distance':
        cost_weights = [0, 1, 0]
    elif option == 'time':
        cost_weights = [0, 1, 0]
    else:
        raise ValueError()
    
    def _get_cost(current_node, next_node):
        
        accident = G.edges[(current_node, next_node)]['accident']
        length = G.edges[(current_node, next_node)]['length']
        speed = G.edges[(current_node, next_node)]['speed']
        time = length / speed

        norm_accident = accident / 4
        norm_accident = norm_accident ** 2
        norm_length = (length - min_length) / (max_length - min_length)
        norm_time = (time - min_time) / (max_time - min_time)

        return norm_accident * cost_weights[0] + norm_length * cost_weights[1] + norm_time * cost_weights[2]
    
    # g = 현재 노드에서 출발 지점까지의 총 cost
    def _get_g(history):
        sum_cost = 0
        for i in range(len(history) - 1):
            cost = _get_cost(history[i], history[i + 1])
            sum_cost += cost
        return sum_cost

    # 현재 노드에서 목적지까지의 추정 cost
    def _get_h(current_node, goal_node):
        goal_dist = haversine(
            G.nodes[current_node]['coordinate'],
            G.nodes[goal_node]['coordinate'],
            unit = 'm')
        return (goal_dist - min_length) / (max_length - min_length)
    
    optimal_history = []
    visited = []

    max_dist = haversine(
        G.nodes[start_node]['coordinate'],
        G.nodes[goal_node]['coordinate'],
        unit = 'm')

    queue = PriorityQueue()

    visited.append(start_node)
    for next_node in G.neighbors(start_node):
        history = copy.deepcopy(visited)
        history.append(next_node)
        g = _get_g(history)
        h = _get_h(next_node, goal_node)
        f = g + h
        
        history = [start_node, next_node]
        queue.put((f, history))

    while True:
        if queue.qsize() == 0:
            break
        item = queue.get()
        f = item[0]
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
            
            next_history = copy.deepcopy(history)
            next_history.append(next_node)
            
            ng = _get_g(next_history)
            nh = _get_h(next_node, goal_node)
            nf = ng + nh
                
            queue.put((nf, next_history))
    return optimal_history

# 최단거리기준, 최소시간기준, 사고위험지역회피 기준의 성능평가
# A* Search 알고리즘과 비교
def evaluation(env, agent, num_iter):
    def get_rl_path(init_state):
        state_shape = env.get_state_shape()
        done = False
        # env 초기화
        state = init_state
        state = np.reshape(state, (1, state_shape[0], state_shape[1]))

        while not done:
            # 현재 상태로 행동을 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, (1, state_shape[0], state_shape[1]))
            state = next_state

            if done:
                return env.history
        # return env.history
    
    sum_ratio_dist = 0
    sum_ratio_time = 0
    sum_0, sum_1, sum_2, sum_3, sum_4 = 0, 0, 0, 0, 0
    fail_num = 0

    for _ in range(num_iter):
        state = env.reset()
        rl_path = get_rl_path(state)
        a_star_path_dist = a_star(
            env.G, env.start_node, env.goal_node,
            env.min_length, env.max_length, env.min_time, env.max_time,
            option='distance')
        a_star_path_time = a_star(
            env.G, env.start_node, env.goal_node,
            env.min_length, env.max_length, env.min_time, env.max_time,
            option='time')
        if not rl_path[-1] ==  env.goal_node:
            fail_num += 1
        else:
            ratio_dist = eval_dist(get_sum_dist(env.G, rl_path), get_sum_dist(env.G, a_star_path_dist))
            sum_ratio_dist += ratio_dist
            
            ratio_time = eval_time(get_sum_time(env.G, rl_path), get_sum_time(env.G, a_star_path_time))
            sum_ratio_time += ratio_time
            
            acc_0, acc_1, acc_2, acc_3, acc_4 = eval_acc(env.G, rl_path)
            sum_0 += acc_0
            sum_1 += acc_1
            sum_2 += acc_2
            sum_3 += acc_3
            sum_4 += acc_4
    score_dist = sum_ratio_dist / (num_iter - fail_num)
    score_time = sum_ratio_time / (num_iter - fail_num)
    score_acc_0 = sum_0 / (num_iter - fail_num)
    score_acc_1 = sum_1 / (num_iter - fail_num)
    score_acc_2 = sum_2 / (num_iter - fail_num)
    score_acc_3 = sum_3 / (num_iter - fail_num)
    score_acc_4 = sum_4 / (num_iter - fail_num)
    # return score_dist, score_time, score_acc_0, score_acc_1, score_acc_2, score_acc_3, score_acc_4
    return 'score_dist: {}\nscore_time: {}\nscore_acc_0: {}\nscore_acc_1: {}\nscore_acc_2: {}\nscore_acc_3: {}\nscore_acc_4: {}'.format(
    score_dist, score_time, score_acc_0, score_acc_1, score_acc_2, score_acc_3, score_acc_4)