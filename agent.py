from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from collections import deque
import numpy as np
import random


class DQNAgent:
    def __init__(
            self,
            model_name,
            state_shape,
            action_size,
            load_model=False,
            discount_factor=0.99,
            learning_rate=0.001,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.001,
            batch_size=256,
            train_start=1000,
            memory_size=5000):

        self.load_model = load_model

        # 상태와 행동의 크기 정의
        self.state_shape = state_shape
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.batch_size = batch_size
        self.train_start = train_start

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=memory_size)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            import os
            model_path = "./data/results/models/{}.h5".format(model_name)
            if os.path.exists(model_path):
                self.model.load_weights(model_path)
            else:
                print('\n  model file does not exist!!!\n')
                exit()

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Conv1D(256, 2, strides=2, activation='relu',
                         input_shape=self.state_shape))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_shape[0], self.state_shape[1]))
        next_states = np.zeros((self.batch_size, self.state_shape[0], self.state_shape[1]))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)