import myLunarLander

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import gym

from IPython import display
import os
#%%  Network and Reward function
class env_mod():
    def __init__(self, render_mode='human', render_fps=1000):
        self.env = myLunarLander.LunarLanderDiscrete(render_mode=render_mode)
        self.env.metadata['render_fps'] = render_fps

        self.state = None

    def reset(self):
        s, info = self.env.reset()
        self.state = s.copy()
        # self.state = torch.tensor(state,dtype=torch.float32).view(1,-1)
        # state = [leg_l, leg_r, ang_d, angle, vel_x, vel_y, pos_x, pos_y,
        #           x_inpos_flag, fuel_d, game_over_flag, landing_flag]

        self.p_shaping = None
        self.game_over = False
        return s

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def step(self, action):
        s_, r, terminated, tr, info = self.env.step(action)
        # s_ is a state changed by the step and we need previous state for reward func

        if self.state is None:
            self.state = s_.copy()
        # reward 설계 필요
        leg_l = s_[0]
        leg_r = s_[1]
        ang_d = s_[2]
        angle = s_[3]
        vel_x = s_[4]
        vel_y = s_[5]
        pos_x = s_[6]
        pos_y = s_[7]
        x_inpos_flag = s_[8]
        fuel_d = s_[9]
        game_over_flag = s_[10]
        landing_flag = s_[11]

        self.state = self.state.squeeze()

        leg_l_p = self.state[0]
        leg_r_p = self.state[1]
        ang_d_p = self.state[2]
        angle_p = self.state[3]
        vel_x_p = self.state[4]
        vel_y_p = self.state[5]
        pos_x_p = self.state[6]
        pos_y_p = self.state[7]
        x_inpos_flag_p = self.state[8]
        fuel_d_p = self.state[9]
        game_over_flag_p = self.state[10]
        landing_flag_p = self.state[11]
        reward = 0.
        # -----------------------------------------
        shaping = (
            -100 * np.sqrt(pos_x**2 + pos_y**2)  # x,y location
            - 100 * np.sqrt(vel_x**2 + vel_y**2)  # x,y velocity
            - 100 * abs(ang_d) + 10 * leg_l + 10 * leg_r
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward

        shaping_p = (
            -100 * np.sqrt(pos_x_p**2 + pos_y_p**2)  # x,y location
            - 100 * np.sqrt(vel_x_p**2 + vel_y_p**2)  # x,y velocity
            - 100 * abs(ang_d_p) + 10 * leg_l_p + 10 * leg_r_p
        )
        if shaping_p is not None:
            reward = shaping - shaping_p
        shaping_p = shaping
#-----------------------------------------------------------------
        st_ang = (1 * np.pi) / 180
        if (abs(angle) <= st_ang):  # 기준 각도보다 각도가 작으면 보상
            reward += 1
        else:  # 기준 각도보다 크면 페널티
            reward -= 1
#-----------------------------------------------------------------
        reward += (5- abs(vel_y))
        if (vel_y < 0):
            reward += 1
        else:  # vel_y가 작을수록 보상 증가
            reward =  -10    #속도가 양수면 페널티
#-----------------------------------------------------------------
        if x_inpos_flag is True:
            reward += 5
            reward += 5 * (1.5 - abs(pos_x))
        else:
            reward -= 20 * abs(pos_x)
#-----------------------------------------------------------------
        terminated = False
        if abs(game_over_flag) or abs(pos_x) >= 1.0:
            terminated = True
            reward = -5000 # 넌 나가라~
            # -500 / 76.5 -100000
            
        if landing_flag:
            reward = 1000
            terminated = True
            if abs(vel_y_p) > 0.1:
                reward -= 500 * abs(vel_y_p)

            if ( (x_inpos_flag_p) and (x_inpos_flag) ):
                reward += 300 * (5 - abs(vel_x_p))
                reward += 300 * (1.5 - abs(pos_x_p))
                reward += (fuel_d + fuel_d_p) * 300

            else:
                reward = -50

        reward -= 0.8 * fuel_d
        self.state=s_.copy()
        return s_, reward, terminated, tr, info
# -----------------------------------------------------------------------------
myenv=env_mod()
# Add at 11.29.wed
# act_spc = list(np.array([[0.0, 0.0],[0.0, -1.0],[0.0, 1.0],[0.5, 0.0],
#                        [0.5, -1.0],[0.5, 1.0],[1.0, 0.0],[1.0, -1.0],[1.0, 1.0]]).T)
leaky_relu=nn.LeakyReLU(0.2)
#%%
class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = leaky_relu(self.fc1(x))
        x = leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def s_normalizer(state):
    minvalue = np.array([[-1., -1.,  -5., -3.14, -5., -5, -1.5, -1.5, -1, -1.1, -1., -1.]])
    maxvalue = np.array([[1., 1., 10., 6.28, 10., 10., 3., 3., 1., 1.1, 1., 1.]])
    state_n = torch.tensor((state - minvalue) / maxvalue,dtype=torch.float32)
    return state_n
    
def obs_func(state):
    return state

def get_action(state, eps, training=True):
    sn = obs_func(s_normalizer(state))
    if training:
        rnd_val = np.random.random()
        if rnd_val < eps:
            a_idx = random.sample(range(9),1)
        else:
            qvals = qnet(sn).detach().cpu().numpy()
            a_idx = np.argmax(qvals,-1).astype(np.int32)
    else:
        qvals = qnet(sn).detach().cpu().numpy()
        a_idx = np.argmax(qvals,-1).astype(np.int32)
    return a_idx

EPISODE_LENGTH = 500

state = myenv.reset()
State_SZ = state.shape[0] # 12
Action_SZ = 9
#%% Initializing Qnetwork
qnet = QNet(State_SZ,Action_SZ)
qnet_target = QNet(State_SZ, Action_SZ)
qnet_target.load_state_dict(qnet.state_dict())
#%% Load learned Data
#qnet.load_state_dict(torch.load('1201_512_512/weight_6251.pth'))
qnet.load_state_dict(torch.load('1204_2/weight_1700.pth'))
#%%
myenv.env.unwrapped.render_mode = 'human'
myenv.env.metadata['render_fps'] = 100 # pygame 속도 조절
s = myenv.reset()
eps = 0
test = 10
for e in range(test):
    state = myenv.reset()
    state_vec = np.nan * np.zeros((EPISODE_LENGTH,12)).astype(np.float32)
    action_vec = np.nan * np.zeros((EPISODE_LENGTH,1)).astype(np.float32)
    return_val = 0
    
    total_fuel = 0
    
    landing = 0.
    flag = 0.
    for i in range(EPISODE_LENGTH):
        a_idx = get_action(state,eps,training=False)
        a = a_idx[0]
        #a_idx = np.random.randint(9)
        state_, reward, te, tr, info = myenv.step(a) # action의 index를 넣어주었다.
        return_val = return_val + reward
        total_fuel += state_[9]
        #print(state_, reward)
        print("\r{:2.1f}".format(state_[9]),end='')
        landing = state_[11]
        state = state_
        if te or tr:
            break
    total_fuel = total_fuel / 550
    print("\rrsum {:5.2f}|Fuel {:3.2f}| landing {:02}|flag {:3.2f}".format(return_val, total_fuel,landing, total_fuel))
    
print(return_val/test)