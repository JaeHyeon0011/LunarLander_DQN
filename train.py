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
    def __init__(self, render_mode='rgb-array', render_fps=1000):
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
myenv=env_mod()
# Add at 11.29.wed
# act_spc = list(np.array([[0.0, 0.0],[0.0, -1.0],[0.0, 1.0],[0.5, 0.0],
#                        [0.5, -1.0],[0.5, 1.0],[1.0, 0.0],[1.0, -1.0],[1.0, 1.0]]).T)
leaky_relu=nn.LeakyReLU(0.2)
# %%
class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc1=nn.Linear(state_size, 256)
        self.fc2=nn.Linear(256, 128)
        self.fc3=nn.Linear(128, action_size)

    def forward(self, x):
        x=leaky_relu(self.fc1(x))
        x=leaky_relu(self.fc2(x))
        x=self.fc3(x)
        return x

def s_normalizer(state):
    minvalue=np.array(
        [[-1., -1., -5., -3.14, -5., -5, -1.5, -1.5, -1, -1.1, -1., -1.]])
    maxvalue=np.array(
        [[1., 1., 10., 6.28, 10., 10., 3., 3., 1., 1.1, 1., 1.]])
    state_n=torch.tensor((state - minvalue) / maxvalue, dtype=torch.float32)
    return state_n


def obs_func(state):
    return state


def get_action(state, eps, training=True):
    sn=obs_func(s_normalizer(state))
    if training:
        rnd_val=np.random.random()
        if rnd_val < eps:
            a_idx=random.sample(range(9), 1)
        else:
            qvals=qnet(sn).detach().cpu().numpy()
            a_idx=np.argmax(qvals, -1).astype(np.int32)
    else:
        qvals=qnet(sn).detach().cpu().numpy()
        a_idx=np.argmax(qvals, -1).astype(np.int32)
    return a_idx

EPISODES=int(1e4)
EPISODE_LENGTH=500
BATCH=128
discount=0.99
target_update_freq=1000

state=myenv.reset()
State_SZ=state.shape[0]  # 12
Action_SZ=9
# %% Initializing Qnetwork
qnet=QNet(State_SZ, Action_SZ)
qnet_target=QNet(State_SZ, Action_SZ)
qnet_target.load_state_dict(qnet.state_dict())

# Optimizer
opt=optim.Adam(qnet.parameters(), lr=0.001)
# Memory
mem_idx=0
mem_len_cur=0
mem_len=10000
mem={
    'state': torch.zeros((mem_len, State_SZ)),
    'action': torch.zeros((mem_len,)),
    'state_': torch.zeros((mem_len, State_SZ)),
    'reward': torch.zeros((mem_len,)),
    'done': torch.zeros((mem_len,)),
}
# %%
print("warming up")
WARMUP=10
warmupstep=0
while True:
    state=myenv.reset()
    for step in range(EPISODE_LENGTH):
        warmupstep += 1

        a_idx=random.sample(range(9), 1)
        a=a_idx[0]
        state_, reward, done, truncated, info=myenv.step(a)
        state_=torch.tensor(state_, dtype=torch.float32)
        state=torch.tensor(state, dtype=torch.float32)

        mem['state'][mem_idx, :]=state
        mem['action'][mem_idx]=a_idx[0]
        mem['state_'][mem_idx, :]=state_
        mem['reward'][mem_idx]=reward
        mem['done'][mem_idx,]=done
        mem_idx=(mem_idx + 1) % mem_len
        mem_len_cur=min((mem_len_cur+1, mem_len))

        state=state_

        if done:
            break
    if warmupstep > WARMUP:
        print('warmup is over')
        break
#%% Checkpoint
checkpoint_dir = 'checkpoints'
model_dir = 'DQN_model'
checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# %% Training
index = [_ for _ in range(EPISODES)]
avg = [0 for _ in range(EPISODES)]
#%%
loss_mv=0.
target_update_step=0
score=[0. for _ in range(EPISODES)]
for e in range(EPISODES):
    state=myenv.reset()
    eps=max(1.0 / (1. + 0.05 * e), 0.05)
    q_update_num=0
    return_val=0.
    average = 0
    state_vec=np.nan*np.zeros((EPISODE_LENGTH, 12)).astype(np.float32)
    action_vec=np.nan*np.zeros((EPISODE_LENGTH, 1)).astype(np.float32)

    for i in range(EPISODE_LENGTH):
        a_idx=get_action(state, eps, training=True)
        a=a_idx[0]
        state_, reward, done, truncated, info=myenv.step(a)

        state_vec[i, :]=state
        action_vec[i, :]=a_idx[0]

        state=torch.tensor(state, dtype=torch.float32)
        state_=torch.tensor(state_, dtype=torch.float32)

        mem['state'][mem_idx, :]=state
        mem['action'][mem_idx]=a_idx[0]
        mem['state_'][mem_idx, :]=state_
        mem['reward'][mem_idx]=reward
        mem['done'][mem_idx,]=done

        mem_idx=(mem_idx + 1) % mem_len
        mem_len_cur=min((mem_len_cur + 1, mem_len))

        mem_batch_idx=torch.tensor(random.sample(
            range(mem_len_cur), k=min(BATCH, mem_len_cur)))

        sBtc=mem['state'][mem_batch_idx, :]
        aBtc=mem['action'][mem_batch_idx].long()
        s_Btc=mem['state_'][mem_batch_idx, :]
        rBtc=mem['reward'][mem_batch_idx]
        dBtc=mem['done'][mem_batch_idx]

        sn=obs_func(s_normalizer(sBtc))
        s_n=obs_func(s_normalizer(s_Btc))
        qvals=qnet(s_n).detach().numpy()
        idx_max=torch.argmax(torch.tensor(qvals), dim=-1)
        
        y=rBtc + (1 - dBtc) * discount * \
            qnet_target(s_n).detach().numpy()[
            torch.arange(len(idx_max)), idx_max]
        opt.zero_grad()
        q=torch.sum(
            qnet(sn) * nn.functional.one_hot(aBtc, 9).float(), dim=-1)
        loss=nn.MSELoss()(y.clone().detach(), q)
        loss.backward()
        opt.step()

        target_update_step=target_update_step + 1
        if target_update_step >= target_update_freq:
            qnet_target.load_state_dict(qnet.state_dict())
            target_update_step=0
            q_update_num=q_update_num + 1
        return_val=return_val + reward

        loss_mv = loss_mv * 0.98 + loss * 0.02
        print('\rep{:05}|eps{:4.2f}|loss{:12.7f}|qu{:03}|rsum{:12.3f}'.format(
            e, eps, loss_mv, q_update_num, return_val), end='')
        state=state_

        if done:
            break
    print()
    score[e]=return_val
    
    '''if not os.path.exists('1203_3'):
        os.makedirs('1203_3')'''
        
    start_i = e - 100
    end_i = e
    
    if (e+1) % 100 == 0:
        torch.save(qnet.state_dict(), f"1204_2/weight_{e+1}.pth")
        for i in range(start_i, end_i):
            average += score[i]
        average = average / 100
        avg[e] = average
        plt.figure(dpi=400)
        plt.scatter(index, score, s=0.5)
        plt.scatter(index, avg, s=3)
        plt.show()
#%%
store_reward = [0 for _ in range(EPISODES)]
store_avg = [0 for _ in range(EPISODES)]
for i in range(691):
    store_reward[i] = score[i]
    store_avg[i] = avg[i]
print(store_avg)
#%%
#index=[_ for _ in range(EPISODES)]
plt.figure(dpi=500)
plt.scatter(index, score,s=0.3)
plt.scatter(index, avg)
plt.xlim(-10, 3000)
plt.ylim(-3000, 10000)
plt.show()
# %% Load learned Data
qnet.load_state_dict(torch.load('1201/weight_2251.pth'))
'''if e % 100 == 0:
    checkpoint_file = os.path.joint(checkpoint_dir,f'model_episode_{e+1}.pth')
    checkpoint = {'episode_num': e, 
                  'model_state_dict()' : qnet.state.dict(),
                  'optimizer_state_dict': opt.state_dict(),}
    torch.save(checkpoint, checkpoint_file)'''
#%% Load Checkpoint
def load(model, optimizer, checkpoint_file):
    print('[*] Reading Checkpoints...')
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        e = checkpoint['episode_num']
        print(' [*] Success to read checkpoint at episode {}'.format(e))
        return True, e
    else:
        print('failed to find a checkpoint')
