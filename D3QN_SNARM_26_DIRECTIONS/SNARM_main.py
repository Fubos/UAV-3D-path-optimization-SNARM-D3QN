# -*- coding: utf-8 -*-
"""
Created on Nov. 7
Modified on Nov. 25, 2019
@author: Yong Zeng

For cellular-connected UAV, implement simultaneous navigation and radio mapping (SNARM) via
deep reinforcement learning (DRL)
"""

import numpy as np
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from keras.layers import Input, Dense,Lambda
from keras.models import Model
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
import keras.backend as K
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

from mpl_toolkits.mplot3d import Axes3D

import radio_environment as rad_env #the actual radio environment
from radio_mapping import RadioMap  #the radio map class


# 使用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 备注：使用 GPU 0

# 动态分配显存
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

X_MAX=1000.0 
Y_MAX=1000.0 #The area region in meters
Z_MAX=100.0
Z_MIN=60

Z_height = [60,70,80,90,100]

MAX_VALS=np.array([[X_MAX,Y_MAX,Z_MAX]])

DESTINATION=np.array([[800,800,100]],dtype="float32") # 原[1400,1600]UAV flying destination in meter
DIST_TOLERANCE=14.1421 # 10根号2 原30 considered as reach destination if UAV reaches the vicinity within DIST_TOLERANCE meters

DISCOUNT = 0.998 # 原1 
REPLAY_MEMORY_SIZE = 200_000  # 原100000How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 10_000  # 原5000Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 100  # 原5 Terminal states (end of episodes)
MAX_STEP = 600 # 原200 maximum number of time steps per episode
MODEL_NAME = '512_256_128_128'
MIN_REWARD = -2000  # 原-1000 For model save
nSTEP=6 # 原30 parameter for multi-step learning

# Environment settings
EPISODES = 10000#Number of training episodes


# Exploration settings
epsilon =0.60  # 原0.5 not a constant, going to be decayed
EPSILON_DECAY = 0.9995 # 原0.998
MIN_EPSILON = 0

episode_all=np.arange(EPISODES)
epsilon_all=epsilon*EPSILON_DECAY**episode_all
epsilon_all=np.maximum(epsilon_all,MIN_EPSILON)

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # 原50 episodes 每50回合统计一次
SHOW_PREVIEW = False


delta_t=0.5 #time step length in seconds

#penalty measured in terms of the time required to reach destination
REACH_DES_REWARD=200
GET_CLOSE_DES_REWARD=2

MOVE_PENALTY = 1
NON_COVER_PENALTY = 50 # 原40
OUT_BOUND_PENALTY = 100 #原10000


x=np.linspace(0,X_MAX,200)
y=np.linspace(0,Y_MAX,200)


OBSERVATION_SPACE_VALUES=(3,) # 原2-dimensional UAV flight, x-y coordinate of UAV
# 增加二维方向数，从而优化路径,原方向数4
# 顺时针方向,改进为8方向
num_sqrt_2 = 1.4142135623731 # 根号2
num_sqrt_3 = 1.7320508075689 # 根号3

ACTIONS=np.array([

             # 二维
             #4方向
             # [0, 1],[1,0],[0,-1],[-1,0],

             # 8方向 每45°一方向
             # [num_sqrt_2/2,num_sqrt_2/2], 
             # [1,0],[num_sqrt_2/2,-num_sqrt_2/2],
             # [0,-1],
             # [-num_sqrt_2/2,-num_sqrt_2/2],
             # [-1,0],[-num_sqrt_2/2,num_sqrt_2/2]

             # # 12方向 每30°一方向
             # [0.5,num_sqrt_3/2],[num_sqrt_3/2,0.5],
             # [1,0],[num_sqrt_3/2,-0.5],
             # [0.5,-num_sqrt_3/2],[0,-1],
             # [-0.5,-num_sqrt_3/2],[-num_sqrt_3/2,-0.5],
             # [-1,0],[-num_sqrt_3/2,0.5],

             # [-0.5,num_sqrt_3/2],

             # 三维
             # 6方向
             [0,1,0],[1,0,0],[0,-1,0],[-1,0,0],[0,0,1],[0,0,-1],

             # 加上8个顶点
             [1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1],[1,1,-1],[1,-1,-1],
             [-1,-1,-1],[-1,1,-1],
             # 加上中间平面4个方向
             [1,1,0],[1,-1,0],[-1,-1,0],[-1,1,0],
             # 加上前后左右垂直平面各4个方向
             [1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],[0,-1,1],[0,-1,-1],[0,1,1],[0,1,-1]



             ],dtype=int)#the possible actions (UAV flying directions)   
ACTION_SPACE_SIZE = ACTIONS.shape[0]
   
MAX_SPEED=20 #maximum UAV speed in m/s
STEP_DISPLACEMENT=MAX_SPEED*delta_t # 每个时间步的位移 The displacement per time step

  
# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # 重写init以设置初始步骤和写入器(我们希望所有.fit()调用都使用一个日志文件) Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_writer_dir = self.log_dir
    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        # self.model = model
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        # print(self.step)
        # if self.step % 10 == 0:
        #     self._write_logs(stats, self.step)
        with self.writer.as_default():
            if self.step % 10 == 0:
                print(self.step)
                for key, value in stats.items():
                    tf.summary.scalar(key, value, step = self.step)
                    self.writer.flush()


    def _write_logs(self, logs, index):
        with self.writer.as_default():
            if self.step % 10 == 0:
                print(self.step)
                for name, value in logs.items():
                    tf.summary.scalar(name, value, step=index)
                    self.writer.flush()

class UAVEnv:
    
    def reset(self):
        self.episode_step = 0
        s0=self.random_generate_states(num_states=1)
         
        return s0
        
    def reset1(self):
        self.episode_step = 0
        # self.sum = 0
        s0 = np.array([[100, 100, 60]]) # 初始化起始点 [200, 300, 70]
        return s0

    def random_generate_states(self,num_states):
        loc_x=np.random.uniform(50,X_MAX-50,(num_states,1))
        loc_y=np.random.uniform(50,Y_MAX-50,(num_states,1))
        loc_z=np.random.choice(Z_height,(num_states,1))
        loc=np.concatenate((loc_x,loc_y,loc_z),axis=1)
        #axis=1表示对应行的数组进行拼接，得到的loc即为一个正方形区域内的所有坐标
        return loc
            

    #对于无人机访问的每个位置，它都将有来自每M个基站单元的J信号的测量 for each location visited by the UAV, it will have the J signal measurements from
    #each of the M cellular BSs
    #基于这些M*J度量，计算经验中断概率 based on these M*J measurements, calculate the empirical outage probability
    def get_empirical_outage(self, location):
        #location given in meters
        #convert the location to kilometer
        loc_km=np.zeros(shape=(1,3))
        loc_km[0,:3]=location/1000
        # loc_km[0,2]=0.1
        Pout=rad_env.getPointMiniOutage(loc_km)
              
        return Pout[0]
    

    
    def step(self, current_state, action_idx, cur_traj):#the actual step
        self.episode_step += 1
        
        # 下一状态等于当前状态坐标加上因当前行动所形成的位移              
        next_state=current_state+STEP_DISPLACEMENT*ACTIONS[action_idx]
        outbound=False
        out_bound_check1=next_state<0
        out_bound_check2=next_state[0,0]>X_MAX
        out_bound_check3=next_state[0,1]>Y_MAX
        out_bound_check4=next_state[0,2]>Z_MAX
        out_bound_check5=next_state[0,2]<Z_MIN
        # 判断下一状态是否超出范围，若超出，则重新赋值
        if out_bound_check1.any() or out_bound_check2.any() or out_bound_check3.any() or out_bound_check4.any() or out_bound_check5.any():
            outbound=True       
            next_state[next_state<0]=0
            next_state[0,0]=np.minimum(X_MAX,next_state[0,0])
            next_state[0,1]=np.minimum(Y_MAX,next_state[0,1])
            if next_state[0,2] > Z_MAX:
                next_state[0,2]=np.minimum(Z_MAX,next_state[0,2])
            if next_state[0,2] < Z_MIN:
                next_state[0,2]=np.maximum(Z_MIN,next_state[0,2])


        # 当下一状态与目的地之间相差小于我们设定的阈值时，表示到达目的地，否则未到达
        if LA.norm(next_state-DESTINATION)<=DIST_TOLERANCE:
            terminal=True
            print('Reach destination====================================================================================!!!!!!!!')
            print('total_step:'+ str(len(cur_traj)))
        else:
            terminal=False
        


        # 若到达终点或越界，则将越界惩罚加到reward上
        if terminal or outbound:
            reward=-MOVE_PENALTY

        # 否则根据得到的下一状态得到最小经验中断概率，并根据中断概率与弱覆盖区域惩罚更新reward
        else: 
            Pout=self.get_empirical_outage(next_state)
            reward=-MOVE_PENALTY-NON_COVER_PENALTY*Pout+GET_CLOSE_DES_REWARD*(LA.norm(current_state-DESTINATION)-LA.norm(next_state-DESTINATION))

            Pout=np.array(Pout)
            Pout=Pout.reshape((-1,1)) # 化成单列数据
            new_row=np.concatenate((next_state,Pout),axis=1) # 将下一状态与对应输出的中断概率数组链接
            radio_map.add_new_measured_data(new_row)# 将测量数据存储到无线电地图数据库中 store the measured data to the database of radio map
            

        done = False
        
        if terminal:
            print('rewards:'+str(reward))

        # 当到达终点或超出最大步骤或出界时，表示该回合结束                       
        if terminal or self.episode_step >= MAX_STEP or outbound:
            done = True


                   
        return next_state, reward, terminal,outbound,done
    
    
    
    def simulated_step(self,current_state, action_idx, cur_traj):
        # 模拟步骤:无人机并没有进行实际的飞行，但使用无线电地图来一步模拟飞行步骤
        #the simulated step: the UAV does not actually take the fly, but use the
        #radio map to have a simulated step              
        self.episode_step+=1
        
        # 当前状态与下一状态间的转换关系 
        next_state=current_state+STEP_DISPLACEMENT*ACTIONS[action_idx]
        outbound=False
        out_bound_check1=next_state<0
        out_bound_check2=next_state[0,0]>X_MAX
        out_bound_check3=next_state[0,1]>Y_MAX
        out_bound_check4=next_state[0,2]>Z_MAX
        out_bound_check5=next_state[0,2]<Z_MIN

        # 越界检查，当有任何一个坐标超出限定范围时，表示出界
        if out_bound_check1.any() or out_bound_check2.any() or out_bound_check3.any() or out_bound_check4.any() or out_bound_check5.any():
            outbound=True       
            next_state[next_state<0]=0
            next_state[0,0]=np.minimum(X_MAX,next_state[0,0])
            next_state[0,1]=np.minimum(Y_MAX,next_state[0,1])
            if next_state[0,2] > Z_MAX:
                next_state[0,2]=np.minimum(Z_MAX,next_state[0,2])
            if next_state[0,2] < Z_MIN:
                next_state[0,2]=np.maximum(Z_MIN,next_state[0,2])

        if LA.norm(next_state-DESTINATION)<=DIST_TOLERANCE:
            terminal=True
            print('Simulated: Reach destination================================================================================!!!!!!!!')
        else:
            terminal=False
    
        if terminal or outbound:
            reward=-MOVE_PENALTY
        else:
            #这部分使实际步骤和模拟步骤有所不同 
            #This part makes a difference between the actual step and the simulated step
            Pout=radio_map.predict_outage_prob(next_state)
            #中断概率是根据无线电图来预测的，而不是测量的
            #The outage probability is predicted based on the radio map, instead of being measured
            reward=-MOVE_PENALTY-NON_COVER_PENALTY*Pout[0]

                          
        done = False
                               
        if terminal or self.episode_step >= MAX_STEP or outbound:
            done = True
                           
        return next_state, reward, terminal,outbound,done
    
            

env = UAVEnv()
sim_env=UAVEnv()#Simulated UAV environment 
radio_map=RadioMap(X_MAX,Y_MAX)    


# 为了获得更多重复的结果 For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


# Agent class
class DQNAgent:
    def __init__(self):
        # Main model
        
        self.model = self.create_model(dueling=True)
        
        #将初始化模型关闭，此时为无模型训练
        self.initilize_model()

        # Target network
        self.target_model = self.create_model(dueling=True)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # 自定义tensorboard对象 Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs\{}-{}".format(MODEL_NAME, int(time.time())))

        # 用于计算何时用主网络的权值更新目标网络 Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self, dueling):
        # 由5个隐藏层组成的全连接前馈神经网络，前4个隐藏层神经元数量分别为512，256，128，128
        inp = Input(shape=OBSERVATION_SPACE_VALUES)
        outp=Dense(512,activation='relu')(inp)
        outp=Dense(256,activation='relu')(outp)
        outp=Dense(128,activation='relu')(outp)
        outp=Dense(128,activation='relu')(outp)
        
        if(dueling):
            # 让网络评估作为中间层的优势功能，其中一个神经元对应于状态-价值的估计，另外K个神经元对应于K个动作的动作优势，即每个状态的动作价值和状态价值之间的差异
            # Have the network estimate the Advantage function as an intermediate layer
            outp=Dense(ACTION_SPACE_SIZE+1, activation='linear')(outp)
            outp=Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(ACTION_SPACE_SIZE,))(outp)
        else:
            outp=Dense(ACTION_SPACE_SIZE,activation='linear')(outp)
            
            
        model=Model(inp,outp)
        
        model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_absolute_error', 'mean_squared_error'])
        model.summary()
        return model


       
    def normalize_data(self,input_data):        
        return input_data/MAX_VALS

       
    def initilize_model(self):       
        #初始化DQN，以便每个(状态，动作)对的Q值等于:-MOVE_PENALTY *distance / STEP_DISPLACEMENT initialize the DQN so that the Q values of each (state,action) pair
        #equal to: -MOVE_PENALTY*distance/STEP_DISPLACEMENT,
        #下一个状态到目的地的距离是多少 where distance is the distance between the next state and the destination
        #当覆盖地图上没有信息时，这将鼓励最初的最短路径飞行 this will encourage shortest path flying initially when there is no information on the coverage map
               
        xx,yy=np.meshgrid(x,y,indexing='ij')
                
        # 原 产生100000个随机位置        
        num_states=200_000
        xy_loc=env.random_generate_states(num_states)
        # xy_loc=[700,800,60] #固定起点
        
        
        Actions_aug=np.zeros((1,xy_loc.shape[1],ACTION_SPACE_SIZE),dtype=int)
        for i in range(Actions_aug.shape[2]):
            Actions_aug[:,:,i]=ACTIONS[i]

        # np.tile用于重复数组A来构建新的数组    
        Actions_aug=np.tile(Actions_aug,(xy_loc.shape[0],1,1))
        xy_loc_aug=np.zeros((xy_loc.shape[0],xy_loc.shape[1],1))
        xy_loc_aug[:,:,0]=xy_loc
        xy_loc_aug=np.repeat(xy_loc_aug,ACTION_SPACE_SIZE,axis=2)
        xy_loc_next_state=xy_loc_aug+STEP_DISPLACEMENT*Actions_aug
        
        xy_loc_next_state[xy_loc_next_state<0]=0
        xy_loc_next_state[:,0,:]=np.minimum(X_MAX,xy_loc_next_state[:,0,:])
        xy_loc_next_state[:,1,:]=np.minimum(Y_MAX,xy_loc_next_state[:,1,:])

        # if xy_loc_next_state[:,2,:].any() > Z_MAX:
        #     xy_loc_next_state[:,2,:]=np.minimum(Z_MAX,xy_loc_next_state[:,2,:])
        # if xy_loc_next_state[:,2,:].any() < Z_MIN:
        #     xy_loc_next_state[:,2,:]=np.maximum(Z_MIN,xy_loc_next_state[:,2,:])

        end_loc_reshaped=np.zeros((1,3,1)) # 原(1,2,1)
        end_loc_reshaped[0,:,0]=DESTINATION
        distance_to_destination=LA.norm(xy_loc_next_state-end_loc_reshaped,axis=1)#计算到目的地的距离 compute the distance to destination            
        Q_init=-distance_to_destination/STEP_DISPLACEMENT*MOVE_PENALTY
        
                
        train_data=xy_loc[:int(num_states*0.8),:]
        train_label=Q_init[:int(num_states*0.8),:]
             
        test_data=xy_loc[int(num_states*0.8):,:]
        test_label=Q_init[int(num_states*0.8):,:]
        
       
        history=self.model.fit(self.normalize_data(train_data),train_label,epochs=20,validation_split=0.2,verbose=2)
         
        # history.history属性记录了损失函数和其他指标的数值随epochs变化的情况            
        history_dict = history.history
        history_dict.keys()
                                                                
        mse = history_dict['mean_squared_error']
        val_mse = history_dict['val_mean_squared_error']
        mae = history_dict['mean_absolute_error']
        val_mae=history_dict['val_mean_absolute_error']
        
     
        epochs = range(1, len(mse) + 1)
        
#         plt.figure()   
        
#         plt.plot(epochs, mse, 'bo', label='Training MSE')
#         plt.plot(epochs, val_mse, 'r', label='Validation MSE')
#         plt.title('Training and validation MSE')
# #        plt.ylim(0,100)
#         plt.xlabel('Epochs')
#         plt.ylabel('MSE')
        # plt.legend()
        
        # plt.show()
        
        
        # plt.figure()   # clear figure
        
        # plt.plot(epochs, mae, 'bo', label='Training MAE')
        # plt.plot(epochs, val_mae, 'r', label='Validation MAE')
        # plt.title('Training and validation accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('MAE')
    #    plt.ylim(0,15)
        # plt.legend()
        
        # plt.show()
             
        result=self.model.evaluate(self.normalize_data(test_data),test_label)
        print(result)
                        
    
 
    #Add data to replay memory for n-step return
    #(St, At, R_nstep, S_{t+n}, terminal, outbound, done)
    #where R_nstep=R_{t+1}+gamma*R_{t+2}+gamma^2*R_{t+3}....+gamma^(nSTEP-1)*R_{t+n}
    def update_replay_memory_nStepLearning(self,slide_window,nSTEP,endEpisode):
        #update only after n steps
        if len(slide_window)<nSTEP:
            return

#        slide_window按以下顺序包含列表 slide_window contains the list in the following order:
#        (current_state,action_idx,reward,next_state,terminal,outbound,done)        
        rewards_nsteps= [transition[2] for transition in slide_window]
        discount_nsteps=DISCOUNT**np.arange(nSTEP)
        R_nstep=sum(rewards_nsteps*discount_nsteps)
        
        St=slide_window[0][0]
        At=slide_window[0][1]
        
        St_plus_n=slide_window[-1][3]
        terminal=slide_window[-1][4]
        outbound=slide_window[-1][5]
        done=slide_window[-1][6]
        
        """ 将经验存储在内存缓冲区中 Store experience in memory buffer
        """         
        self.replay_memory.append((St,At,R_nstep,St_plus_n,terminal,outbound,done))
        
             
        if endEpisode:#本回合最后几步的截断n步返回 Truncated n-step return for the last few steps at the end of the episode 
            for i in range(1,nSTEP):
                rewards_i=rewards_nsteps[i:]
                discount_i=DISCOUNT**np.arange(nSTEP-i)
                R_i=sum(rewards_i*discount_i)
                
                St_i=slide_window[i][0]
                At_i=slide_window[i][1]
                              
                self.replay_memory.append((St_i,At_i,R_i,St_plus_n,terminal,outbound,done))
            
        
    def sample_batch_from_replay_memory(self):
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_state_batch = np.zeros((MINIBATCH_SIZE, OBSERVATION_SPACE_VALUES[0]))
        next_state_batch = np.zeros((MINIBATCH_SIZE, OBSERVATION_SPACE_VALUES[0]))
        
        actions_idx, rewards, terminal, outbound, done= [], [], [],[],[]
                
        for idx, val in enumerate(minibatch):     
            current_state_batch[idx] = val[0]
            actions_idx.append(val[1])
            rewards.append(val[2])
            next_state_batch[idx] = val[3]           
            terminal.append(val[4])
            outbound.append(val[5])
            done.append(val[6])
            
        return current_state_batch, actions_idx, rewards, next_state_batch, terminal, outbound, done

  
    
    def deepDoubleQlearn(self,episode_done):
        # 只有在保存了一定数量的样本后才开始训练 Start training only if certain number of samples is already saved                 
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
                                
        current_state_batch, actions_idx, rewards, next_state_batch, terminal,outbound, done = self.sample_batch_from_replay_memory()
                    
        
        current_Q_values=self.model.predict(self.normalize_data(current_state_batch))
       
        next_Q_values_currentNetwork=self.model.predict(self.normalize_data(next_state_batch))  # 使用当前的网络来评估行动 use the current network to evaluate action
        next_actions=np.argmax(next_Q_values_currentNetwork,axis=1)        
          
        next_Q_values = self.target_model.predict(self.normalize_data(next_state_batch))  # 仍然使用目标网络来评估价值 still use the target network to evaluate value
        

        Y=current_Q_values
        
        for i in range(MINIBATCH_SIZE):
       
            if terminal[i]:
                target = rewards[i]+REACH_DES_REWARD
            elif outbound[i]:
                target=rewards[i]-OUT_BOUND_PENALTY
            else:
                target = rewards[i] + DISCOUNT**nSTEP*next_Q_values[i,next_actions[i]]
                        
            Y[i,actions_idx[i]]=target
            
     
                   
        self.model.fit(self.normalize_data(current_state_batch), Y, batch_size=MINIBATCH_SIZE,verbose=0, shuffle=False, callbacks=None)
        
        
        # 每回合更新目标网络计数器 Update target network counter every episode
        if episode_done:
            self.target_update_counter += 1            
            # 当指标达到设定值时，用主网络的权值更新目标网络
            # If counter reaches set value, update target network with weights of main network
            if self.target_update_counter >= UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
                      
   
    def choose_action(self,current_state,cur_traj,epsilon):                       
        next_possible_states=current_state+STEP_DISPLACEMENT*ACTIONS       
        
        next_possible_states[next_possible_states<0]=0
        next_possible_states[:,0]=np.minimum(next_possible_states[:,0],X_MAX)
        next_possible_states[:,1]=np.minimum(next_possible_states[:,1],Y_MAX)

        # if next_possible_states[:,2].any() > Z_MAX:
        #     next_possible_states[:,2]=np.minimum(Z_MAX,next_possible_states[:,2])
        # if next_possible_states[:,2].any() < Z_MIN:
        #     next_possible_states[:,2]=np.maximum(Z_MIN,next_possible_states[:,2])

        next_possible_states=next_possible_states.tolist()
        
        no_repetition=[]
        
        cur_traj=cur_traj[-10:] # 没有回到之前的几个地方 no return to the previous few locations
        
        for state in next_possible_states:
            no_repetition.append(state not in cur_traj)
             
        
        actions_idx_all=np.arange(ACTION_SPACE_SIZE)
        actions_idx_valid=actions_idx_all[no_repetition]
           
        if np.random.rand()<=epsilon or len(actions_idx_valid)==0:#Exploration
            action_idx=np.random.randint(0,ACTION_SPACE_SIZE) 
            return action_idx
        else:        
            Q_value=self.model.predict(self.normalize_data(current_state))
            Q_value=Q_value[0]            
            action_idx_maxVal=np.argmax(Q_value)
            if action_idx_maxVal in actions_idx_valid:
                action_idx=action_idx_maxVal
            else:
                action_idx=random.sample(actions_idx_valid.tolist(),1)
                action_idx=action_idx[0]                                                         
            return action_idx
        


agent = DQNAgent()

ep_rewards,ep_trajecotry,ep_reach_terminal,ep_outbound,ep_actions=[],[],[],[],[]
# ep_MSE,ep_MAE,ep_Max_Absolute_Error,ep_bin_cross_entr=[],[],[],[]
ep_sim_rewards,ep_sim_trajecotry,ep_sim_reach_terminal,ep_sim_outbound,ep_sim_actions=[],[],[],[],[]

# 回合迭代 Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # # Update tensorboard step every episode
    # agent.tensorboard.step = episode

    # 重启回合-重置章节奖励和步骤数 Restarting episode - reset episode reward and step number
    episode_reward = 0
    episode_sim_reward = 0

    # 重置环境并获得初始状态 Reset environment and get initial state
    current_state = env.reset() # 原reset
    cur_trajectory=[]
#    cur_trajectory=np.array([]).reshape(0,OBSERVATION_SPACE_VALUES[0])
    cur_actions=[]   
    slide_window=deque(maxlen=nSTEP)
    # 重置标志并开始迭代，直到回合结束 Reset flag and start iterating until episode ends   
    done=False
    
    
    # 模拟轨迹的起始状态 The starting state of the simulated trajectory
#    sim_current_state=env.random_generate_states(num_states=1)
    sim_current_state=sim_env.reset() #原reset
    sim_cur_trajectory=[]
    sim_slide_window=deque(maxlen=nSTEP)
    sim_done=False

    sim_step_per_act_step=int(np.floor(episode/50))# 原100 每个实际步骤执行的模拟步骤数随着回合数增加而增加 Number of simulation steps performed per actual step
    sim_step_per_act_step=np.minimum(sim_step_per_act_step,10)
    
    
    
    while not done:
        # 实际的无人机飞行 The actual UAV flight
        cur_trajectory.append(np.squeeze(current_state).tolist()) # 数组转列表
        action_idx=agent.choose_action(current_state,cur_trajectory,epsilon)
             
        # 实际步长及测量 actual step and measurement
        next_state, reward, terminal, outbound, done = env.step(current_state,action_idx,cur_trajectory)
        
        radio_map.update_radio_map(verbose_on=0)
        
        
        episode_reward += reward        
        slide_window.append((current_state,action_idx,reward,next_state,terminal,outbound,done)) 

        agent.update_replay_memory_nStepLearning(slide_window,nSTEP,done)  
        agent.deepDoubleQlearn(done)
        current_state = next_state
        
        # 模拟轨迹 The simulated trajectory
        for temp_counter in range(sim_step_per_act_step):
            sim_cur_trajectory.append(np.squeeze(sim_current_state).tolist())
            sim_action_idx=agent.choose_action(sim_current_state,sim_cur_trajectory,epsilon)
            sim_next_state, sim_reward, sim_terminal, sim_outbound, sim_done =sim_env.simulated_step(sim_current_state,sim_action_idx,sim_cur_trajectory)
            episode_sim_reward+=sim_reward
            sim_slide_window.append((sim_current_state,sim_action_idx,sim_reward,sim_next_state,sim_terminal,sim_outbound,sim_done))
            agent.update_replay_memory_nStepLearning(sim_slide_window,nSTEP,sim_done)
            agent.deepDoubleQlearn(sim_done)
            sim_current_state = sim_next_state
            if sim_done:#start a new episode if the simulation trajectory completes one episode
                ep_sim_rewards.append(episode_sim_reward)
                ep_sim_trajecotry.append(sim_cur_trajectory)
                ep_sim_reach_terminal.append(sim_terminal)
                ep_sim_outbound.append(sim_outbound)
                
                episode_sim_reward = 0
                sim_current_state=sim_env.reset() #原reset
                sim_cur_trajectory=[]
                sim_slide_window=deque(maxlen=nSTEP)
                sim_done=False
                
                

                   
    # MSE,MAE,Max_Absolute_Error,bin_cross_entr=radio_map.check_radio_map_acc()
    # ep_MSE.append(MSE)
#    print('ep_MSE:', ep_MSE)
    # ep_MAE.append(MAE)
#    print('ep_MAE:', ep_MAE)
    # ep_Max_Absolute_Error.append(Max_Absolute_Error)
#    print('ep_Max_Absolute_Error:', ep_Max_Absolute_Error)
    # ep_bin_cross_entr.append(bin_cross_entr)
#    print('bin_cross_entr:', ep_bin_cross_entr)
    
    
    
    # 在列表和日志中添加回合奖励(每个给定的回合数量) Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    ep_trajecotry.append(cur_trajectory)
    ep_reach_terminal.append(terminal)
    ep_outbound.append(outbound)
#    ep_actions.append(cur_actions)
    
   
    
    # if episode%10 == 0:
#        dist_to_dest=LA.norm(start_loc-end_loc)
#        print("Start location:{}, distance to destination:{}".format(start_loc,dist_to_dest))
        # print("Episode: {}, total steps: {},  final return: {}".format(episode,len(cur_trajectory),episode_reward))
        
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
    
        print("Episode: {}, total steps: {},  final return: {}".format(episode,len(cur_trajectory),min_reward))# episode_reward

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        
        





# fig=plt.figure(60)
# plt.plot(np.arange(len(ep_MSE))+1,ep_MSE,'b-',linewidth=2)
# plt.grid(which='major', axis='both')
# plt.xlabel('Episode')
# plt.ylabel('MSE')
# # fig.savefig('MSE.eps')
# # fig.savefig('MSE.pdf')
# fig.savefig('MSE.jpg')

# fig=plt.figure(70)
# plt.plot(np.arange(len(ep_MAE))+1,ep_MAE,'b-',linewidth=2)
# plt.grid(which='major', axis='both')
# plt.xlabel('Episode')
# plt.ylabel('MAE')
# # fig.savefig('MAE.eps')
# # fig.savefig('MAE.pdf')
# fig.savefig('MAE.jpg')

# fig=plt.figure(80)
# plt.plot(np.arange(len(ep_Max_Absolute_Error))+1,ep_Max_Absolute_Error,'b-',linewidth=2)
# plt.grid(which='major', axis='both')
# plt.xlabel('Episode')
# plt.ylabel('Maximum absolute eror')
# # fig.savefig('Max_Absolute_Error.eps')
# # fig.savefig('Max_Absolute_Error.pdf')
# fig.savefig('Max_Absolute_Error.jpg')

# fig=plt.figure(90)
# plt.plot(np.arange(len(ep_bin_cross_entr))+1,ep_bin_cross_entr,'b-',linewidth=2)
# plt.grid(which='major', axis='both')
# plt.xlabel('Episode')
# plt.ylabel('Binary cross entropy')
# # fig.savefig('Bin_cross_entr.eps')
# # fig.savefig('Bin_cross_entr.pdf')
# fig.savefig('Bin_cross_entr.jpg')
    
   

def get_moving_average(mylist,N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)

        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)
    return moving_aves
        

    

fig=plt.figure()
plt.xlabel('Episode',fontsize=14)
plt.ylabel('Return per episode',fontsize=14,labelpad=-2)
N=200 # 


return_mov_avg=get_moving_average(ep_rewards,N)



plt.plot(np.arange(len(return_mov_avg))+N,return_mov_avg,'r-',linewidth=5)
#plt.ylim(-6000,0)
# fig.savefig('return.eps')
# fig.savefig('return.pdf')
fig.savefig('return.jpg')


#---------------------------------------------保存结果图---------------------------------
#Save the simulation ressult
np.savez('SNARM_main_Results_26_act',return_mov_avg,ep_rewards,ep_trajecotry,ep_reach_terminal,ep_outbound) 
print('{}/{} episodes reach terminal'.format(ep_reach_terminal.count(True),episode))


##============VIew the learned radio map for given height
# UAV_height=0.1 # UAV height in km
step=101 #include the start point at 0 and end point, the space between two sample points is D/(step-1)
D=1
X_vec=range(D*(step-1)+1)
Y_vec=range(D*(step-1)+1)
Z_vec=[0.06,0.07,0.08,0.09,0.1]

numX,numY=np.size(X_vec),np.size(Y_vec)

OutageMapLearned_60=np.zeros(shape=(numX,numY))
OutageMapLearned_70=np.zeros(shape=(numX,numY))
OutageMapLearned_80=np.zeros(shape=(numX,numY))
OutageMapLearned_90=np.zeros(shape=(numX,numY))
OutageMapLearned_100=np.zeros(shape=(numX,numY))

#Loc_All=np.zeros(shape=(0,3))
for z_loc in range(5):
    for i in range(numX):
        Loc_cur=np.zeros(shape=(numY,3))
        Loc_cur[:,0]=X_vec[i]/step
        Loc_cur[:,1]=np.array(Y_vec)/step
        Loc_cur[:,2]=Z_vec[z_loc]
        Loc_cur=Loc_cur*1000 #convert to meter
        if z_loc == 0:
            OutageMapLearned_60[i,:]=radio_map.predict_outage_prob(Loc_cur)
        if z_loc == 1:
            OutageMapLearned_70[i,:]=radio_map.predict_outage_prob(Loc_cur)
        if z_loc == 2:
            OutageMapLearned_80[i,:]=radio_map.predict_outage_prob(Loc_cur)
        if z_loc == 3:
            OutageMapLearned_90[i,:]=radio_map.predict_outage_prob(Loc_cur)
        if z_loc == 4:
            OutageMapLearned_100[i,:]=radio_map.predict_outage_prob(Loc_cur)

        




fig=plt.figure(5)
ax = fig.add_subplot(111, projection='3d')

clist=['lightsteelblue','beige','darksalmon']
# N 表示插值后的颜色数目
newcmp = LinearSegmentedColormap.from_list('chaos',clist,N=256)

c=ax.contourf(np.array(X_vec)*10, np.array(Y_vec)*10, 1-OutageMapLearned_60, zdir='z', offset=60, cmap=newcmp)
ax.contourf(np.array(X_vec)*10, np.array(Y_vec)*10, 1-OutageMapLearned_70, zdir='z', offset=70, cmap=newcmp)
ax.contourf(np.array(X_vec)*10, np.array(Y_vec)*10, 1-OutageMapLearned_80, zdir='z', offset=80, cmap=newcmp)
ax.contourf(np.array(X_vec)*10, np.array(Y_vec)*10, 1-OutageMapLearned_90, zdir='z', offset=90, cmap=newcmp)
ax.contourf(np.array(X_vec)*10, np.array(Y_vec)*10, 1-OutageMapLearned_100, zdir='z', offset=100, cmap=newcmp)

xticks = [0,200,400,600,800,1000]
yticks = [0,200,400,600,800,1000]

ax.set_xticks(xticks)
ax.set_xlabel('X(meter)')
# ax.set_xticklabels('x/meter', rotation=500)
ax.set_xlim(-50,1050)
ax.set_yticks(yticks)
ax.set_ylabel('Y(meter)')
# ax.set_yticklabels('y/meter', rotation=500)
ax.set_ylim(1050,-50)

ax.set_zlabel('Z(meter)')
ax.set_zlim(100,60)

#底变为白色
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

#视角初始化
ax.view_init(-166, -68)


v = np.linspace(0, 1.0, 6, endpoint=True)
cbar=fig.colorbar(c,ticks=v,shrink=0.4)
cbar.set_label('coverage probability',labelpad=12, rotation=270,fontsize=14)



# 取消背景网格线
ax.grid(False)


plt.show()

# fig.savefig('CoverageMapLearned.eps')
# fig.savefig('CoverageMapLearned.pdf')
fig.savefig('CoverageMapLearned.jpg')
          


npzfile_100 = np.load('radioenvir_100m.npz')
OutageMapActual_100m=npzfile_100['arr_0']
X_vec_100=npzfile_100['arr_1']
Y_vec_100=npzfile_100['arr_2']

npzfile_90 = np.load('radioenvir_90m.npz')
OutageMapActual_90m=npzfile_90['arr_0']
X_vec_90=npzfile_90['arr_1']
Y_vec_90=npzfile_90['arr_2']

npzfile_80 = np.load('radioenvir_80m.npz')
OutageMapActual_80m=npzfile_80['arr_0']
X_vec_80=npzfile_80['arr_1']
Y_vec_80=npzfile_80['arr_2']

npzfile_70 = np.load('radioenvir_70m.npz')
OutageMapActual_70m=npzfile_70['arr_0']
X_vec_70=npzfile_70['arr_1']
Y_vec_70=npzfile_70['arr_2']

npzfile_60 = np.load('radioenvir_60m.npz')
OutageMapActual_60m=npzfile_60['arr_0']
X_vec_60=npzfile_60['arr_1']
Y_vec_60=npzfile_60['arr_2']




fig=plt.figure(30)
#xx,yy=np.meshgrid(x,y,indexing='ij')
#plt.contourf(xx,yy,coverage_map)

# plt.contourf(np.array(X_vec)*10,np.array(Y_vec)*10,1-OutageMapActual)
# v = np.linspace(0, 1.0, 11, endpoint=True)
# cbar=plt.colorbar(ticks=v) 




# ax = fig.add_subplot(111, projection='3d')

ax = Axes3D(fig)

clist=['lightsteelblue','beige','darksalmon']
# N 表示插值后的颜色数目
newcmp = LinearSegmentedColormap.from_list('chaos',clist,N=256)

c=ax.contourf(np.array(X_vec_60)*10, np.array(Y_vec_60)*10, 1-OutageMapActual_60m, zdir='z', offset=60, cmap=newcmp,zorder=1,alpha=0.9)
ax.contourf(np.array(X_vec_70)*10, np.array(Y_vec_70)*10, 1-OutageMapActual_70m, zdir='z', offset=70, cmap=newcmp,zorder=1,alpha=0.9)
ax.contourf(np.array(X_vec_80)*10, np.array(Y_vec_80)*10, 1-OutageMapActual_80m, zdir='z', offset=80, cmap=newcmp,zorder=1,alpha=0.9)
ax.contourf(np.array(X_vec_90)*10, np.array(Y_vec_90)*10, 1-OutageMapActual_90m, zdir='z', offset=90, cmap=newcmp,zorder=1,alpha=0.9)
ax.contourf(np.array(X_vec_100)*10, np.array(Y_vec_100)*10, 1-OutageMapActual_100m, zdir='z', offset=100, cmap=newcmp,zorder=1,alpha=0.9)

xticks = [0,200,400,600,800,1000]
yticks = [0,200,400,600,800,1000]





zorder_th = 10
for episode_idx in range(0, episode): # 原episode-200
    S_seq=ep_trajecotry[episode_idx]
    S_seq=np.squeeze(np.asarray(S_seq))

    # if S_seq.ndim == 1:   
    #     plt.plot(S_seq[0],S_seq[1],'rx',markersize=5)
    #     plt.plot(S_seq[0],S_seq[1],'-')

    if S_seq.ndim == 2:   

        if ep_reach_terminal[episode_idx]==True and S_seq[0,0] < 150 and S_seq[0,1] < 150 and S_seq[0,2] == 60:
            ax.text3D(S_seq[0,0],S_seq[0,1],S_seq[0,2]+1,'$q_s$',zdir='x',fontsize=14,zorder=20)
            ax.scatter3D(S_seq[0,0],S_seq[0,1],S_seq[0,2]+1,marker='^',s=80,zorder=20) #,'rx',markersize=5
            ax.plot(S_seq[:,0],S_seq[:,1],S_seq[:,2]+1,zdir='z',linestyle=':',linewidth=2,
                marker='8',fillstyle='full',markersize=4,markevery=1,visible=True,alpha=1,zorder=200,drawstyle='steps')

        # ax.legend(['First line', 'Second line'])
    if S_seq.ndim == 3:
        print('weidu_3:') 
        plt.plot(S_seq[0,0],S_seq[0,1],'rx',markersize=5) # ,S_seq[0,2]
        plt.plot(S_seq[:,0],S_seq[:,1],'-')# ,S_seq[:,2]
        

ax.text3D(DESTINATION[0,0],DESTINATION[0,1],DESTINATION[0,2]+1,'$q_f$',zdir='x',fontsize=14,zorder=20)
ax.scatter3D(DESTINATION[0,0],DESTINATION[0,1],DESTINATION[0,2],marker='D',s=200,zorder=20) 

# axes.set_xticks(xticks)
ax.set_xlabel('X(meter)')
ax.set_xlim(-50,1050)
# axes.set_yticks(yticks)
ax.set_ylabel('Y(meter)')
ax.set_ylim(1050,-50)


ax.set_zlabel('Z(meter)')
ax.set_zlim(100,60)

#底变为白色
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

#视角初始化
ax.view_init(-167, 105)

v = np.linspace(0, 1.0, 6, endpoint=True)
cbar=fig.colorbar(c,ticks=v,shrink=0.4)
cbar.set_label('coverage probability',labelpad=12, rotation=270,fontsize=10)

# 取消背景网格线
ax.grid(False)

plt.show()
# fig.savefig('trajectoriesSNARM.eps')
# fig.savefig('trajectoriesSNARM.pdf')
fig.savefig('trajectoriesSNARM.jpg')


