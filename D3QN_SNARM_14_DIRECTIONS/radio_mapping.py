# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:20:58 2019
#Radio mapping based on the measured empirical outage probabilities of the visited locations by UAV
#Based on the measured data, train a DNN for radio mapping:
#Inpt: location, output： outage probability at the location 
@author: Yong Zeng
"""
import numpy as np
#from keras.callbacks import TensorBoard
#import tensorflow as tf
#from collections import deque
#import time
import random
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
#from tqdm import tqdm
#import os
#from sklearn.utils import shuffle
#import scipy.io as spio
#import matplotlib.pyplot as plt
#from numpy import linalg as LA

import radio_environment as rad_env


class RadioMap:
    def __init__(self,X_MAX_VAL=1000.0,Y_MAX_VAL=1000.0,Z_MAX_VAL=100.0):
        # Main model
        self.LOC_DIM=3 # 原2 2D or 3D trajectory
        self.MINIBATCH_SIZE = 64
        self.X_MAX=X_MAX_VAL
        self.Y_MAX=Y_MAX_VAL#the area region in meters
        self.Z_MAX=Z_MAX_VAL

        self.measured_database=np.zeros(shape=(0,self.LOC_DIM+1), dtype=np.float32)
        #存储无人机访问的所有地点的实测中断概率的数据库
        #the database storing the mesaured empirical outage probabilities for all locations visited by UAV
        #每一行对应于一个位置的测量数据。最后一个元素对应中断概率
        #each row corresponds to the measured data at one location. The last element correspond to the outage probability
        #前2或3个元素提供2D/3D位置the first 2 or 3 elements give the 2D/3D location 
        self.OUTPUT_ACT='softmax' 
        # softmax激活函数能够将输入序列的每一个值挤压到[0,1]范围内，并且所有项的和为1，
        # softmax函数的输出值相互关联，其概率的总和始终为1，互斥输出。
#        self.OUTPUT_ACT='sigmoid'
        #sigmoid函数会分别处理各个原始输出值，因此其结果相互独立，概率总和不一定为1。非独占输出
        self.map_model = self.create_map_model()      
#        self.initilize_map_model()

    #无线电映射ANN（人工神经网络），由5个隐藏层组成，所有隐藏层由Relu激活
    def create_map_model(self):
        inp = Input(shape=(self.LOC_DIM,))
        outp=Dense(512,activation='relu')(inp)
        outp=Dense(256,activation='relu')(outp)
        outp=Dense(128,activation='relu')(outp)
        outp=Dense(64,activation='relu')(outp)
        outp=Dense(32,activation='relu')(outp)
        if self.OUTPUT_ACT=='sigmoid': 
            outp=Dense(1,activation='sigmoid')(outp)
        else:
            outp=Dense(2,activation='softmax')(outp)
                
            
        model=Model(inp,outp)
        
        # model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
        if self.OUTPUT_ACT=='sigmoid':
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        else:#with softmax output
            model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
            
        model.summary() # 输出各层参数的状态
        return model
    
    def add_new_measured_data(self,new_row):
        self.measured_database=np.concatenate((self.measured_database,new_row),axis=0)
        
    
    def generate_random_locations(self,num_loc):
        loc_x=np.random.uniform(50,self.X_MAX-50,(num_loc,1))
        loc_y=np.random.uniform(50,self.Y_MAX-50,(num_loc,1))
        loc=np.concatenate((loc_x,loc_y),axis=1)
        return loc
        
    
    def normalize_location(self,location):#将位置坐标归一到0和1之间的范围 normalize the location coordinate to the range betwee 0 and 1 
        MAX_VALS=np.array([[self.X_MAX,self.Y_MAX,self.Z_MAX]])
        return location/MAX_VALS
    
    def predict_outage_prob(self,location):
         pred_array=self.map_model.predict(self.normalize_location(location))
         return pred_array[:,0]
        
    def update_radio_map(self,verbose_on=0):
        if self.measured_database.shape[0]<np.maximum(100,self.MINIBATCH_SIZE):
            return
         # random.sample多用于截取列表的指定长度的随机数，但是不会改变列表本身的排序
        sampled_idx=random.sample(range(self.measured_database.shape[0]),self.MINIBATCH_SIZE)
        train_data=self.measured_database[sampled_idx,:self.LOC_DIM]
        train_label=self.measured_database[sampled_idx,-1]
        train_label=train_label.reshape((-1,1))
        
        if self.OUTPUT_ACT=='softmax':
            train_label=np.concatenate((train_label,1.0-train_label),axis=1) # axis=1表示对应行的数组进行拼接
            
            
        self.map_model.fit(self.normalize_location(train_data),train_label,verbose=verbose_on)
        

        
        
    def check_radio_map_acc(self):
        pred_outage=self.predict_outage_prob(rad_env.TEST_LOC_meter)

        diff_abs=np.abs(rad_env.TEST_LOC_ACTUAL_OUTAGE-pred_outage)
        
        MSE=np.sum(np.square(diff_abs))/len(rad_env.TEST_LOC_ACTUAL_OUTAGE)
        
        MAE=np.sum(diff_abs)/len(rad_env.TEST_LOC_ACTUAL_OUTAGE)
        
        Max_Absolute_Error=np.max(diff_abs)
        
        bin_cross_entr=self.binary_cross_entropy(rad_env.TEST_LOC_ACTUAL_OUTAGE,pred_outage)
        
        return MSE,MAE,Max_Absolute_Error,bin_cross_entr
        
  
    def binary_cross_entropy(self,p_true,p_pred):
        N=len(p_true)
        return -1/N*np.sum(p_true*np.log(p_pred)+(1-p_true)*np.log(1-p_pred))
    
   

