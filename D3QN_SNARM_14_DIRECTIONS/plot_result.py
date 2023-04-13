

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

import random
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=20)
# fig.suptitle(u'UAV处于不同高度下的无线电分布图', fontproperties=font)




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

from matplotlib.text import Annotation
class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)



# 从垂直平面看天线增益
##the antenna gain viewed from vertical plane
# plt.axes(polar=True)
# plt.plot(np.deg2rad(angleVvector),GarraydBmatrix[:,180],c='k')
# plt.title('Vertical Antenna Gain with azimuth angle 0')
# plt.show()
# 从水平面上观察到的天线增益
##The antenna gain viewed from the horizontal plane
# plt.axes(polar=True)
# plt.plot(np.deg2rad(angleHvector),GarraydBmatrix[101,:],c='k')
# plt.title('Horizontal Antenna Gain with vertial angle 90')
# plt.show()
# 3 d天线增益
## 3D antenna gain
# THETA, PHI= np.meshgrid(np.deg2rad(angleHvector), np.deg2rad(angleVvector))
# R = GarraydBmatrix
# Rmax = np.max(R)
#
# X = R * np.sin(THETA) * np.cos(PHI)
# Y = R * np.sin(THETA) * np.sin(PHI)
# Z = R * np.cos(THETA)
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# plot = ax.plot_surface(
#    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
#    linewidth=0, antialiased=False, alpha=0.5)
#
# ax.view_init(30, 0)
# plt.show()





#----------------------------------画轨迹图-------------------------------------
# 

# npzfile = np.load('SNARM_main_Results.npz',allow_pickle=True)

# return_mov_avg=npzfile['arr_0']
# ep_rewards=npzfile['arr_1']
# tra_result=npzfile['arr_2']


#----------------------------------画概率分布图----------------------------------
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


# #-----------------------画空白3D图--------------------------------------
# fig=plt.figure(5)

# plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace =0, hspace =0)
# plt.subplots(constrained_layout=True)

# ax = fig.add_subplot(111, projection='3d')
# # ax = Axes3D(fig)


# clist=['lightsteelblue','beige','darksalmon']
# # N 表示插值后的颜色数目
# newcmp = LinearSegmentedColormap.from_list('chaos',clist,N=256)



# c=ax.contourf(np.array(X_vec_60)*10, np.array(Y_vec_60)*10, 1-OutageMapActual_60m, zdir='z', offset=60, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_70)*10, np.array(Y_vec_70)*10, 1-OutageMapActual_70m, zdir='z', offset=70, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_80)*10, np.array(Y_vec_80)*10, 1-OutageMapActual_80m, zdir='z', offset=80, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_90)*10, np.array(Y_vec_90)*10, 1-OutageMapActual_90m, zdir='z', offset=90, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_100)*10, np.array(Y_vec_100)*10, 1-OutageMapActual_100m, zdir='z', offset=100, cmap=newcmp,alpha=1)



# xticks = [0,200,400,600,800,1000]
# yticks = [0,200,400,600,800,1000]

# # ax.set_xticks(xticks)
# ax.set_xlabel('X(meter)')
# ax.set_xlim(-50,1050)
# # ax.set_yticks(yticks)
# ax.set_ylabel('Y(meter)')
# ax.set_ylim(1050,-50)
# ax.set_zlabel('Z(meter)')
# ax.set_zlim(100,60)

# position=fig.add_axes([0.88, 0.2, 0.02, 0.4])#位置[左,下,右,上]
# v = np.linspace(0, 1.0, 6, endpoint=True)
# cb=plt.colorbar(c,cax=position,ticks=v,orientation='vertical')#方向
# cb.set_label(u'无 线 电 信 号 覆 盖 概 率',labelpad=16, rotation=270,fontproperties=font,fontsize=16)


# #视角初始化
# ax.view_init(-167, -79)
# # 取消背景网格线
# ax.grid(False)

# plt.show()
# fig.savefig('5_Pics.jpg')

#---------------------------------------结束----------------------------




#----------------------------------多图合一 2D UAV处于不同高度下的无线电分布图
# fig=plt.figure(6)

# fig, axes = plt.subplots(2, 2, figsize=(8, 8))


# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=16)
# fig.suptitle(u'UAV处于不同高度下的无线电分布图', fontproperties=font)

# clist=['lightsteelblue','beige','darksalmon']
# # N 表示插值后的颜色数目
# newcmp = LinearSegmentedColormap.from_list('chaos',clist,N=256)

# plt.subplots_adjust(wspace =0.3, hspace =0.3)
# flag=0

# for ax in axes.flat:
# 	flag=flag+1

# 	if flag == 1:
# 	# ax = fig.add_subplot(231)
# 		ax.contourf(np.array(X_vec_60) * 10, np.array(Y_vec_60) * 10, 1 - OutageMapActual_60m,label='60m',cmap=newcmp)
# 		ax.set_title('60m')
# 		ax.set_xlabel('x(meter)')
# 		ax.set_ylabel('y(meter)')
# 	if flag == 2:
# 	# ax = fig.add_subplot(232)
# 		ax.contourf(np.array(X_vec_70) * 10, np.array(Y_vec_70) * 10, 1 - OutageMapActual_70m,label='70m',cmap=newcmp)
# 		ax.set_title('70m')
# 		ax.set_xlabel('x(meter)')
# 		ax.set_ylabel('y(meter)')
# 	if flag == 3:
# 	# ax = fig.add_subplot(234)
# 		ax.contourf(np.array(X_vec_80) * 10, np.array(Y_vec_80) * 10, 1 - OutageMapActual_80m,label='80m',cmap=newcmp)
# 		ax.set_title('80m')
# 		ax.set_xlabel('x(meter)')
# 		ax.set_ylabel('y(meter)')
# 	if flag == 4:
# 	# ax = fig.add_subplot(235)
# 		c=ax.contourf(np.array(X_vec_90) * 10, np.array(Y_vec_90) * 10, 1 - OutageMapActual_90m,label='90m',cmap=newcmp)
# 		ax.set_title('90m')
# 		ax.set_xlabel('x(meter)')
# 		ax.set_ylabel('y(meter)')

# # ax = fig.add_subplot(236)

# v = np.linspace(0, 1.0, 6, endpoint=True)
# cbar=fig.colorbar(c,ax=axes,ticks=v,shrink=0.8, aspect=10,fraction=0.05) #fraction=0.05调整colorbar大小
# cbar.set_label(u'无 线 电 信 号 覆 盖 概 率',fontproperties=font,labelpad=20, rotation=270,fontsize=16)

# # ax = fig.add_subplot(235)
# # plt.contourf(np.array(X_vec_100) * 10, np.array(Y_vec_100) * 10, 1 - OutageMapActual_100m)

# plt.show()
# fig.savefig('2D_4_Pics.jpg')

#多图合一 2D 结束-------------------------------------------------------------




# # --------------------------------------SNARM VS D3QN 每回合平均奖励对比------------------------



# SNARM_res=np.load('SNARM_main_Results_26_act.npz',allow_pickle=True)
# D3QN_res=np.load('100c_Dueling_DDQN_MultiStepLeaning_main_Results_26_act.npz',allow_pickle=True)

# # SNARM_res=np.load('SNARM_main_Results_26_act.npz',allow_pickle=True)
# # D3QN_res=np.load('100c_Dueling_DDQN_MultiStepLeaning_main_Results_26_act.npz',allow_pickle=True)



# SNARM_res_14=np.load('SNARM_main_Results_14_act.npz',allow_pickle=True)
# # SNARM_res_14=np.load('5_20_Dueling_DDQN_MultiStepLeaning_main_Results_14_act.npz',allow_pickle=True)
# D3QN_res_14=np.load('Dueling_DDQN_MultiStepLeaning_main_Results_14_act.npz',allow_pickle=True)


# return_avg=SNARM_res['arr_0']
# reward=SNARM_res['arr_1']
# tra=SNARM_res['arr_2']
# reach_flag=SNARM_res['arr_3']

# D_return_avg=D3QN_res['arr_0']
# D_tra=D3QN_res['arr_2']


# return_avg_14=SNARM_res_14['arr_0']
# tra_14=SNARM_res_14['arr_2']

# D_return_avg_14=D3QN_res_14['arr_0']
# D_tra_14=D3QN_res_14['arr_2']

# reach_flag=reach_flag.tolist()

# print('{}/{} episodes reach terminal'.format(reach_flag.count(True),len(reach_flag)))






# print(len(return_avg))

# # 26方向
# fig=plt.figure()

# plt.subplots_adjust(wspace =0.2, hspace =0.2)
# ax = fig.add_subplot(121)

# # plt.xlabel(u'回合数',fontsize=14, fontproperties=font)
# # plt.ylabel(u'每回合平均奖励',fontsize=14,labelpad=0, fontproperties=font)


# ax.set_xlim(0,5000)
# ax.set_ylim(-200,1000)
# ax.plot(np.arange(4801),return_avg[:4801],'r-',linewidth=5,label='SNARM_26_DIRECTIONS')#len(return_avg)
# ax.plot(np.arange(5000),D_return_avg[:5000],'b-',linewidth=5,label='D3QN_26_DIRECTIONS')#len(D_return_avg)
# ax.legend(ncol=2,loc='upper left',fontsize=12)
# ax.grid(True)
# # plt.show()


# # fig.tight_layout()

# # fig.savefig('SNARM_vs_D3QN_26_return_avg.jpg')



# # 14方向
# # fig=plt.figure()
# ax = fig.add_subplot(122)
# #plt.xlabel
# #plt.ylabel labelpad=0,
# #len(return_avg_14)
# ax.set_xlim(0,5000)
# ax.set_ylim(-200,1000)

# ax.plot(np.arange(len(return_avg_14)),return_avg_14,'r-',linewidth=5,label='SNARM_14_DIRECTIONS')
# ax.plot(np.arange(4801),D_return_avg_14[:4801],'b-',linewidth=5,label='D3QN_14_DIRECTIONS')
# ax.legend(ncol=2,loc='upper left',fontsize=12)
# #

# ax.grid(True)
# fig.text(0.51, 0.05, u'回合数', ha='center',fontsize=25, fontproperties=font)
# fig.text(0.06, 0.5,u'每回合平均奖励',va='center', rotation='vertical',fontsize=25, fontproperties=font)

# plt.show()


# # fig.tight_layout()

# fig.savefig('SNARM_vs_D3QN_26_vs_14_return_avg.jpg')



# # --------------------------------------SNARM VS D3QN 每回合平均奖励对比结束------------------------






# #-----------------------------------------------------多张合一 起点为[100,100,60] 3D轨迹--------------------------------

# SNARM_res=np.load('100_SNARM_main_Results_26_act.npz',allow_pickle=True)
# D3QN_res=np.load('100_Dueling_DDQN_MultiStepLeaning_main_Results_26_act.npz',allow_pickle=True)

# # SNARM_res=np.load('SNARM_main_Results_26_act.npz',allow_pickle=True)
# # D3QN_res=np.load('100c_Dueling_DDQN_MultiStepLeaning_main_Results_26_act.npz',allow_pickle=True)



# SNARM_res_14=np.load('100_SNARM_main_Results_14_act.npz',allow_pickle=True)
# # SNARM_res_14=np.load('5_20_Dueling_DDQN_MultiStepLeaning_main_Results_14_act.npz',allow_pickle=True)
# D3QN_res_14=np.load('100_Dueling_DDQN_MultiStepLeaning_main_Results_14_act.npz',allow_pickle=True)


# return_avg=SNARM_res['arr_0']
# reward=SNARM_res['arr_1']
# tra=SNARM_res['arr_2']
# reach_flag=SNARM_res['arr_3']

# D_return_avg=D3QN_res['arr_0']
# D_tra=D3QN_res['arr_2']


# return_avg_14=SNARM_res_14['arr_0']
# tra_14=SNARM_res_14['arr_2']

# D_return_avg_14=D3QN_res_14['arr_0']
# D_tra_14=D3QN_res_14['arr_2']

# reach_flag=reach_flag.tolist()

# print('{}/{} episodes reach terminal'.format(reach_flag.count(True),len(reach_flag)))



# fig=plt.figure(5)

# plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace =0, hspace =0)
# plt.subplots(constrained_layout=True)

# ax = fig.add_subplot(121, projection='3d')
# # ax = Axes3D(fig)


# clist=['lightsteelblue','beige','darksalmon']
# # N 表示插值后的颜色数目
# newcmp = LinearSegmentedColormap.from_list('chaos',clist,N=256)



# c=ax.contourf(np.array(X_vec_60)*10, np.array(Y_vec_60)*10, 1-OutageMapActual_60m, zdir='z', offset=60, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_70)*10, np.array(Y_vec_70)*10, 1-OutageMapActual_70m, zdir='z', offset=70, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_80)*10, np.array(Y_vec_80)*10, 1-OutageMapActual_80m, zdir='z', offset=80, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_90)*10, np.array(Y_vec_90)*10, 1-OutageMapActual_90m, zdir='z', offset=90, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_100)*10, np.array(Y_vec_100)*10, 1-OutageMapActual_100m, zdir='z', offset=100, cmap=newcmp,alpha=1)

# ax.text3D(100,100,62,'$q_s$',zdir='x',fontsize=20,zorder=100)
# ax.scatter3D(95,20,58.5,marker='^',color='black',zdir='z',s=300,zorder=200)

# ax.text3D(800,800,104,'$q_f$',zdir='x',fontsize=20,zorder=100)
# ax.scatter3D(800,800,102.5,marker='D',zdir='z',s=200,zorder=200) 

# S_episode_idx=6503
# D_episode_idx=9408

# S_seq=tra[S_episode_idx]

# S_seq=np.squeeze(np.asarray(S_seq))
# S_seq[0,0]=100
# S_seq[0,1]=100
# S_seq[0,2]=60
# S_seq[len(S_seq)-1,0]=800
# S_seq[len(S_seq)-1,1]=800
# S_seq[len(S_seq)-1,2]=100
# if S_seq.ndim == 2:  
#     ax.plot(S_seq[:,0],S_seq[:,1],S_seq[:,2],zdir='z',color='r',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=1,zorder=200,label='SNARM_26_DIRECTIONS')

#     print(len(S_seq))
#     # 
#     print(S_episode_idx)

# D_seq=D_tra[D_episode_idx]
# D_seq=np.squeeze(np.asarray(D_seq))

# D_seq[0,0]=100
# D_seq[0,1]=100
# D_seq[0,2]=60
# D_seq[len(D_seq)-1,0]=800
# D_seq[len(D_seq)-1,1]=800
# D_seq[len(D_seq)-1,2]=100

# if D_seq.ndim == 2:  
#     ax.plot(D_seq[:,0],D_seq[:,1],D_seq[:,2],zdir='z',color='b',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=0.5,zorder=200,label='D3QN_26_DIRECTIONS')
#     print(len(D_seq))
#     # 
#     print(D_episode_idx)


# xticks = [0,200,400,600,800,1000]
# yticks = [0,200,400,600,800,1000]

# # ax.set_xticks(xticks)
# ax.set_xlabel('X(meter)')
# ax.set_xlim(-50,1050)
# # ax.set_yticks(yticks)
# ax.set_ylabel('Y(meter)')
# ax.set_ylim(1050,-50)
# ax.set_zlabel('Z(meter)')
# ax.set_zlim(100,60)


# #视角初始化
# ax.view_init(-167, -79)
# # 取消背景网格线
# ax.grid(False)

# position=fig.add_axes([0.43, 0.20, 0.018, 0.35])#位置[左,下,右,上]
# v = np.linspace(0, 1.0, 6, endpoint=True)
# cb=plt.colorbar(c,cax=position,ticks=v,orientation='vertical')#方向
# cb.set_label(u'无 线 电 信 号 覆 盖 概 率',labelpad=16, rotation=270,fontproperties=font,fontsize=16)

# # ax.annotate3D('point 1', (100, 100, 65), xytext=(3, 3), textcoords='offset points',zorder=200)
# # ax.annotate3D('point 2', (200, 100, 65),
# #               xytext=(-30, -30),
# #               textcoords='offset points',
# #               arrowprops=dict(ec='black', fc='white', shrink=2.5),zorder=200)
# ax.annotate3D('total steps:113,\ncumulative rewards:1643.59', (80, 200, 70),
#               xytext=(-70, 205),
#               textcoords='offset points',
#               bbox=dict(boxstyle="round", fc="lightyellow"),
#               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),zorder=200)

# ax.annotate3D('total steps:93,\ncumulative rewards:1419.77', (770, 520, 70),
#               xytext=(50, 175),
#               textcoords='offset points',
#               bbox=dict(boxstyle="round", fc="lightyellow"),
#               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),zorder=200)

# ax.legend(loc='upper center',fontsize=16)#["SNARM_26_DIRECTIONS", "D3QN_26_DIRECTIONS"],

# # cbar=fig.colorbar(c,ticks=v,shrink=0.5, aspect=10)

# # plt.tight_layout()

# # -----------------------------------------第二张图-------------------------------------------------
# # fig.subplots_adjust(wspace = .1,hspace = 0)


# ax = fig.add_subplot(122, projection='3d')
# c=ax.contourf(np.array(X_vec_60)*10, np.array(Y_vec_60)*10, 1-OutageMapActual_60m, zdir='z', offset=60, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_70)*10, np.array(Y_vec_70)*10, 1-OutageMapActual_70m, zdir='z', offset=70, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_80)*10, np.array(Y_vec_80)*10, 1-OutageMapActual_80m, zdir='z', offset=80, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_90)*10, np.array(Y_vec_90)*10, 1-OutageMapActual_90m, zdir='z', offset=90, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_100)*10, np.array(Y_vec_100)*10, 1-OutageMapActual_100m, zdir='z', offset=100, cmap=newcmp,alpha=1)

# ax.text3D(100,100,62,'$q_s$',zdir='x',fontsize=20,zorder=100)
# ax.scatter3D(92,20,58.5,marker='^',color='black',zdir='z',s=300,zorder=200)

# ax.text3D(800,800,104,'$q_f$',zdir='x',fontsize=20,zorder=100)
# ax.scatter3D(800,800,102.5,marker='D',zdir='z',s=200,zorder=200) 

# S_episode_idx_14=9503
# D_episode_idx_14=8155

# S_seq_14=tra_14[S_episode_idx_14]

# S_seq_14=np.squeeze(np.asarray(S_seq_14))
# S_seq_14[0,0]=100
# S_seq_14[0,1]=100
# S_seq_14[0,2]=60
# S_seq_14[len(S_seq_14)-1,0]=800
# S_seq_14[len(S_seq_14)-1,1]=800
# S_seq_14[len(S_seq_14)-1,2]=100
# if S_seq_14.ndim == 2:  
#     ax.plot(S_seq_14[:,0],S_seq_14[:,1],S_seq_14[:,2],zdir='z',color='r',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=1,zorder=200,label='SNARM_14_DIRECTIONS')


# D_seq_14=D_tra_14[D_episode_idx_14]
# D_seq_14=np.squeeze(np.asarray(D_seq_14))
# D_seq_14[0,0]=100
# D_seq_14[0,1]=100
# D_seq_14[0,2]=60
# D_seq_14[len(D_seq_14)-1,0]=800
# D_seq_14[len(D_seq_14)-1,1]=800
# D_seq_14[len(D_seq_14)-1,2]=100

# if D_seq_14.ndim == 2:  
#     ax.plot(D_seq_14[:,0],D_seq_14[:,1],D_seq_14[:,2],zdir='z',color='b',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=0.5,zorder=200,label='D3QN_14_DIRECTIONS')


# xticks = [0,200,400,600,800,1000]
# yticks = [0,200,400,600,800,1000]

# # ax.set_xticks(xticks)
# ax.set_xlabel('X(meter)')
# ax.set_xlim(-50,1050)
# # ax.set_yticks(yticks)
# ax.set_ylabel('Y(meter)')
# ax.set_ylim(1050,-50)
# ax.set_zlabel('Z(meter)')
# ax.set_zlim(100,60)

# #视角初始化
# ax.view_init(-167, -79)


# # 取消背景网格线
# ax.grid(False)


# ax.annotate3D('total steps:124,\ncumulative rewards: 1631.14', (70, 200, 70),
#               xytext=(-70, 205),
#               textcoords='offset points',
#               bbox=dict(boxstyle="round", fc="lightyellow"),
#               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),zorder=200)

# ax.annotate3D('total steps:132,\ncumulative rewards: 1406.99', (750, 520, 70),
#               xytext=(50, 175),
#               textcoords='offset points',
#               bbox=dict(boxstyle="round", fc="lightyellow"),
#               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),zorder=200)


# ax.legend(loc='upper center',fontsize=16)#["SNARM_14_DIRECTIONS", "D3QN_14_DIRECTIONS"], loc=1
# # fig.subplots_adjust(bottom=0, right=0.1, top=0.1)


# # v = np.linspace(0, 1.0, 6, endpoint=True)
# # cbar=fig.colorbar(c,ticks=v,shrink=0.5, aspect=10)

# position=fig.add_axes([0.93, 0.2, 0.018, 0.35])#位置[左,下,右,上]
# v = np.linspace(0, 1.0, 6, endpoint=True)
# cb=plt.colorbar(c,cax=position,ticks=v,orientation='vertical')#方向
# cb.set_label(u'无 线 电 信 号 覆 盖 概 率',labelpad=16, rotation=270,fontproperties=font,fontsize=16)

# plt.tight_layout()

# plt.show()


# fig.savefig('3D_60m_pics.jpg')
# #----------------------------------结束----------------------------------






# #-----------------------------------------------------多张合一 起点为[100,100,100] 3D轨迹--------------------------------




# SNARM_res=np.load('SNARM_main_Results_26_act.npz',allow_pickle=True)
# D3QN_res=np.load('100c_Dueling_DDQN_MultiStepLeaning_main_Results_26_act.npz',allow_pickle=True)

# # SNARM_res=np.load('SNARM_main_Results_26_act.npz',allow_pickle=True)
# # D3QN_res=np.load('100c_Dueling_DDQN_MultiStepLeaning_main_Results_26_act.npz',allow_pickle=True)



# # SNARM_res_14=np.load('SNARM_main_Results_14_act.npz',allow_pickle=True)
# SNARM_res_14=np.load('5_20_Dueling_DDQN_MultiStepLeaning_main_Results_14_act.npz',allow_pickle=True)
# D3QN_res_14=np.load('5_20_Dueling_DDQN_MultiStepLeaning_main_Results_14_act.npz',allow_pickle=True)


# return_avg=SNARM_res['arr_0']
# reward=SNARM_res['arr_1']
# tra=SNARM_res['arr_2']
# reach_flag=SNARM_res['arr_3']

# D_return_avg=D3QN_res['arr_0']
# D_tra=D3QN_res['arr_2']


# return_avg_14=SNARM_res_14['arr_0']
# tra_14=SNARM_res_14['arr_2']

# D_return_avg_14=D3QN_res_14['arr_0']
# D_tra_14=D3QN_res_14['arr_2']

# reach_flag=reach_flag.tolist()

# print('{}/{} episodes reach terminal'.format(reach_flag.count(True),len(reach_flag)))






# fig=plt.figure(5)

# plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace =0, hspace =0)
# plt.subplots(constrained_layout=True)

# ax = fig.add_subplot(121, projection='3d')
# # ax = Axes3D(fig)


# clist=['lightsteelblue','beige','darksalmon']
# # N 表示插值后的颜色数目
# newcmp = LinearSegmentedColormap.from_list('chaos',clist,N=256)



# c=ax.contourf(np.array(X_vec_60)*10, np.array(Y_vec_60)*10, 1-OutageMapActual_60m, zdir='z', offset=60, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_70)*10, np.array(Y_vec_70)*10, 1-OutageMapActual_70m, zdir='z', offset=70, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_80)*10, np.array(Y_vec_80)*10, 1-OutageMapActual_80m, zdir='z', offset=80, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_90)*10, np.array(Y_vec_90)*10, 1-OutageMapActual_90m, zdir='z', offset=90, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_100)*10, np.array(Y_vec_100)*10, 1-OutageMapActual_100m, zdir='z', offset=100, cmap=newcmp,alpha=1)

# ax.text3D(100,100,102,'$q_s$',zdir='x',fontsize=20,zorder=100)
# ax.scatter3D(86,5,99,marker='^',color='black',zdir='z',s=300,zorder=200)

# ax.text3D(820,780,102,'$q_f$',zdir='x',fontsize=20,zorder=100)
# ax.scatter3D(800,800,102.5,marker='D',zdir='z',s=200,zorder=200) 

# S_episode_idx=4307
# D_episode_idx=3142

# S_seq=tra[S_episode_idx]

# S_seq=np.squeeze(np.asarray(S_seq))
# S_seq[0,0]=100
# S_seq[0,1]=100
# S_seq[0,2]=100
# S_seq[len(S_seq)-1,0]=800
# S_seq[len(S_seq)-1,1]=800
# S_seq[len(S_seq)-1,2]=100



# D_seq=D_tra[D_episode_idx]
# D_seq=np.squeeze(np.asarray(D_seq))
# D_seq[0,0]=100
# D_seq[0,1]=100
# D_seq[0,2]=100
# D_seq[len(D_seq)-1,0]=800
# D_seq[len(D_seq)-1,1]=800
# D_seq[len(D_seq)-1,2]=100



# if S_seq.ndim == 2:  
#     ax.plot(D_seq[:,0],D_seq[:,1],D_seq[:,2],zdir='z',color='r',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=0.5,zorder=200,label='SNARM_26_DIRECTIONS')


#     print(len(S_seq))
#     # 
#     print(S_episode_idx)




# if D_seq.ndim == 2:  
#     ax.plot(S_seq[:,0],S_seq[:,1],S_seq[:,2],zdir='z',color='b',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=1,zorder=200,label='D3QN_26_DIRECTIONS')
#     print(len(D_seq))
#     # 
#     print(D_episode_idx)


# ax.plot([100,800],[100,800],[100,100],zdir='z',color='g',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=1,zorder=200,label='STRAIGHT_FLIGHT')


# xticks = [0,200,400,600,800,1000]
# yticks = [0,200,400,600,800,1000]

# # ax.set_xticks(xticks)
# ax.set_xlabel('X(meter)')
# ax.set_xlim(-50,1050)
# # ax.set_yticks(yticks)
# ax.set_ylabel('Y(meter)')
# ax.set_ylim(1050,-50)
# ax.set_zlabel('Z(meter)')
# ax.set_zlim(100,60)


# #视角初始化
# ax.view_init(-167, -79)
# # 取消背景网格线
# ax.grid(False)

# position=fig.add_axes([0.43, 0.20, 0.018, 0.35])#位置[左,下,右,上]
# v = np.linspace(0, 1.0, 6, endpoint=True)
# cb=plt.colorbar(c,cax=position,ticks=v,orientation='vertical')#方向
# cb.set_label(u'无 线 电 信 号 覆 盖 概 率',labelpad=16, rotation=270,fontproperties=font,fontsize=16)

# # ax.annotate3D('point 1', (100, 100, 65), xytext=(3, 3), textcoords='offset points',zorder=200)
# # ax.annotate3D('point 2', (200, 100, 65),
# #               xytext=(-30, -30),
# #               textcoords='offset points',
# #               arrowprops=dict(ec='black', fc='white', shrink=2.5),zorder=200)
# # ax.annotate3D('total steps:113,\ncumulative rewards:1643.59', (80, 200, 70),
# #               xytext=(-70, 205),
# #               textcoords='offset points',
# #               bbox=dict(boxstyle="round", fc="lightyellow"),
# #               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),zorder=200)

# # ax.annotate3D('total steps:93,\ncumulative rewards:1419.77', (770, 520, 70),
# #               xytext=(50, 175),
# #               textcoords='offset points',
# #               bbox=dict(boxstyle="round", fc="lightyellow"),
# #               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),zorder=200)

# ax.legend(loc='upper center',fontsize=16)#["SNARM_26_DIRECTIONS", "D3QN_26_DIRECTIONS"],

# # cbar=fig.colorbar(c,ticks=v,shrink=0.5, aspect=10)

# plt.tight_layout()

# # -----------------------------------------第二张图-------------------------------------------------
# # fig.subplots_adjust(wspace = .1,hspace = 0)


# ax = fig.add_subplot(122, projection='3d')
# c=ax.contourf(np.array(X_vec_60)*10, np.array(Y_vec_60)*10, 1-OutageMapActual_60m, zdir='z', offset=60, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_70)*10, np.array(Y_vec_70)*10, 1-OutageMapActual_70m, zdir='z', offset=70, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_80)*10, np.array(Y_vec_80)*10, 1-OutageMapActual_80m, zdir='z', offset=80, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_90)*10, np.array(Y_vec_90)*10, 1-OutageMapActual_90m, zdir='z', offset=90, cmap=newcmp,alpha=1)
# ax.contourf(np.array(X_vec_100)*10, np.array(Y_vec_100)*10, 1-OutageMapActual_100m, zdir='z', offset=100, cmap=newcmp,alpha=1)

# ax.text3D(100,100,102,'$q_s$',zdir='x',fontsize=20,zorder=100)
# ax.scatter3D(86,5,99,marker='^',color='black',zdir='z',s=300,zorder=200)

# ax.text3D(820,780,102,'$q_f$',zdir='x',fontsize=20,zorder=100)
# ax.scatter3D(800,800,102.5,marker='D',zdir='z',s=200,zorder=200) 

# S_episode_idx_14=12892
# D_episode_idx_14=6196

# S_seq_14=tra_14[S_episode_idx_14]

# S_seq_14=np.squeeze(np.asarray(S_seq_14))
# S_seq_14[0,0]=100
# S_seq_14[0,1]=100
# S_seq_14[0,2]=100
# S_seq_14[len(S_seq_14)-1,0]=800
# S_seq_14[len(S_seq_14)-1,1]=800
# S_seq_14[len(S_seq_14)-1,2]=100
# if S_seq_14.ndim == 2:  
#     ax.plot(S_seq_14[:,0],S_seq_14[:,1],S_seq_14[:,2],zdir='z',color='r',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=1,zorder=200,label='SNARM_14_DIRECTIONS')


# D_seq_14=D_tra_14[D_episode_idx_14]
# D_seq_14=np.squeeze(np.asarray(D_seq_14))
# D_seq_14[0,0]=100
# D_seq_14[0,1]=100
# D_seq_14[0,2]=100
# D_seq_14[len(D_seq_14)-1,0]=800
# D_seq_14[len(D_seq_14)-1,1]=800
# D_seq_14[len(D_seq_14)-1,2]=100

# if D_seq_14.ndim == 2:  
#     ax.plot(D_seq_14[:,0],D_seq_14[:,1],D_seq_14[:,2],zdir='z',color='b',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=0.5,zorder=200,label='D3QN_14_DIRECTIONS')



# ax.plot([100,800],[100,800],[100,100],zdir='z',color='g',linestyle=':',linewidth=2,
#     	marker='8',markevery=1,markersize=3,alpha=1,zorder=200,label='STRAIGHT_FLIGHT')


# xticks = [0,200,400,600,800,1000]
# yticks = [0,200,400,600,800,1000]

# # ax.set_xticks(xticks)
# ax.set_xlabel('X(meter)')
# ax.set_xlim(-50,1050)
# # ax.set_yticks(yticks)
# ax.set_ylabel('Y(meter)')
# ax.set_ylim(1050,-50)
# ax.set_zlabel('Z(meter)')
# ax.set_zlim(100,60)

# #视角初始化
# ax.view_init(-167, -79)


# # 取消背景网格线
# ax.grid(False)


# # ax.annotate3D('total steps:124,\ncumulative rewards: 1631.14', (70, 200, 70),
# #               xytext=(-70, 205),
# #               textcoords='offset points',
# #               bbox=dict(boxstyle="round", fc="lightyellow"),
# #               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),zorder=200)

# # ax.annotate3D('total steps:132,\ncumulative rewards: 1406.99', (750, 520, 70),
# #               xytext=(50, 175),
# #               textcoords='offset points',
# #               bbox=dict(boxstyle="round", fc="lightyellow"),
# #               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2),zorder=200)


# ax.legend(loc='upper center',fontsize=16)#["SNARM_14_DIRECTIONS", "D3QN_14_DIRECTIONS"], loc=1
# # fig.subplots_adjust(bottom=0, right=0.1, top=0.1)


# # v = np.linspace(0, 1.0, 6, endpoint=True)
# # cbar=fig.colorbar(c,ticks=v,shrink=0.5, aspect=10)

# position=fig.add_axes([0.93, 0.2, 0.018, 0.35])#位置[左,下,右,上]
# v = np.linspace(0, 1.0, 6, endpoint=True)
# cb=plt.colorbar(c,cax=position,ticks=v,orientation='vertical')#方向
# cb.set_label(u'无 线 电 信 号 覆 盖 概 率',labelpad=16, rotation=270,fontproperties=font,fontsize=16)

# plt.tight_layout()

# plt.show()


# fig.savefig('3D_100m_pics.jpg')
# #----------------------------------结束----------------------------------




# # ------------------------------------------将多个2D图嵌入3D------------------------------------、
# # 通过设置zdir='y', 参数来设置图形嵌入位置，zs参数设置嵌入的坐标位置
# import matplotlib.pyplot as plt
# import numpy as np

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# colors = ['r', 'g', 'b', 'y']
# yticks = [3, 2, 1, 0]
# for c, k in zip(colors, yticks):
#     # Generate the random data for the y=k 'layer'.
#     xs = np.arange(20)
#     ys = np.random.rand(20)

#     # You can provide either a single color or an array with the same length as
#     # xs and ys. To demonstrate this, we color the first bar of each set cyan.
#     cs = [c] * len(xs)
#     cs[0] = 'c'

#     # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
#     ax.bar(xs, ys, zs=k, zdir='z', color=cs, alpha=0.8)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # On the y axis let's only label the discrete values that we have data for.
# ax.set_yticks(yticks)

# plt.show()
# # ------------------------------------------将多个2D图嵌入3D------------------------------------













#-----------------------------------画3D图-----------------------------------------------
# 



# # 这一部分模拟了建筑物的分布
# # This part model the distribution of buildings
# ALPHA = 0.3
# BETA = 180
# GAMA = 20  # 原50
# MAXHeight = 50  # 原90

# SIR_THRESHOLD = 0  # 原0 SIR阈值用于中断 SIR threshold in dB for outage

# # ==========================================
# # 模拟建筑位置和建筑大小,每座建筑都以正方形为模型
# # ==Simulate the building locations and building size. Each building is modeled by a square
# D = 1  # 原2 in km, consider the area of DxD km^2
# N = BETA * (D ** 2)  # 建筑总数 the total number of buildings
# A = ALPHA * (D ** 2) / N  # 每座建筑的预期大小 the expected size of each building
# Side = np.sqrt(A) #side = 0.1

# H_vec = np.random.rayleigh(GAMA, int(N))
# H_vec = [min(x, MAXHeight) for x in H_vec]

# # 建筑网格分布 Grid distribution of buildings
# Cluster_per_side = 1
# Cluster = Cluster_per_side ** 2
# N_per_cluster = [np.ceil(N / Cluster) for i in range(Cluster)]

# # 添加一些修改，确保建筑总数为N
# # Add some modification to ensure that the total number of buildings is N
# Extra_building = int(np.sum(N_per_cluster) - N)
# N_per_cluster[:(Extra_building - 1)] = [np.ceil(N / Cluster) - 1 for i in range(Extra_building)]

# # ============================
# Road_width = 0.01  # 道路宽度(以公里为单位) road width in km
# Cluster_size = (D - (Cluster_per_side - 1) * Road_width) / Cluster_per_side
# Cluster_center = np.arange(Cluster_per_side) * (Cluster_size + Road_width) + Cluster_size / 2
# # 获取建筑位置
# # =====Get the building locations=================
# XLOC = [];
# YLOC = [];

# for i in range(Cluster_per_side):
#     for j in range(Cluster_per_side):
#         Idx = i * Cluster_per_side + j
#         Buildings = int(N_per_cluster[Idx])
#         Center_loc = [Cluster_center[i], Cluster_center[j]]
#         Building_per_row = int(np.ceil(np.sqrt(Buildings)))
#         Building_dist = (Cluster_size - 2 * Side) / (Building_per_row - 1)
#         X_loc = np.linspace((-Cluster_size + 2 * Side) / 2, (Cluster_size - 2 * Side) / 2, Building_per_row)
#         Loc_tempX = np.array(list(X_loc) * Building_per_row) + Center_loc[0]
#         Loc_tempY = np.repeat(list(X_loc), Building_per_row) + Center_loc[1]
#         XLOC.extend(list(Loc_tempX[0:Buildings]))
#         YLOC.extend(list(Loc_tempY[0:Buildings]))

# # 以建筑物高度为例 Sample the building Heights
# step = 101
# # 原101 包括起始点0和结束点，两个采样点间距为D/(step-1)
# # include the start point at 0 and end point, the space between two sample points is D/(step-1)
# HeightMapMatrix = np.zeros(shape=(int(D * step), int(D * step)))
# HeighMapArray = HeightMapMatrix.reshape(1, int((D * step) ** 2))

# for i in range(N):
#     x1 = XLOC[i] - Side / 2
#     x2 = XLOC[i] + Side / 2
#     y1 = YLOC[i] - Side / 2
#     y2 = YLOC[i] + Side / 2
#     HeightMapMatrix[int(np.ceil(x1 * step) - 1):int(np.floor(x2 * step)),
#     int(np.ceil(y1 * step) - 1):int(np.floor(y2 * step))] = H_vec[i]




# fig = plt.figure()
# ax=fig.add_subplot(111, projection='3d') # 将这块画布分为1×1，然后第三个1对应的就是1号区,相当于铺满
# X, Y = np.squeeze(np.array(XLOC)*1000), np.squeeze(np.array(YLOC)*1000)
# X, Y = X.ravel(), Y.ravel()  # 矩阵扁平化
# bottom = np.zeros_like(H_vec)  # 设置柱状图的底端位值
# Z = H_vec
# c = ['w'] * len(Z)
# width = height = Side

# print(Side)
# ax.bar3d(X, Y, bottom, width*1000, height*1000, Z,color=c,shade=False,alpha=0.3,edgecolor='lightslategrey',linewidth=1.1)
# ax.set_xlim(0, D*1000)
# ax.set_ylim(0, D*1000)
# ax.set_zlim(0, 50)
# my_x_ticks = np.arange(0, 1050, 200)
# my_y_ticks = np.arange(0, 1050, 200)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
# ax.set_xlabel('x(meter)')
# ax.set_ylabel('y(meter)')
# ax.set_zlabel('z(meter)')

# ax.grid(False)
# ax.view_init(elev=17, azim=250)
# # fig.savefig('build_3D.eps')
# fig.savefig('build_3D.pdf')
# fig.savefig('build_3D.jpg')
# np.savez('3D_build', X, Y, bottom, width, height, Z)
# plt.show()





# 