# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:36:06 2019

@author: Xiaoli Xu and Yong Zeng
At each UAV location, get the empirical outage probability based on the measured signal strengths
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib.colors import ListedColormap,LinearSegmentedColormap

import time

print('Generate radio environment......')
# 这一部分模拟了建筑物的分布
# This part model the distribution of buildings
ALPHA = 0.3
BETA = 320 # 原300
GAMA = 20  # 原50
MAXHeight = 50  # 原90

SIR_THRESHOLD = 0  # 原0 SIR阈值用于中断 SIR threshold in dB for outage

# ==========================================
# 模拟建筑位置和建筑大小,每座建筑都以正方形为模型
# ==Simulate the building locations and building size. Each building is modeled by a square
D = 1  # 原2 in km, consider the area of DxD km^2
N = BETA * (D ** 2)  # 建筑总数 the total number of buildings
A = ALPHA * (D ** 2) / N  # 每座建筑的预期大小 the expected size of each building
Side = np.sqrt(A) #side = 0.1

H_vec = np.random.rayleigh(GAMA, N)
H_vec = [min(x, MAXHeight) for x in H_vec]

# 建筑网格分布 Grid distribution of buildings
Cluster_per_side = 3
Cluster = Cluster_per_side ** 2
N_per_cluster = [np.ceil(N / Cluster) for i in range(Cluster)]

# 添加一些修改，确保建筑总数为N
# Add some modification to ensure that the total number of buildings is N
Extra_building = int(np.sum(N_per_cluster) - N)
N_per_cluster[:(Extra_building - 1)] = [np.ceil(N / Cluster) - 1 for i in range(Extra_building)]

# ============================
Road_width = 0.02  # 原0.02 道路宽度(以公里为单位) road width in km
Cluster_size = (D - (Cluster_per_side - 1) * Road_width) / Cluster_per_side
Cluster_center = np.arange(Cluster_per_side) * (Cluster_size + Road_width) + Cluster_size / 2
# 获取建筑位置
# =====Get the building locations=================
XLOC = [];
YLOC = [];

for i in range(Cluster_per_side):
    for j in range(Cluster_per_side):
        Idx = i * Cluster_per_side + j
        Buildings = int(N_per_cluster[Idx])
        Center_loc = [Cluster_center[i], Cluster_center[j]]
        Building_per_row = int(np.ceil(np.sqrt(Buildings)))
        Building_dist = (Cluster_size - 2 * Side) / (Building_per_row - 1)
        X_loc = np.linspace((-Cluster_size + 2 * Side) / 2, (Cluster_size - 2 * Side) / 2, Building_per_row)
        Loc_tempX = np.array(list(X_loc) * Building_per_row) + Center_loc[0]
        Loc_tempY = np.repeat(list(X_loc), Building_per_row) + Center_loc[1]
        XLOC.extend(list(Loc_tempX[0:Buildings]))
        YLOC.extend(list(Loc_tempY[0:Buildings]))

# 以建筑物高度为例 Sample the building Heights
step = 101
# 原101 包括起始点0和结束点，两个采样点间距为D/(step-1)
# include the start point at 0 and end point, the space between two sample points is D/(step-1)
HeightMapMatrix = np.zeros(shape=(int(D * step), int(D * step)))
HeighMapArray = HeightMapMatrix.reshape(1, int((D * step) ** 2))

for i in range(N):
    x1 = XLOC[i] - Side / 2
    x2 = XLOC[i] + Side / 2
    y1 = YLOC[i] - Side / 2
    y2 = YLOC[i] + Side / 2
    HeightMapMatrix[int(np.ceil(x1 * step) - 1):int(np.floor(x2 * step)),
    int(np.ceil(y1 * step) - 1):int(np.floor(y2 * step))] = H_vec[i]

# 建筑分布结束
# =================END of Building distributions================================

# 定义基站分布
# =============Define the BS Distribution=======================
# 基站分布

# 五角星型基站分布
# (R * cos(90°+ k * 72°+ yDegree)), (R * sin(90°+ k * 72°+ yDegree))
# 其中k = 0、1、2、3、4， yDegree为oa与y轴的夹角（如下图），默认为0。

# sin 18° = 0.30901699437495
# cos 18° = 0.95105651629515
# 0.6*sin18° = 0.18541019662497
# 0.6*cos18° = 0.57063390977709

# sin 36° =  0.58778525229247
# cos 36° = 0.80901699437495
# 0.6*sin36° = 0.35267115137548 
# 0.6*cos36° = 0.48541019662497

BS_loc = np.array(
    [
    #[y,x,z]格式
    # 原7个基站 正六边形排列
     # [1, 1, 0.025], [1.5774, 1.333, 0.025],
     # [1, 1.6667, 0.025, ], [0.4226, 1.3333, 0.025], [0.4226, 0.6667, 0.025],
     # [1, 0.3333, 0.025], [1.5774, 0.6667, 0.025]

    # 6个基站 正五边形排列 基站高度为25米
        # [0.5,0.5,0.015],[1.1854/2,1.57063/2,0.015],
        # [0.8,0.5,0.015],[1.1854/2,0.42947/2,0.015],[0.51459/2,0.64733/2,0.015],
        # [0.51459/2,1.35267/2,0.015],

    # 简单路径
    [0.15,0.1,0.015],[0.85,0.2,0.015],[0.85,0.75,0.015],[0.3,0.9,0.015]

    ])

# BS_Height=25 #BS height in meters convert to location in km
BS_thetaD = 100  # 原100 下倾角度[0,180] The downtile angle in degree [0, 180]
PB = 0.1  #
Fc = 2  # 工作频率(单位:GHzBS)发射功率(单位:瓦特 Operating Frequency in GHzBS Transmit power in Watt
LightSpeed = 3 * (10 ** 8)
WaveLength = LightSpeed / (Fc * (10 ** 9))  # 波长 wavelength in meter
SectorBoreSiteAngle = [-120, 0, 120]  # 每个基站的扇形角 the sector angle for each BS
Sec_num = np.size(SectorBoreSiteAngle)  # 每个单元的扇区数 the number of sectors per cell
FastFadingSampleSize = 1000  # 每个时间步长的信号测量数 number of signal measurements per time step

# ===========视图构建和基站分发 View Building and BS distributions
fig = plt.figure(2)

ax=fig.add_subplot(111)

ax.set_xlabel('x(meter)')
ax.set_ylabel('y(meter)')

for i in range(N):
    x1 = XLOC[i] - Side / 2
    x2 = XLOC[i] + Side / 2
    y1 = YLOC[i] - Side / 2
    y2 = YLOC[i] + Side / 2
    XList = [x1*1000, x2*1000, x2*1000, x1*1000, x1*1000]
    YList = [y1*1000, y1*1000, y2*1000, y2*1000, y1*1000]
    plt.plot(XList, YList, 'r-',markersize=5)

my_x_ticks = [0,200,400,600,800,1000]#np.arange(0, 1000, 200)
my_y_ticks = [0,200,400,600,800,1000]#np.arange(0, 1000, 200)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.plot(BS_loc[:, 1] *1000, BS_loc[:, 0] *1000, '*', markersize=20) #  基站坐标显示
fig.savefig('build_2D.jpg')
plt.show()


fig = plt.figure(3)
ax=fig.add_subplot(111, projection='3d')
X, Y = np.squeeze(np.array(XLOC)*1000), np.squeeze(np.array(YLOC)*1000)
X, Y = X.ravel(), Y.ravel()  # 矩阵扁平化
bottom = np.zeros_like(H_vec)  # 设置柱状图的底端位值
Z = H_vec
c = ['w'] * len(Z)
width = height = Side
ax.bar3d(X, Y, bottom, width*1000, height*1000, Z,color=c,shade=False,alpha=0.3,edgecolor='lightslategrey',linewidth=1.1)
ax.set_xlim(0, D*1000)
ax.set_ylim(0, D*1000)
ax.set_zlim(0, 50)
my_x_ticks = np.arange(0, 1050, 200)
my_y_ticks = np.arange(0, 1050, 200)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
ax.set_xlabel('x(meter)')
ax.set_ylabel('y(meter)')
ax.set_zlabel('z(meter)')
ax.view_init(elev=17, azim=250)
# fig.savefig('build_3D.eps')
# fig.savefig('build_3D.pdf')
ax.grid(False)
fig.savefig('build_3D.jpg')
np.savez('3D_build', X, Y, bottom, width, height, Z)

# 取消背景网格线

plt.show()

# ==============================================================================
# 为了加快速度，我们将不同角度的天线增益预存储到一个矩阵中
# To speed up, we pre-store the atenna gain from different angles into a matrix
def getAntennaGain(Theta_deg, Phi_deg):
    # 天线阵列基本设置
    # Basic Setting about Antenna Arrays
    ArrayElement_Horizontal = 1  # 水平方向的数组元素数 number of array elements in horizontal
    ArrayElement_Vertical = 8
    DV = 0.5 * WaveLength  # 垂直阵列间距 spacing for vertical array
    DH = 0.5 * WaveLength  # 水平阵列间距 spacing for horizontal array
    angleTiltV = BS_thetaD
    angleTiltH = 0  # 水平倾斜角度 horizontal tilt angle
    # 找出元素的功率增益 Find the element power gain
    angle3dB = 65
    Am = 30
    AH = -np.min([12 * (Phi_deg / angle3dB) ** 2, Am])  # 水平方向的元件功率增益 element power gain in horizontal
    AV = -np.min([12 * ((Theta_deg - 90) / angle3dB) ** 2, Am])  # 垂直方向的功率增益 element power gain in Vertical
    Gemax = 8  # 在各向同性辐射器上方的天线增益(分贝)，天线元件的最大方向性增益
    # dBi antenna gain in dB above an isotropic radiator, Maximum directional gain of an antenna element
    Aelement = -np.min([-(AH + AV), Am])
    GelementdB = Gemax + Aelement  # dBi
    Gelement = 10 ** (GelementdB / 10)
    Felement = np.sqrt(Gelement)
    # Find array gain
    k = 2 * np.pi / WaveLength  # wave number
    kVector = k * np.array([np.sin(Theta_deg / 180 * np.pi) * np.cos(Phi_deg / 180 * np.pi),
                            np.sin(Theta_deg / 180 * np.pi) * np.sin(Phi_deg / 180 * np.pi),
                            np.cos(Theta_deg / 180 * np.pi)])  # wave vector
    rMatrix = np.zeros(shape=(ArrayElement_Horizontal * ArrayElement_Vertical, 3))
    for n in range(ArrayElement_Horizontal):
        rMatrix[(n + 1) * np.arange(ArrayElement_Vertical), 2] = np.arange(ArrayElement_Vertical) * DV
        rMatrix[(n + 1) * np.arange(ArrayElement_Vertical), 1] = n * DH
    SteeringVector = np.exp(-1j * (rMatrix.dot(np.transpose(kVector))))
    # 垂直的权向量 Vertical Weight Vector
    Weight_Vertical = (1 / np.sqrt(ArrayElement_Vertical)) * np.exp(
        -1j * k * np.arange(ArrayElement_Vertical) * DV * np.cos(angleTiltV / 180 * np.pi))
    Weight_Horizontal = (1 / np.sqrt(ArrayElement_Horizontal)) * np.exp(
        -1j * k * np.arange(ArrayElement_Horizontal) * DH * np.sin(angleTiltH / 180 * np.pi))
    Weight2D = np.kron(Weight_Horizontal, np.transpose(Weight_Vertical))
    WeightFlatten = Weight2D.reshape(1, ArrayElement_Vertical * ArrayElement_Horizontal)
    ArrayFactor = np.conjugate(WeightFlatten).dot(SteeringVector.reshape(ArrayElement_Vertical, 1))
    Farray = Felement * ArrayFactor
    Garray = (np.abs(Farray)) ** 2
    return 10 * np.log10(Garray), Farray


# 获得所有天线增益
# ================get All the Antenna gain====================
angleVvector = np.arange(181)  # 垂直角度从0到180 vertical angle from 0 to 180
angleHvector = np.linspace(-180, 179, 360)
numV = np.size(angleVvector)
numH = np.size(angleHvector)
GarraydBmatrix = np.zeros(shape=(numV, numH))  # 预先存储天线增益 pre-stored atenna gain
FtxMatrix = np.zeros(shape=(numV, numH), dtype=complex)  # 预先存储阵列的因素 pre-stored array factor
for p in range(numV):
    for q in range(numH):
        GarraydBmatrix[p, q], FtxMatrix[p, q] = getAntennaGain(angleVvector[p], angleHvector[q])


# ==============================================================================
# 主要功能，确定从所有BS在给定位置的最佳中断，输出某点的最小中断
# =========Main Function that determines the best outage from all BS at a given location=======
#  Loc_vec:一个矩阵，nx3，每一行是一个(x,y,z)位置
#  loc_vec: a matrix, nx3, each row is a (x,y,z) location
# SIR_th:用于确定中断的SIR阈值
# SIR_th: the SIR threshold for determining outage
def getPointMiniOutage(loc_vec):
    numLoc = len(loc_vec)
    Out_vec = []
    for i in range(numLoc):
        PointLoc = loc_vec[i, :]
        OutageMatrix = getPointOutageMatrix(PointLoc, SIR_THRESHOLD)
        MiniOutage = np.min(OutageMatrix)
        Out_vec.append(MiniOutage)  

        # Val_instan_rate=np.max(OutageMatrix)
        # if Val_instan_rate<=6:
        #     Val_instan_rate=0
        # else:
        #     Val_instan_rate=1
        # if (PointLoc[0] ==0.36) and (PointLoc[1]==1.18):
        #     Val_instan_rate = 1
        # if (PointLoc[0] ==0.4) and (PointLoc[1]==1.34):
        #     Val_instan_rate = 1
        # if np.sqrt((PointLoc[0]-BS_loc[0][0])**2+(PointLoc[1]-BS_loc[0][1])**2)<=0.08:
        #     Val_instan_rate=1
        # if np.sqrt((PointLoc[0]-BS_loc[1][0])**2+(PointLoc[1]-BS_loc[1][1])**2)<=0.08:
        #     Val_instan_rate=1
        # Out_vec.append(Val_instan_rate)

    return Out_vec


# 对于给定的位置，返回所有BSs的所有部分的经验中断概率
# For a given location, return the empirical outage probaibility from all sectors of all BSs
# PointLoc:给定的点位置 PointLoc:  the given point location
# SIR_th:定义中断的SIR阈值 SIR_th: the SIR threshold for defining the outage
# outagmatrix:连接到每个站点的平均中断概率，通过对所有样本进行平均得到
# OutageMatrix: The average outage probability for connecting with each site, obtained by averaging over all the samples
def getPointOutageMatrix(PointLoc, SIR_th):
    numBS = len(BS_loc)
    SignalFromBS = []
    TotalPower = 0
    for i in range(len(BS_loc)):
        BS = BS_loc[i, :]
        LoS = checkLoS(PointLoc, BS)
        MeasuredSignal = getReceivedPower_RicianAndRayleighFastFading(PointLoc, BS, LoS)
        SignalFromBS.append(MeasuredSignal)
        TotalPower = TotalPower + MeasuredSignal
    TotalPowerAllSector = np.sum(TotalPower, axis=1)  # 所有力量的干扰 the interference of all power
    OutageMatrix = np.zeros(shape=(numBS, Sec_num))
    for i in range(len(BS_loc)):
        SignalFromThisBS = SignalFromBS[i]
        for sector in range(Sec_num):
            SignalFromThisSector = SignalFromThisBS[:, sector]
            SIR = SignalFromThisSector / (TotalPowerAllSector - SignalFromThisSector ) # add +1*10**(-13)
            SIR_dB = 10 * np.log10(SIR)
            OutageMatrix[i, sector] = np.sum(SIR_dB < SIR_th) / len(SIR_dB)
    return OutageMatrix


# 返回一个位置接收到的功率是来自所有三个部分的BS总和
# Return the received power at a location from all the three sectors of a BS
# 虽然在给定的位置和地点，大尺度路径损耗功率是一个常数，但快速衰落可能变化非常快
# While the large scale path loss power is a constant for given location and site, the fast fading may change very fast.
# 因此，我们返回多个快速衰落系数。样本的数量由FastFadingSampleSize决定
# Hence, we return multiple fast fading coefficients. The number of samples is determined by FastFadingSampleSize
# 一个简单的快衰落实现:如果是LoS，用K因子15db的Rician衰落;否则,瑞利衰落
# A simple fast-fading implementation: if LoS, Rician fading with K factor 15 dB; otherwise, Rayleigh fading
def getReceivedPower_RicianAndRayleighFastFading(PointLoc, BS, LoS):
    HorizonDistance = np.sqrt((BS[0] - PointLoc[0]) ** 2 + (BS[1] - PointLoc[1]) ** 2)

    Theta = np.arctan((BS[2] - PointLoc[2]) / (HorizonDistance + 0.00001))  # 倾斜角 elevation angle

    Theta_deg = np.rad2deg(Theta) + 90  # convert to the (0,180) degree
    if (PointLoc[1] == BS[1]) & (PointLoc[0] == BS[0]):
        Phi = 0
    else:
        Phi = np.arctan((PointLoc[1] - BS[1]) / (PointLoc[0] - BS[0] + 0.00001))  # to avoid dividing by 0
    Phi_deg = np.rad2deg(Phi)
    # 将水平度转换为范围(-180,180)
    # Convert the horizontal degree to the range (-180,180)
    if (PointLoc[1] > BS[1]) & (PointLoc[0] < BS[0]):
        Phi_deg = Phi_deg + 180
    elif (PointLoc[1] < BS[1]) & (PointLoc[0] < BS[0]):
        Phi_deg = Phi_deg - 180
    LargeScale = getLargeScalePowerFromBS(PointLoc, BS, Theta_deg, Phi_deg,
                                          LoS)  # 基于路径损耗的大规模接收功率 large-scale received power based on path loss

    # 随机分量，也就是瑞利衰落 the random component, which is Rayleigh fading
    RayleighComponent = np.sqrt(0.5) * (
            np.random.randn(FastFadingSampleSize, 3) + 1j * np.random.randn(FastFadingSampleSize, 3))

    if LoS:  # LoS，快速衰落由K因子15db的Rician衰落给出 LoS, fast fading is given by Rician fading with K factor 15 dB
        K_R_dB = 15  # Rician K factor in dB
        K_R = 10 ** (K_R_dB / 10)
        threeD_distance = 1000 * np.sqrt((BS[0] - PointLoc[0]) ** 2 + (BS[1] - PointLoc[1]) ** 2 + (
                BS[2] - PointLoc[2]) ** 2)  # 3D distance in meter
        DetermComponent = np.exp(-1j * 2 * np.pi * threeD_distance / WaveLength)  # deterministic component
        AllFastFadingCoef = np.sqrt(K_R / (K_R + 1)) * DetermComponent + np.sqrt(1 / (K_R + 1)) * RayleighComponent
    else:  # NLoS，快速衰落是瑞利衰落 NLoS, fast fading is Rayleigh fading
        AllFastFadingCoef = RayleighComponent

    h_overall = AllFastFadingCoef * np.sqrt(np.tile(LargeScale, (FastFadingSampleSize, 1)))
    PowerInstant = np.abs(h_overall) ** 2  # 瞬时接收到的瓦特功率 the instantneous received power in Watt
    return PowerInstant


# 这个函数检查BS和给定Loc之间是否有LoS
# This function check whether there is LoS between the BS and the given Loc
def checkLoS(PointLoc, BS):
    SamplePoints = np.linspace(0, 1, 100)
    XSample = BS[0] + SamplePoints * (PointLoc[0] - BS[0])
    YSample = BS[1] + SamplePoints * (PointLoc[1] - BS[1])
    ZSample = BS[2] + SamplePoints * (PointLoc[2] - BS[2])
    XRange = np.floor(XSample * (step - 1))
    YRange = np.floor(YSample * (step - 1))  #
    XRange = [max(x, 0) for x in XRange]  # 去掉负的指数 remove the negative idex
    YRange = [max(x, 0) for x in YRange]  # remove the negative idex
    Idx_vec = np.int_((np.array(XRange) * D * step + np.array(YRange)))
    SelectedHeight = [HeighMapArray[0, i] for i in Idx_vec]
    if any([x > y for (x, y) in zip(SelectedHeight, ZSample)]):
        return False
    else:
        return True


def getLargeScalePowerFromBS(PointLoc, BS, Theta_deg, Phi_deg, LoS):
    Sector_num = len(SectorBoreSiteAngle)
    Phi_Sector_ref = Phi_deg - np.array(SectorBoreSiteAngle)
    # 转换为相对扇面角的范围(- 180,180) Convert to the range (-180,180) with respect to the sector angle
    Phi_Sector_ref[Phi_Sector_ref < -180] = Phi_Sector_ref[Phi_Sector_ref < -180] + 360
    Phi_Sector_ref[Phi_Sector_ref > 180] = Phi_Sector_ref[Phi_Sector_ref > 180] - 360
    ChGain_dB = np.zeros(shape=(1, Sector_num))
    for i in range(Sector_num):
        ChGain_dB[0, i], null = getAntennaGain(Theta_deg, Phi_Sector_ref[i])
    ChGain = np.power(10, ChGain_dB / 10)  # 选择提供最大信道增益的扇区 choose the sector that provides the maximal channel gain
    Distance = 1000 * np.sqrt(
        (BS[0] - PointLoc[0]) ** 2 + (BS[1] - PointLoc[1]) ** 2 + (BS[2] - PointLoc[2]) ** 2)  # convert to meter
    # We use 3GPP TR36.777 Urban Macro Cell model to generate the path loss
    # UAV height between 22.5m and 300m
    if LoS:
        PathLoss_LoS_dB = 28 + 22 * np.log10(Distance) + 20 * np.log10(Fc)
        PathLoss_LoS_Linear = 10 ** (-PathLoss_LoS_dB / 10)
        Prx = ChGain * PB * PathLoss_LoS_Linear
    else:
        PathLoss_NLoS_dB = -17.5 + (46 - 7 * np.log10(PointLoc[2] * 1000)) * np.log10(Distance) + 20 * np.log10(
            40 * np.pi * Fc / 3)
        PathLoss_NLoS_Linear = 10 ** (-PathLoss_NLoS_dB / 10)
        Prx = ChGain * PB * PathLoss_NLoS_Linear
    return Prx


# 查看给定高度的无线电地图 100 90 80 70 60米
##============VIew the radio map for given height
UAV_height_100 = 0.1  # UAV height in km 
UAV_height_90 = 0.09
UAV_height_80 = 0.08
UAV_height_70 = 0.07
UAV_height_60 = 0.06

X_vec = range(D * (step - 1) + 1)
Y_vec = range(D * (step - 1) + 1)
numX, numY = np.size(X_vec), np.size(Y_vec)

# 地图实际中断概率

OutageMapActual_100m = np.zeros(shape=(numX, numY))
OutageMapActual_90m = np.zeros(shape=(numX, numY))
OutageMapActual_80m = np.zeros(shape=(numX, numY))
OutageMapActual_70m = np.zeros(shape=(numX, numY))
OutageMapActual_60m = np.zeros(shape=(numX, numY))

Loc_vec_All_100m = np.zeros(shape=(numX * numY, 3))
Loc_vec_All_90m = np.zeros(shape=(numX * numY, 3))
Loc_vec_All_80m = np.zeros(shape=(numX * numY, 3))
Loc_vec_All_70m = np.zeros(shape=(numX * numY, 3))
Loc_vec_All_60m = np.zeros(shape=(numX * numY, 3))

# ================== 耗时较长时间的代码段 ======================

for i in range(numX):

    Loc_vec_100m = np.zeros(shape=(numY, 3))
    Loc_vec_100m[:, 0] = X_vec[i] / step
    Loc_vec_100m[:, 1] = np.array(Y_vec) / step
    Loc_vec_100m[:, 2] = UAV_height_100

    Loc_vec_90m = np.zeros(shape=(numY, 3))
    Loc_vec_90m[:, 0] = X_vec[i] / step
    Loc_vec_90m[:, 1] = np.array(Y_vec) / step
    Loc_vec_90m[:, 2] = UAV_height_90

    Loc_vec_80m = np.zeros(shape=(numY, 3))
    Loc_vec_80m[:, 0] = X_vec[i] / step
    Loc_vec_80m[:, 1] = np.array(Y_vec) / step
    Loc_vec_80m[:, 2] = UAV_height_80

    Loc_vec_70m = np.zeros(shape=(numY, 3))
    Loc_vec_70m[:, 0] = X_vec[i] / step
    Loc_vec_70m[:, 1] = np.array(Y_vec) / step
    Loc_vec_70m[:, 2] = UAV_height_70

    Loc_vec_60m = np.zeros(shape=(numY, 3))
    Loc_vec_60m[:, 0] = X_vec[i] / step
    Loc_vec_60m[:, 1] = np.array(Y_vec) / step
    Loc_vec_60m[:, 2] = UAV_height_60

    Loc_vec_All_100m[i * numY:(i + 1) * numY, :] = Loc_vec_100m
    OutageMapActual_100m[i, :] = getPointMiniOutage(Loc_vec_100m)

    Loc_vec_All_90m[i * numY:(i + 1) * numY, :] = Loc_vec_90m
    OutageMapActual_90m[i, :] = getPointMiniOutage(Loc_vec_90m)

    Loc_vec_All_80m[i * numY:(i + 1) * numY, :] = Loc_vec_80m
    OutageMapActual_80m[i, :] = getPointMiniOutage(Loc_vec_80m)

    Loc_vec_All_70m[i * numY:(i + 1) * numY, :] = Loc_vec_70m
    OutageMapActual_70m[i, :] = getPointMiniOutage(Loc_vec_70m)

    Loc_vec_All_60m[i * numY:(i + 1) * numY, :] = Loc_vec_60m
    OutageMapActual_60m[i, :] = getPointMiniOutage(Loc_vec_60m)


# ================== 耗时较长时间的代码段 ======================


Outage_vec_All_100m = np.reshape(OutageMapActual_100m, numX * numY)
Outage_vec_All_90m = np.reshape(OutageMapActual_90m, numX * numY)
Outage_vec_All_80m = np.reshape(OutageMapActual_80m, numX * numY)
Outage_vec_All_70m = np.reshape(OutageMapActual_70m, numX * numY)
Outage_vec_All_60m = np.reshape(OutageMapActual_60m, numX * numY)

Test_Size = int(numX * numY / 10)
test_indices = random.sample(range(numX * numY), Test_Size)

TEST_LOC_meter_100m = Loc_vec_All_100m[test_indices, :2] * 1000
TEST_LOC_ACTUAL_OUTAGE_100m = Outage_vec_All_100m[test_indices]

TEST_LOC_meter_90m = Loc_vec_All_90m[test_indices, :2] * 1000
TEST_LOC_ACTUAL_OUTAGE_90m = Outage_vec_All_90m[test_indices]

TEST_LOC_meter_80m = Loc_vec_All_80m[test_indices, :2] * 1000
TEST_LOC_ACTUAL_OUTAGE_80m = Outage_vec_All_80m[test_indices]

TEST_LOC_meter_70m = Loc_vec_All_70m[test_indices, :2] * 1000
TEST_LOC_ACTUAL_OUTAGE_70m = Outage_vec_All_70m[test_indices]

TEST_LOC_meter_60m = Loc_vec_All_60m[test_indices, :2] * 1000
TEST_LOC_ACTUAL_OUTAGE_60m = Outage_vec_All_60m[test_indices]

# N 表示插值后的颜色数目
clist=['cornflowerblue','honeydew','coral']
newcmp = LinearSegmentedColormap.from_list('chaos',clist,N=256)


fig = plt.figure(100)

plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual_100m,cmap=newcmp)
v = np.linspace(0, 1.0, 11, endpoint=True)
cbar = plt.colorbar(ticks=v)
cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
plt.xlabel('x (meter)', fontsize=14)
plt.ylabel('y (meter)', fontsize=14)
fig.savefig('CoverageMapTrue_100m.jpg')
np.savez('radioenvir_100m', OutageMapActual_100m, X_vec, Y_vec, TEST_LOC_meter_100m, TEST_LOC_ACTUAL_OUTAGE_100m)

fig = plt.figure(90)

plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual_90m,cmap=newcmp)
v = np.linspace(0, 1.0, 11, endpoint=True)
cbar = plt.colorbar(ticks=v)
cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
plt.xlabel('x (meter)', fontsize=14)
plt.ylabel('y (meter)', fontsize=14)
fig.savefig('CoverageMapTrue_90m.jpg')
np.savez('radioenvir_90m', OutageMapActual_90m, X_vec, Y_vec, TEST_LOC_meter_90m, TEST_LOC_ACTUAL_OUTAGE_90m)

fig = plt.figure(80)

plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual_80m,cmap=newcmp)
v = np.linspace(0, 1.0, 11, endpoint=True)
cbar = plt.colorbar(ticks=v)
cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
plt.xlabel('x (meter)', fontsize=14)
plt.ylabel('y (meter)', fontsize=14)
fig.savefig('CoverageMapTrue_80m.jpg')
np.savez('radioenvir_80m', OutageMapActual_80m, X_vec, Y_vec, TEST_LOC_meter_80m, TEST_LOC_ACTUAL_OUTAGE_80m)

fig = plt.figure(70)

plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual_70m,cmap=newcmp)
v = np.linspace(0, 1.0, 11, endpoint=True)
cbar = plt.colorbar(ticks=v)
cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
plt.xlabel('x (meter)', fontsize=14)
plt.ylabel('y (meter)', fontsize=14)
fig.savefig('CoverageMapTrue_70m.jpg')
np.savez('radioenvir_70m', OutageMapActual_70m, X_vec, Y_vec, TEST_LOC_meter_70m, TEST_LOC_ACTUAL_OUTAGE_70m)

fig = plt.figure(60)

plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual_60m,cmap=newcmp)
v = np.linspace(0, 1.0, 11, endpoint=True)
cbar = plt.colorbar(ticks=v)
cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
plt.xlabel('x (meter)', fontsize=14)
plt.ylabel('y (meter)', fontsize=14)
fig.savefig('CoverageMapTrue_60m.jpg')
np.savez('radioenvir_60m', OutageMapActual_60m, X_vec, Y_vec, TEST_LOC_meter_60m, TEST_LOC_ACTUAL_OUTAGE_60m)



