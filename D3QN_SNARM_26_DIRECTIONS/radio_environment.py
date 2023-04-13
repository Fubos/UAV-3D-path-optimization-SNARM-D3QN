# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:36:06 2019

@author: Xiaoli Xu and Yong Zeng
At each UAV location, get the empirical outage probability based on the measured signal strengths
"""

import numpy as np
import matplotlib.pyplot as plt
import random

import time

#参数

ALPHA = 0.3
BETA = 320 # 原300
GAMA = 20  # 原50
MAXHeight = 50  # 原90

SIR_THRESHOLD = 0  # 原0 SIR阈值用于中断 SIR threshold in dB for outage

BS_thetaD = 100  # 原100 下倾角度[0,180] The downtile angle in degree [0, 180]
PB = 0.1  #
Fc = 2  # 工作频率(单位:GHzBS)发射功率(单位:瓦特 Operating Frequency in GHzBS Transmit power in Watt
LightSpeed = 3 * (10 ** 8)
WaveLength = LightSpeed / (Fc * (10 ** 9))  # 波长 wavelength in meter
SectorBoreSiteAngle = [-120, 0, 120]  # 每个基站的扇形角 the sector angle for each BS
Sec_num = np.size(SectorBoreSiteAngle)  # 每个单元的扇区数 the number of sectors per cell
FastFadingSampleSize = 1000  # 每个时间步长的信号测量数 number of signal measurements per time step



D = 1  # 原2 in km, consider the area of DxD km^2
N = BETA * (D ** 2)  # 建筑总数 the total number of buildings
A = ALPHA * (D ** 2) / N  # 每座建筑的预期大小 the expected size of each building
Side = np.sqrt(A) #side = 0.1

step = 101
HeightMapMatrix = np.zeros(shape=(int(D * step), int(D * step)))
HeighMapArray = HeightMapMatrix.reshape(1, int((D * step) ** 2))

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





BS_loc = np.array(
    [
    #[y,x,z]格式
    # 原7个基站 正六边形排列
     # [1, 1, 0.025], [1.5774, 1.333, 0.025],
     # [1, 1.6667, 0.025, ], [0.4226, 1.3333, 0.025], [0.4226, 0.6667, 0.025],
     # [1, 0.3333, 0.025], [1.5774, 0.6667, 0.025]

    # 6个基站 正五边形排列 基站高度为25米
   [0.15,0.1,0.015],[0.85,0.2,0.015],[0.85,0.75,0.015],[0.3,0.9,0.015]
   
    ])



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
