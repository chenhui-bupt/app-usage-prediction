# -*- coding: utf-8 -*-
import numpy as np

usernum = 15314
appnum = 55
nodelist = []
userLE = None
appLE = None


def laplace_matrix(A):
    D = np.repeat(A.sum(axis=1), A.shape[1], axis=1)
    D = np.diag(np.diag(D))
    L = D - A
    return L


def ACT(A):
    L = laplace_matrix(A)
    pinvL = np.linalg.pinv(L)  # 伪逆矩阵
    Lxx = np.diag(pinvL)
    Lxx = np.repeat(Lxx.reshape(-1, 1), pinvL.shape[1], axis=1)  # 将对角线元素向量扩展为n×n阶矩阵,新矩阵每个列向量都是伪逆矩阵的对角线向量
    sim = 1.0/(Lxx+Lxx.T-2*pinvL)
    sim[np.isinf(sim)] = 0
    sim[np.isnan(sim)] = 0
    return sim[:usernum, usernum:]  # return useful matrix


def CosPlus(A):
    L = laplace_matrix(A)
    pinvL = np.linalg.pinv(L)  # 伪逆矩阵
    Lxx = np.diag(pinvL).reshape(-1, 1)  # 规成二维数组，便于叉乘
    sim = pinvL/((np.dot(Lxx, Lxx.T))**0.5)  # Lxx*Lxx.T此处为叉乘
    sim[np.isinf(sim)] = 0
    sim[np.isnan(sim)] = 0
    return sim[:usernum, usernum:]


def RWR(A, c):
    deg = np.repeat(A.sum(axis=1), A.shape[1], axis=1)
    P = A/deg  # 概率转移矩阵
    sim = (1-c) * np.linalg.inv(np.eye(P.shape[0]) - c*P.T)
    sim = sim + sim.T
    return sim[:usernum, usernum:]


def LRW(A, iters, c):
    deg = np.repeat(A.sum(axis=1), A.shape[1], axis=1)
    P = A/deg  # 概率转移矩阵
    I = np.eye(A.shape[0])
    sim = I
    k = 0
    while(k < iters):
        sim = (1-c)*I + c*P.T*sim  # 矩阵叉乘,注意
        k += 1
    sim = sim + sim.T
    return sim[:usernum, usernum:]


def MFI(A):
    L = laplace_matrix(A)
    sim=np.linalg.inv(np.eye(A.shape[0])+L)
    return sim[:usernum,usernum:]


def TSRWR(A, c):
    deg = np.repeat(A.sum(axis=1), A.shape[1], axis=1)
    P = A/deg  # 概率转移矩阵
    I = np.eye(A.shape[0])
    sim = (1-c)*np.linalg.inv(I - c*P.T)  # 还应该继续叉乘一单位阵，单位阵为初始向量，这里省略了，节约计算
    sim = sim + sim.T
    sim = np.dot(np.linalg.inv(I - c*sim), sim)
    return sim[:usernum, usernum:]

