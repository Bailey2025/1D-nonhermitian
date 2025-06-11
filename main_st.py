# 利用相似变换，优化所有的更新和计算
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

import math
from scipy.signal import find_peaks
import torch.multiprocessing as mp
import os, time, sys, mkl
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
mkl.set_num_threads(1)
import platform

if platform.system() == 'Linux':
    plt.switch_backend('agg')


def comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def Omega(n1, n2, nd):  # 状态数
    total = 2 * comb(n1 - 1, nd - 1) * comb(n2 - 1, nd - 1)
    if nd <= n1 - 1:
        total += comb(n1 - 1, nd) * comb(n2 - 1, nd - 1)
    if nd <= n2 - 1:
        total += comb(n1 - 1, nd - 1) * comb(n2 - 1, nd)
    return total


def Omega_matrix(L, J):
    """
    n1=0,1,2,...,L
    nd = 0,1,2,...min(n1,L-n1)
    """
    if J == 0:
        combination = [comb(L, k) for k in range(L + 1)]
        return combination
    else:
        com = np.zeros((L + 1, int(L / 2) + 1))  # (n1,nd)
        for n1 in np.arange(L + 1):
            if min(n1, L - n1) == 0:
                nd = 0
                com[n1, nd] = 1
            else:
                for nd in np.arange(1, min(n1, L - n1) + 1):
                    com[n1, nd] = Omega(n1, L - n1, nd)
        return com


def compute_for_prob_nd(beta, varepsilon, J, L, combination, gs_approx):
    mask = (combination != 0)
    try:
        logweight1 = np.sum(np.log(1 + np.exp(-beta * varepsilon)), axis=1)  # (n1)
    except FloatingPointError:  # 捕捉溢出等浮点运算错误
        a = np.log(1 + np.exp(-beta * varepsilon))
        a[np.isinf(a)] = -beta * varepsilon[np.isinf(a)]  # 处理无穷大的情况
        logweight1 = np.sum(a, axis=1)
    if not (logweight1.imag < 1E-5).all():
        assert (logweight1.imag < 1E-5).all()
    logweight1 = logweight1.real

    logweight2 = beta * J * L - 4 * beta * J * np.arange(int(L / 2) + 1)
    a = np.repeat(logweight1[:, np.newaxis], int(L / 2) + 1, axis=1) \
        + np.repeat(logweight2[np.newaxis, :], L + 1, axis=0)  # (n1,nd)
    a = a * mask

    # t0 = time.time()
    p = np.zeros((L + 1, int(L / 2) + 1), dtype=complex)  # (n1,nd)
    for n1 in range(L + 1):
        nd_index = mask[n1, :]
        a_diff = a[:, :, np.newaxis] - a[np.newaxis, np.newaxis, n1, nd_index]  # (n1,nd,nd) 前两个维度是矩阵a，最后一个维度是批处理nd
        p[n1, nd_index] = combination[n1, nd_index] / \
                          np.sum(np.exp(np.clip(a_diff, a_min=None, a_max=600)) * combination[:, :, np.newaxis],
                                 axis=(0, 1))

    if not (p.imag < 1E-5).all():
        assert (p.imag < 1E-5).all()
    p = p.real

    if gs_approx:
        E1 = np.sum(np.where(varepsilon > 0, varepsilon, 0), axis=1)
    else:
        E1 = np.nansum(varepsilon / (np.exp(beta * varepsilon) + 1), axis=1)  # (n1)
    E2 = 4 * beta * J * np.arange(int(L / 2) + 1)  # (nd)
    E = mask * (np.repeat(E1[:, np.newaxis], int(L / 2) + 1, axis=1) \
                - np.repeat(E2[np.newaxis, :], L + 1, axis=0))  # (n1,nd)
    if not (E.imag < 1E-5).all():
        assert (E.imag < 1E-5).all()
    E = E.real
    Cv = (np.sum(E ** 2 * p) - np.sum(E * p) ** 2) * beta ** 2 / L
    return Cv, np.sum(p, axis=1)


def compute_for_prob(beta, varepsilon, J, L, combination, gs_approx):
    logweight1 = np.sum(np.log(1 + np.exp(-beta * varepsilon)), axis=1)  # (n1)
    if np.isinf(logweight1).any():
        a = np.log(1 + np.exp(-beta * varepsilon))
        a[np.isinf(a)] = -beta * varepsilon[np.isinf(a)]
        logweight1 = np.sum(a, axis=1)
    p = np.zeros((L + 1), dtype=complex)  # (n1)
    for n1 in range(L + 1):
        p[n1] = combination[n1] / np.sum(np.exp(logweight1 - logweight1[n1]) * combination)
    if np.isnan(p).any():
        p = np.nan_to_num(p, nan=0)

    if not (p.imag < 1E-5).all():
        assert (p.imag < 1E-5).all()
    p = p.real

    if gs_approx:
        E = np.sum(np.where(varepsilon.real < 0, varepsilon, 0), axis=1)
    else:
        E = np.nansum(varepsilon / (np.exp(beta * varepsilon) + 1), axis=1)  # (n1)

    E = E.real
    Cv = (np.sum(E ** 2 * p) - np.sum(E * p) ** 2) * beta ** 2 / L
    return Cv, p


def compute_vvcor(L, U, J, beta, combination, gs_approx=True):
    varepsilon = np.zeros((L + 1, L), dtype=complex)  # (configuration,n)
    vvcor = np.zeros((L + 1, int(L / 2) + 1))
    for n1 in range(L + 1):
        tR = (1 + U) ** (n1 / L) * (1 - U) ** (1 - n1 / L)
        tL = (1 - U) ** (n1 / L) * (1 + U) ** (1 - n1 / L)
        k = np.arange(L) * 2 * np.pi / L
        sp_energy = (tR + tL) * np.cos(k) + ((tR - tL) * np.sin(k)) * 1.j
        varepsilon[n1, :] = sp_energy
        "计算vop"
        v_eig = (tR + tL) * np.sin(k) - (tR - tL) * np.cos(k) * 1.j
        if gs_approx:
            index = sp_energy.real < 0
            "计算vvcor"
            kwarg = {'k': k[index], 'L': L}
            for r in np.arange(int(L / 2) + 1):
                i, i_, j, j_ = 0, 1, r, r + 1
                vv = -(f_ijkh(i, i_, j, j_, **kwarg) * tL ** 2
                       - f_ijkh(i, i_, j_, j, **kwarg) * tR * tL
                       - f_ijkh(i_, i, j, j_, **kwarg) * tR * tL
                       + f_ijkh(i_, i, j_, j, **kwarg) * tR ** 2)
                vvcor[n1, r] = np.abs(vv.real)

    if J == 0:
        Cv, p_beta = compute_for_prob(beta, varepsilon, J, L, combination, gs_approx)
    else:
        Cv, p_beta = compute_for_prob_nd(beta, varepsilon, J, L, combination, gs_approx)

    if gs_approx:
        vv_exp_beta = np.sum(vvcor * p_beta[:, np.newaxis], axis=0)
    else:
        vvcor = np.zeros((L + 1, int(L / 2) + 1))
        for n1 in range(L + 1):
            tR = (1 + U) ** (n1 / L) * (1 - U) ** (1 - n1 / L)
            tL = (1 - U) ** (n1 / L) * (1 + U) ** (1 - n1 / L)
            k = np.arange(L) * 2 * np.pi / L
            kwarg = {'k': k, 'L': L, 'n_k': 1 / (np.exp(beta * varepsilon[n1, :]) + 1)}
            for r in range(int(L / 2) + 1):
                i, i_, j, j_ = 0, 1, r, r + 1
                vv = -(f_ijkh(i, i_, j, j_, **kwarg) * tL ** 2
                       - f_ijkh(i, i_, j_, j, **kwarg) * tR * tL
                       - f_ijkh(i_, i, j, j_, **kwarg) * tR * tL
                       + f_ijkh(i_, i, j_, j, **kwarg) * tR ** 2)
                vvcor[n1, r] = np.abs(vv.real)
        vv_exp_beta = np.sum(vvcor * p_beta[:, np.newaxis], axis=0)
    return vv_exp_beta


def compute_physical_quantity(beta, U, J, L, varepsilon, combination, X, Xcor, vare_v, vvcor, gs_approx,compute_v):
    """在确定的beta下计算各个X的概率，和总的物理量期望值"""
    if J == 0:
        Cv, p_beta = compute_for_prob(beta, varepsilon, J, L, combination, gs_approx)
    else:
        Cv, p_beta = compute_for_prob_nd(beta, varepsilon, J, L, combination, gs_approx)
    Cv_exp_beta = Cv
    X_exp_beta = np.sum(X * p_beta)
    Xcor_exp_beta = np.sum(Xcor * p_beta)

    if compute_v:  # 计算fermion相关量
        if gs_approx:
            v_dis_beta = np.sum(np.where(varepsilon.real < 0, vare_v, 0), axis=1)
            assert (v_dis_beta.real < 1E-5).all()
            v_dis_beta = v_dis_beta.imag
            vv_exp_beta = np.sum(vvcor * p_beta)
        else:
            v_dis_beta = np.nansum(vare_v / (np.exp(beta * varepsilon) + 1), axis=1)  # (L+1)
            assert (v_dis_beta.real < 1E-5).all()
            v_dis_beta = v_dis_beta.imag
            vv_exp_beta = np.zeros(L + 1)

        return Cv_exp_beta, X_exp_beta, Xcor_exp_beta, v_dis_beta, vv_exp_beta, p_beta
    else:
        return Cv_exp_beta, X_exp_beta, Xcor_exp_beta, None, None, p_beta


def fM(delta_r, **kwargs):
    k = kwargs.get('k', None)
    n_k = kwargs.get('n_k', None)
    L = kwargs.get('L', None)
    if n_k is None:
        return np.sum(np.exp(-1.j * k * delta_r)) / L
    else:
        return np.sum(np.exp(-1.j * k * delta_r) * n_k) / L


def f_ijkh(i, j, l, h, **kwargs):
    # 这个这个函数是  c_i^{\dagger} c_j c_k^{\dagger} c_h的期望值
    item = 1 if j == l else 0
    return fM(j - i, **kwargs) * fM(h - l, **kwargs) - fM(h - i, **kwargs) * fM(j - l, **kwargs) + fM(h - i,
                                                                                                      **kwargs) * item


def main_st(L, U, J, beta_list, combination, gs_approx=True, compute_v=False,
            imag_mu=0,phi=0,h=0):
    """
    相似变换方式精确计算，只用于nnn=0,dimension=1,pbc
    :param beta_list: 要计算的beta，对于每个beta并行计算
    :param combination: 组合数
    :param gs_approx: 是否基态近似
    :return: Cv_exp (beta_num) 每个beta下的Cv
             X_exp (beta_num) 每个beta下的X
             Xcor_exp (beta_num) 每个beta下的Xcor
             v_dis (beta_num,configuration) 每个beta下的不同configuration(n1)的v
             vv_exp (beta_num) 每个beta下的vv的期望
             p (beta_num,configuration) 每个beta下的不同configuration(n1)的v
    """

    # 设置浮点数异常行为，使得溢出时抛出异常
    np.seterr(over='ignore')

    # t0 = time.time()
    varepsilon = np.zeros((L + 1, L), dtype=complex)  # (configuration,n)
    X = np.zeros(L + 1)
    Xcor = np.zeros(L + 1)
    vare_v = []  # (configuration) if gs_approx else (configuration,n)
    vvcor = np.zeros(L + 1)
    for n1 in range(L + 1):
        tR = (1 + U) ** (n1 / L) * (1 - U) ** (1 - n1 / L)
        tL = (1 - U) ** (n1 / L) * (1 + U) ** (1 - n1 / L)
        # print(tR-tL)
        k = np.arange(L) * 2 * np.pi / L
        sp_energy = (tR + tL) * np.cos(k+phi) + ((tR - tL) * np.sin(k+phi)+imag_mu) * 1.j
        varepsilon[n1, :] = sp_energy
        "计算Xop"
        X[n1] = np.abs(2 * n1 / L - 1)
        Xcor[n1] = (2 * n1 / L - 1) ** 2
        "计算v相关量"
        if compute_v:
            "计算vop"
            v_eig = (tR + tL) * np.sin(k) - (tR - tL) * np.cos(k) * 1.j
            vare_v.append(v_eig)

    vare_v = np.array(vare_v)

    args = [(beta, U, J, L, varepsilon, combination, X, Xcor, vare_v, vvcor, gs_approx,compute_v) for beta in beta_list]
    Cv_exp = []
    X_exp = []
    Xcor_exp = []
    v_dis = []
    vv_exp = []
    p = []

    if len(args) == 1:
        Cv_exp, X_exp, Xcor_exp, v_dis, vv_exp, p = compute_physical_quantity(beta_list[0], U, J, L, varepsilon,
                                                                              combination, X,
                                                                              Xcor, vare_v, vvcor, gs_approx,compute_v)
    else:
        with mp.Pool() as pool:
            results = pool.starmap(compute_physical_quantity, args)

        # 将结果逐步解包并存储
        for result in results:
            Cv_exp_beta, X_exp_beta, Xcor_exp_beta, v_dis_beta, vv_exp_beta, p_beta = result
            Cv_exp.append(Cv_exp_beta)
            X_exp.append(X_exp_beta)
            Xcor_exp.append(Xcor_exp_beta)
            if vare_v.size != 0:
                v_dis.append(v_dis_beta)
                vv_exp.append(vv_exp_beta)
            p.append(p_beta)



    if vare_v.size != 0:
        return np.array(Cv_exp), np.array(X_exp), np.array(Xcor_exp), np.array(v_dis), np.array(vv_exp), np.array(p),
    else:
        return np.array(Cv_exp), np.array(X_exp), np.array(Xcor_exp), None, np.array(varepsilon), np.array(p),


if __name__ == '__main__':
    L = 30
    U = 0.4
    J = 0.0
    gs_approx = False
    compute_v = False
    beta_list = np.arange(0.1, 35, 1)
    combination = Omega_matrix(L, J)
    Cv_exp, X_exp, Xcor_exp, v, vv_exp, p = main_st(L, U, J, beta_list, combination, gs_approx, compute_v)

