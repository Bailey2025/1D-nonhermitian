# 利用相似变换，优化所有的更新和计算
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp
from functools import partial
import os, time, sys, mkl
import math

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
mkl.set_num_threads(1)
import platform

if platform.system() == 'Linux':
    plt.switch_backend('agg')


def calculate_prob(logweight_, logweight):
    prob = np.exp(logweight_ - logweight) if logweight_ < logweight else 1
    try:
        assert not np.isinf(prob)
    except AssertionError:
        print("The weight contains inf value:", logweight, logweight_)
        print("prob inf")
        exit(0)
    return prob


def l2c(x, y, L):  # localization to coordinate
    return x + y * L


def c2l(i, L):
    x = i % L
    y = i // L
    return x, y


def pbc(x, L):
    return x % L



def mean_outlier(data):
    data = np.array(data)
    if np.std(data) == 0:
        return np.mean(data)
    # 计算 Z 分数
    z_scores = np.abs((data - np.mean(data)) / np.std(data))

    # 定义阈值，例如，如果 Z 分数超过 2，则被认为是极端值
    threshold = 2

    # 标识极端值的索引
    outliers = np.where(z_scores > threshold)

    # 移除极端值
    data_no_outliers = np.delete(data, outliers)

    # 计算去除极端值后的数组的平均值
    mean_no_outliers = np.mean(data_no_outliers)
    return mean_no_outliers


class MentoCarlo:
    def __init__(self, L, U, Beta=10, dimension=2, ispbc=True, isBond=True, nnn=None):
        self.L = L
        self.U = U
        self.Beta = Beta
        self.dimension = dimension
        self.ispbc = ispbc
        # self.epoch = epoch
        # self.filepath = filepath
        self.isBond = isBond
        self.nnn = nnn
        self.make_lattice()
        if nnn != 0:
            self.make_lattice_nn()

    def make_lattice(self):
        L = self.L
        if self.dimension == 1:
            self.N = L
            self.Nb = L if self.ispbc else L - 1
            self.bond_index = np.zeros((2, self.Nb), dtype=int)
            self.bond_index[0, :L - 1] = np.arange(L - 1)  # bond 左端
            self.bond_index[1, :L - 1] = np.arange(1, L)  # bond 右端
            if self.ispbc:
                self.bond_index[0, L - 1] = L - 1
                self.bond_index[1, L - 1] = 0
        elif self.dimension == 2:
            self.N = int(L * L)
            self.Nb = 2 * L ** 2 if self.ispbc else 2 * L * (L - 1)
            self.bond_index = np.zeros((2, self.Nb), dtype=int)
            if self.ispbc:
                x = np.tile(np.arange(L), L)
                y = np.arange(L).repeat(L)
                self.bond_index[0, :int(self.Nb / 2)] = l2c(x, y, L)  # xbond 左端
                self.bond_index[1, :int(self.Nb / 2)] = l2c(pbc(x + 1, L), y, L)  # xbond 右端
                self.bond_index[0, int(self.Nb / 2):] = l2c(x, y, L)  # ybond 下端
                self.bond_index[1, int(self.Nb / 2):] = l2c(x, pbc(y + 1, L), L)  # ybond 上端
            else:
                x = np.tile(np.arange(L - 1), L)
                y = np.arange(L).repeat(L - 1)
                self.bond_index[0, :int(self.Nb / 2)] = l2c(x, y, L)
                self.bond_index[1, :int(self.Nb / 2)] = l2c(x + 1, y, L)
                x = np.tile(np.arange(L), L - 1)
                y = np.arange(L - 1).repeat(L)
                self.bond_index[0, int(self.Nb / 2):] = l2c(x, y, L)
                self.bond_index[1, int(self.Nb / 2):] = l2c(x, y + 1, L)
        elif dimension == 1.5:  # 准1维pbc
            self.N = int(L * L)
            self.Nb = 2 * L ** 2 if self.ispbc else 2 * L ** 2 - L  # 准一维pbc
            x = np.tile(np.arange(L - 1), L)
            y = np.arange(L).repeat(L - 1)
            self.bond_index[0, :int(L * (L - 1))] = l2c(x, y, L)
            self.bond_index[1, :int(L * (L - 1))] = l2c(x + 1, y, L)
            x = np.tile(np.arange(L), L)
            y = np.arange(L).repeat(L)
            self.bond_index[0, int(L * (L - 1)):] = l2c(x, y, L)
            self.bond_index[1, int(L * (L - 1)):] = l2c(x, pbc(y + 1, L), L)
        else:
            print("dimension tbc")

    def make_lattice_nn(self):  # 次近邻
        L = self.L
        if self.dimension == 2:
            if self.ispbc:
                # 前N个是x+1,y+1,后N个是x+1,y-1
                self.nnnb = 2 * self.N
                self.nnnbond_index = np.zeros((2, self.nnnb), dtype=int)  # next near neighbor
                x = np.tile(np.arange(L), L)
                y = np.arange(L).repeat(L)
                self.nnnbond_index[0, :int(self.Nb / 2)] = l2c(x, y, L)
                self.nnnbond_index[1, :int(self.Nb / 2)] = l2c(pbc(x + 1, L), pbc(y + 1, L), L)
                self.nnnbond_index[0, int(self.Nb / 2):] = l2c(x, y, L)
                self.nnnbond_index[1, int(self.Nb / 2):] = l2c(pbc(x + 1, L), pbc(y - 1, L), L)
            else:  # obc
                # 前(L - 1) ** 2个是x+1,y+1,后(L - 1) ** 2个是x+1,y-1
                self.nnnb = 2 * (L - 1) ** 2
                self.nnnbond_index = np.zeros((2, self.nnnb), dtype=int)  # next near neighbor
                x = np.tile(np.arange(L), L).reshape(L, L)
                y = np.arange(L).repeat(L).reshape(L, L)
                xi = x[0, :(L - 1)]
                yi = y[0, :(L - 1)]
                self.nnnbond_index[0, :(L - 1)] = l2c(xi, yi, L)
                self.nnnbond_index[1, :(L - 1)] = l2c(xi + 1, yi + 1, L)
                xi = x[1:-1, :-1].flatten()
                yi = y[1:-1, : - 1].flatten()
                self.nnnbond_index[0, (L - 1):int((L - 1) ** 2)] = l2c(xi, yi, L)
                self.nnnbond_index[1, (L - 1):int((L - 1) ** 2)] = l2c(xi + 1, yi + 1, L)
                self.nnnbond_index[0, int((L - 1) ** 2):int((L - 1) * (2 * L - 3))] = l2c(xi, yi, L)
                self.nnnbond_index[1, int((L - 1) ** 2):int((L - 1) * (2 * L - 3))] = l2c(xi + 1, yi - 1, L)
                xi = x[-1, :-1].flatten()
                yi = y[-1, : - 1].flatten()
                self.nnnbond_index[0, int((L - 1) * (2 * L - 3)):] = l2c(xi, yi, L)
                self.nnnbond_index[1, int((L - 1) * (2 * L - 3)):] = l2c(xi + 1, yi - 1, L)
        elif self.dimension == 1:
            if self.ispbc:
                self.nnnb = self.N
                self.nnnbond_index = np.zeros((2, self.nnnb), dtype=int)
                self.nnnbond_index[0, :] = np.arange(L)
                self.nnnbond_index[1, :] = pbc(np.arange(2, L + 2), L)
            else:
                self.nnnb = self.N - 2
                self.nnnbond_index = np.zeros((2, self.nnnb), dtype=int)
                self.nnnbond_index[0, :] = np.arange(L - 2)
                self.nnnbond_index[1, :] = np.arange(2, L)
        else:
            print("dimension tbc")

    def random_config_initial(self):
        if self.isBond:
            configuration = np.random.choice([-1, 1], size=self.Nb, replace=True)
        else:
            configuration = np.random.choice([-1, 1], size=self.N, replace=True)
        return configuration

    def construct_velocity(self, configuration):
        N = self.N
        v = np.zeros((N, N), dtype=np.complex128)
        assert self.isBond
        if self.dimension == 2:  # 对于2D，只计算vx
            v[self.bond_index[0, :N], self.bond_index[1, :N]] = 1.j * (1 - self.U * configuration[:N])
            v[self.bond_index[1, :N], self.bond_index[0, :N]] += -1.j * (1 + self.U * configuration[:N])
            if self.nnn != 0:
                v[self.nnnbond_index[0, :], self.nnnbond_index[1, :]] += 1.j * self.nnn
                v[self.nnnbond_index[1, :], self.nnnbond_index[0, :]] += -1.j * self.nnn
        elif self.dimension == 1:
            v[self.bond_index[0, :], self.bond_index[1, :]] = 1.j * (1 - self.U * configuration)
            v[self.bond_index[1, :], self.bond_index[0, :]] += -1.j * (1 + self.U * configuration)
            if self.nnn != 0:
                v[self.nnnbond_index[0, :], self.nnnbond_index[1, :]] += 2.j * self.nnn
                v[self.nnnbond_index[1, :], self.nnnbond_index[0, :]] += -2.j * self.nnn
        else:
            raise ValueError
        return v

    def construct_H(self, configuration, phi_x=0, phi_y=0, similar_transformation=True):
        if phi_x or phi_y:
            H = np.zeros((self.N, self.N), dtype=complex)
        else:
            H = np.zeros((self.N, self.N))
        if self.isBond:
            H[self.bond_index[0, :], self.bond_index[1, :]] = 1 + (-self.U) * configuration
            if (H[self.bond_index[1, :], self.bond_index[0, :]] != 0).any():
                print('twist boundary condition error')
            H[self.bond_index[1, :], self.bond_index[0, :]] += 1 + (self.U) * configuration
        if self.nnn:
            if (H[self.nnnbond_index[0, :], self.nnnbond_index[1, :]] != 0).any():
                print('twist boundary condition error')
            H[self.nnnbond_index[0, :], self.nnnbond_index[1, :]] += self.nnn
            if (H[self.nnnbond_index[1, :], self.nnnbond_index[0, :]] != 0).any():
                print('twist boundary condition error')
            H[self.nnnbond_index[1, :], self.nnnbond_index[0, :]] += self.nnn
        if phi_x != 0 and self.dimension == 2:
            L = self.L
            boundary_index = (L - 1) + np.arange(L) * L  # (L-1) + n * L,n=0,1,...L-1
            H[self.bond_index[0, boundary_index], self.bond_index[1, boundary_index]] *= np.exp(1.j * phi_x)
            H[self.bond_index[1, boundary_index], self.bond_index[0, boundary_index]] *= np.exp(-1.j * phi_x)
            if self.nnn:
                boundary_index = (L - 1) + np.arange(2 * L) * L  # (L-1) + n * L, ,n=0,1,...2L-1
                H[self.nnnbond_index[0, boundary_index], self.nnnbond_index[1, boundary_index]] *= np.exp(1.j * phi_x)
                H[self.nnnbond_index[1, boundary_index], self.nnnbond_index[0, boundary_index]] *= np.exp(-1.j * phi_x)
        if phi_y != 0 and self.dimension == 2:
            L = self.L
            boundary_index = L * (2 * L - 1) + np.arange(L)  # L**2+ L(L-1) + n, n=0,1,...L-1
            H[self.bond_index[0, boundary_index], self.bond_index[1, boundary_index]] *= np.exp(1.j * phi_y)
            H[self.bond_index[1, boundary_index], self.bond_index[0, boundary_index]] *= np.exp(-1.j * phi_y)
            if self.nnn:
                boundary_index = L * (L - 1) + np.arange(L)  # L(L-1) + n, n=0,1,...L-1
                H[self.nnnbond_index[0, boundary_index], self.nnnbond_index[1, boundary_index]] *= np.exp(1.j * phi_y)
                H[self.nnnbond_index[1, boundary_index], self.nnnbond_index[0, boundary_index]] *= np.exp(-1.j * phi_y)
                boundary_index = L ** 2 + np.arange(L)  # L**2 + n, n=0,1,...L-1 这里的bond是向下跨过边界条件，所以是先负后正
                H[self.nnnbond_index[0, boundary_index], self.nnnbond_index[1, boundary_index]] *= np.exp(-1.j * phi_y)
                H[self.nnnbond_index[1, boundary_index], self.nnnbond_index[0, boundary_index]] *= np.exp(1.j * phi_y)
        similar_P = []
        return H, similar_P

    def calculate_weight(self, configuration, gs_approx=False):
        if self.nnn == 0:
            assert self.ispbc and self.dimension == 1
            n1 = int((np.sum(configuration) + self.L) / 2)
            tR = (1 + self.U) ** (n1 / self.L) * (1 - self.U) ** (1 - n1 / self.L)
            tL = (1 - self.U) ** (n1 / self.L) * (1 + self.U) ** (1 - n1 / self.L)
            eigenvalues = np.zeros(self.L, dtype=complex)
            for n in range(self.L):
                k = 2 * np.pi * n / self.L
                eigenvalues[n] = (tR + tL) * np.cos(k) + 1.j * (tR - tL) * np.sin(k)
        else:
            H, _ = self.construct_H(configuration)
            eigenvalues = np.linalg.eigvals(H)

        if gs_approx:
            energy = np.sum(eigenvalues[np.where(eigenvalues.real <= 0)[0]])
        else:
            energy = np.sum(eigenvalues / (1 + np.exp(self.Beta * eigenvalues)))
        log_weight = np.sum(np.log(1 + np.exp(-self.Beta * eigenvalues)))

        try:
            assert not np.isinf(log_weight)
        except AssertionError:
            print("The weight contains inf value:", log_weight)
            print("log_weight inf")
            exit(0)
        return log_weight, energy

    def local_update(self, configuration, charge_number=1, restart=False):
        indices = np.random.choice(range(configuration.shape[0]), size=charge_number, replace=False)
        configuration_ = configuration.copy()
        configuration_[indices] = -configuration_[indices]
        if restart and np.random.random() < 0.005:  # 随机改变全局
            configuration_ = np.random.choice([-1, 1], size=configuration.shape[0])
        logweight_, energy = self.calculate_weight(configuration_)
        return configuration_, logweight_, energy

    def calculate(self, configuration, beta):
        assert self.dimension == 1
        L = self.L
        # 1.计算Xop
        Xop = np.abs(np.sum(configuration)) / L

        # 2.计算XX_cor
        if self.ispbc:
            XX_cor = np.zeros(int(L / 2) + 1)
            for i in range(L):
                for r in range(int(L / 2) + 1):
                    XX_cor[r] += configuration[i] * configuration[pbc(i + r, L)]  # delta r = int(L/2)
            XX_cor = XX_cor / L
        else:
            XX_cor = np.zeros(int(L / 2) + 1)
            for r in range(int(L / 2) + 1):
                XX_cor[r] += configuration[0] * configuration[r]  # delta r = int(L/2)
        # 对角化矩阵，计算费米子相关量
        H, similar_P = self.construct_H(configuration, similar_transformation=True)
        eigenvalues, P = np.linalg.eig(H)
        # index = np.where(eigenvalues.real < 0)[0]
        P_inv = np.linalg.inv(P)

        # 3.and 4. 计算能谱和多体基态能量
        E = np.nansum(eigenvalues / (np.exp(beta * eigenvalues) + 1))

        # 5. 计算v期望值
        v = self.construct_velocity(configuration)  # vx的矩阵，表象是vx，vy
        if similar_P:
            v = np.diag(1 / similar_P).dot(v.dot(np.diag(similar_P)))
        a = P_inv.dot(v).dot(P)
        vop = np.nansum(np.diag(a) / (np.exp(beta * eigenvalues) + 1))
        vv_cor = 0
        return Xop, XX_cor, eigenvalues, E, vop, vv_cor

    def calculate_classical_order_parameter(self, configuration):
        """classical order parameter"""
        L = self.L
        if self.dimension == 2:
            if self.isBond:
                opx = np.sum(configuration[:int(self.Nb / 2)])
                opy = np.sum(configuration[int(self.Nb / 2):])
                X_op = (np.abs(opx) + np.abs(opy)) / self.Nb
                X_corr = 0
                for i in range(self.Nb):
                    X_corr += configuration[i] * configuration[pbc(i + 1, L)]
                X_corr = X_corr / self.Nb
            else:
                opx = 0
                opy = 0
                X_corr = 0
                for i in range(len(configuration)):
                    x, y = c2l(i, L)
                    opx += configuration[i] * configuration[l2c(pbc(x + 1, L), y, L)]
                    opy += configuration[i] * configuration[l2c(x, pbc(y + 1, L), L)]
                    X_corr += configuration[i] * configuration[l2c(pbc(x + 2, L), y, L)] \
                              + configuration[i] * configuration[l2c(x, pbc(y + 2, L), L)]
                X_op = (np.abs(opx) + np.abs(opy)) / (2 * self.N)
                X_corr = X_corr / (2 * self.N)
            return X_op, X_corr
        elif self.dimension == 1:
            if self.isBond:
                X_op = np.abs(np.sum(configuration)) / (self.N)
            else:
                X_op = 0
                for i in range(len(configuration)):
                    X_op += configuration[i] * configuration[pbc(i + 1, L)]
                X_op = np.abs(X_op) / (self.N)
            return X_op, None
        else:
            print("calculate_classical_order_parameter tbc")
            exit(0)


    def calculate_twist_DOS(self, configuration):
        M = 30
        energys = []
        for i in range(M):
            for j in range(M):
                H1, _ = self.construct_H(configuration, phi_x=2 * np.pi * i / M, phi_y=2 * np.pi * j / M)
                eigenvalues1 = np.linalg.eigvals(H1)
                energys.append(eigenvalues1)


def f_ijkh(M, i, j, k, h):
    # 这个这个函数是求解general f_ijkh  c_i^{\dagger} c_j c_k^{\dagger} c_h的期望值
    item = 1 if j == k else 0
    return M[j, i] * M[h, k] - M[h, i] * M[j, k] + M[h, i] * item


def autoCorr(conf1, conf2):
    auto = np.mean(conf1 * conf2)
    return auto


def is_equilibrium(warm_energys, warm_energy2s):
    # if len(warm_energys) >= 500 and np.abs(
    #         mean_outlier(warm_energys[-200:-100]) - mean_outlier(warm_energy2s[-200:-100])) < 0.1:
    if len(warm_energys) >= 500:
        diff = np.abs(mean_outlier(warm_energys[-200:-100]) - mean_outlier(warm_energy2s[-200:-100]))
        if diff < 0.1 and diff / np.abs(mean_outlier(warm_energys[-200:-100])) < 0.0001:
            return True
    return False


def do_measure(mc, sample_num, configuration, log_weight, f, filepath):
    # print(f"Beta {mc.Beta:.2f} begin measurement")
    t0 = time.time()
    """------MC 测量----------"""
    Xop_list = []
    XX_cor_list = []
    spectra_list = []
    MBE_list = []
    vop_list = []
    vv_cor_list = []
    """------MC 更新相关量----------"""
    accept_num = 0
    autocorrelation_list = []
    configuration_set = configuration.copy()
    auto_length = None  # None 采样步长
    sample_id = 0  # 采样次数
    i = 0
    while True:
        i += 1
        if not auto_length:  # 设置自相关长度
            auto_cor = autoCorr(configuration, configuration_set)
            autocorrelation_list.append(auto_cor)
            if auto_cor < 0.4:
                auto_length = max(i, 100)
                print(f"auto_length = {auto_length:>5}", file=f)
            if i >= 500:
                auto_length = 500  # 最大采样长度为500
                print(f"reach max auto_length = {auto_length:>5}", file=f)
                # plt.plot(autocorrelation_list)
                # plt.title("auto_length")
                # plt.show()
        configuration_, log_weight_, Energy = mc.local_update(configuration, restart=True)
        if np.random.rand() < calculate_prob(log_weight_, log_weight):  # accept
            accept_num += 1
            configuration = configuration_.copy()
            log_weight = log_weight_
        if auto_length and i % auto_length == 0:
            Xop, XX_cor, eigenvalues, E, vop, vv_cor = mc.calculate(configuration, mc.Beta)
            Xop_list.append(Xop)
            XX_cor_list.append(XX_cor)
            spectra_list.append(eigenvalues)
            MBE_list.append(E)
            vop_list.append(vop)
            vop_list.append(vop.conj())  # 考虑了-X
            vv_cor_list.append(vv_cor)
            sample_id += 1
            if sample_id >= sample_num:
                break
    Xop_mean = np.mean(np.array(Xop_list))
    XX_cor_mean = np.mean(np.array(XX_cor_list),axis=0)
    spectra = np.array(spectra_list)
    Energy = np.array(MBE_list)
    Cv = (np.mean(Energy ** 2) - np.mean(Energy) ** 2).real * mc.Beta ** 2 / (mc.L ** mc.dimension)
    vop_array = np.array(vop_list)
    vv_cor_mean = np.mean(np.array(vv_cor_list)).real
    print("accept rate=", accept_num / i, file=f)
    print("Xop=", Xop_mean, file=f)
    print("Cv=", Cv, file=f)
    print("vop=", np.mean(vop_array), file=f)
    print("vv_corr=", vv_cor_mean, file=f)
    print(f"completed measurement at step {i:>10}", file=f)
    f.flush()
    # cal_k_green_func(N, H_sum, rf"L={mc.L} $\beta$={mc.Beta} U={mc.U}, $\Delta$={op_mean:.2f}")
    if filepath:
        np.savez(os.path.join(filepath, f'Beta{mc.Beta:.2f}.npz'),
                 Xop=Xop_mean, XX_cor=XX_cor_mean, spectra=spectra,
                 Cv=Cv, vop=vop_array, vv_cor=vv_cor_mean,
                 begin_conf=configuration, )
    print(f"L{mc.L} Beta {mc.Beta:.2f} measurement end, used time = {time.time() - t0:.2f}s")


def main(L, U, beta_list, msteps, sample_num, dimension, ispbc, load_beta=None, epoch=1, filesave=False, isBond=False,
         nnn=None, breakpoint_load=False):
    t_begin = time.time()
    np.random.seed(epoch + int((time.time() * 1e6) % 1e9))
    if filesave:
        name = f"results_{dimension}D_nnn{nnn}_U{U:.2f}" if ispbc else f"results_obc_{dimension}D_nnn{nnn}_U{U}"
        filepath = os.path.join(name, f"L{L}_ep{epoch}")
        os.makedirs(filepath, exist_ok=True)  # 如果不存在创建这个文件夹
        if breakpoint_load:
            f = open(os.path.join(filepath, f'info.txt'), 'a+')  # 如果文件不存在则创建，且读写不覆盖
        else:
            f = open(os.path.join(filepath, f'info.txt'), 'w+')  # 如果文件不存在则创建，且读写覆盖
        print(f"L:{L}, U:{U},,nnn=:{nnn}, dimension:{dimension}", file=f)
        print(f"beta_list = {beta_list[0]:.2f}:{beta_list[-1]:.2f}:{beta_list[1]-beta_list[0]:.2f}", file=f)
        print(f"msteps:{msteps}, sample_num:{sample_num},ispbc:{ispbc},isBond:{isBond}", file=f)
        f.flush()
    else:
        filepath = None
        f = sys.stdout

    inital_beta = beta_list[0]  # 初始温度
    beta_index = 0  # 当前准备测量的是 Beta = beta_list[beta_index]
    mc = MentoCarlo(L, U, inital_beta, dimension, ispbc, isBond, nnn)
    """------热化----------"""
    configuration = mc.random_config_initial()
    configuration2 = np.ones(mc.Nb)  # auxiliary order state
    if load_beta:
        assert beta_list[0] >= load_beta
        mc.Beta = load_beta
        data = np.load(os.path.join(filepath, f'Beta{mc.Beta:.2f}.npz'))
        configuration = data['begin_conf']
    if breakpoint_load:  # 断点加载
        if os.path.exists(os.path.join(filepath, f'Beta{beta_list[-1]:.2f}.npz')):
            print(f"L{L} epoch{epoch} complete !")
            exit(0)
        for idx, beta in enumerate(beta_list):
            if not os.path.exists(os.path.join(filepath, f'Beta{beta:.2f}.npz')):
                # 如果这个数值之前没有run过，那么从前一个数值开始
                if idx != 0:
                    mc.Beta = beta_list[idx - 1]
                    beta_index = idx-1
                    data = np.load(os.path.join(filepath, f'Beta{beta_list[idx - 1]:.2f}.npz'))
                    configuration = data['begin_conf']
                    print(f"L{L} Load data from beta={beta_list[idx - 1]:.2f}")
                break

    log_weight, energy = mc.calculate_weight(configuration)
    log_weight2, energy2 = mc.calculate_weight(configuration2)
    warm_energys = []
    warm_energy2s = []
    i = 0
    while True:  # 退火
        i += 1
        configuration_, log_weight_, energy = mc.local_update(configuration, restart=True)
        if np.random.rand() < calculate_prob(log_weight_, log_weight):  # accept
            configuration = configuration_.copy()
            log_weight = log_weight_
        configuration_2, log_weight_2, energy2 = mc.local_update(configuration2, restart=True)
        if np.random.rand() < calculate_prob(log_weight_2, log_weight2):  # accept
            configuration2 = configuration_2.copy()
            log_weight2 = log_weight_2
        warm_energys.append(energy)
        warm_energy2s.append(energy2)

        if i % 10 == 0 and is_equilibrium(warm_energys, warm_energy2s):  # 判断热化是否完成
            print(f"L{L} Beta={mc.Beta:.2f} reach equilibrium at step {i:>5}", file=f)
            f.flush()
            if mc.Beta == beta_list[beta_index]:  # 如果热化完成且刚好是所需要的温度，则做测量
                do_measure(mc, sample_num, configuration2, log_weight, f, filepath)
                if beta_index < len(beta_list) - 1:  # 如果还有下一个温度，则更新温度
                    beta_index += 1
                else:
                    print(f"L{L} All measurement finish!")
                    break
            mc.Beta = min(mc.Beta * 1.05, beta_list[beta_index])
            configuration2 = np.ones(mc.Nb)
            warm_energys = []
            warm_energy2s = []
            i = 0
        if i > msteps:
            print(f"L{L}_U{U}_Beta{mc.Beta} failed to reach equilibrium within {i} step", file=f)
            print("used time:", time.time() - t_begin, file=f)
            break

    print("used time:", time.time() - t_begin, file=f)


if __name__ == '__main__':
    msteps = 100000  # 最大热化步长
    sample_num = 100  # 采样次数
    dimension = 1
    ispbc = True
    isBond = True
    filesave = True
    nnn = 0.

    "----单核-----"
    L = 10  # 系统大小
    U = 0.4  # u/t
    epoch = 0
    t0 = time.time()
    load_beta = None  # None
    beta_list = list(np.arange(2, 25, 0.1))  # 温度列表要求按从小到大顺序排列，且beta_list[0]<=load_beta,
    # beta_list = [15]
    # 注意 beta的精度是0.01且 Beta最小不能是0
    ret = main(L, U, beta_list, msteps, sample_num, dimension, ispbc, load_beta, epoch, True, isBond, nnn)

