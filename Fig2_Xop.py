import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from main_st import *

# 设置全局字体大小
plt.rcParams.update({'font.size': 10})

if __name__ == '__main__':
    U = 0.4
    J = 0.0
    gs_approx = False
    compute_v = False

    beta_list = np.arange(0.1, 35, 0.01)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 7))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20  # 设置坐标轴标签字体大小
    plt.rcParams['axes.linewidth'] = 1  # 设置坐标轴粗细
    plt.rcParams['lines.linewidth'] = 3  # 设置线宽
    plt.rcParams['lines.markersize'] = 8  # 设置符号大小
    fig = plt.figure(figsize=(11, 8.5))  # 调整整体图形的尺寸
    gs = gridspec.GridSpec(2, 2, width_ratios=[6, 1], hspace=0)  # 设置第四列较窄，用于图例
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    # ax_legend = fig.add_subplot(gs[:, 1])  # 创建 legend 子图
    colors = ["#0249B2", "#F07E40", "#4B7DD1"]
    """MC 数据"""
    dimension, nnn, L = 1, 0.0, 70
    for idx, nnn in enumerate([0.0, 0.5]):
        data = np.load(f"Fig2_mc/result_L{L}_nnn{nnn}.npz")
        op_mc = data['op']
        Cv_mc = data['Cv']
        beta_list_mc = data['beta_list']
        ax1.fill_between(beta_list_mc, np.nanmean(op_mc, axis=0) - np.nanstd(op_mc, axis=0),
                         np.mean(op_mc, axis=0) + np.std(op_mc, axis=0)
                         , color=colors[idx], alpha=0.5, label=f"MC: $t'={nnn}$")
        ax2.fill_between(beta_list_mc, np.nanmean(Cv_mc, axis=0) - np.nanstd(Cv_mc, axis=0),
                         np.mean(Cv_mc, axis=0) + np.std(Cv_mc, axis=0)
                         , color=colors[idx], alpha=0.5, label=f"MC: $t'={nnn}$")
    """ST 数据"""
    for idx, L in enumerate([70]):
        if os.path.exists(f"phase_diag/L{L}_J{J:.2f}_U{U:.2f}.npz"):
            data = np.load(f"phase_diag/L{L}_J{J:.2f}_U{U:.2f}.npz")
            X_exp = data['X_exp']
            Cv_exp = data['Cv_exp']
            Xcor_exp = data['Xcor_exp']
            p = data['p']
            beta_c = data['beta_c']
            peaks_id, _ = find_peaks(Cv_exp, height=1E-2)
        else:
            combination = Omega_matrix(L, J)
            Cv_exp, X_exp, Xcor_exp, v, vv_exp, p = main_st(L, U, J, beta_list, combination, gs_approx, compute_v)
            peaks_id, _ = find_peaks(Cv_exp, height=1E-2)
            if peaks_id.size != 0:
                beta_c.append(beta_list[peaks_id])
            else:
                beta_c.append(None)
            np.savez(f"phase_diag/L{L}_J{J:.2f}_U{U:.2f}.npz", X_exp=X_exp, Cv_exp=Cv_exp,
                     Xcor_exp=Xcor_exp, v=v, p=p,
                     vv_exp=vv_exp,
                     beta_list=beta_list,
                     beta_c=beta_c)
        ax1.plot(beta_list, X_exp, '-', label=f"Exact Sum", c=colors[idx])
        ax2.plot(beta_list, Cv_exp, '-', label=f"Exact Sum", c=colors[idx])
        # ax2.scatter(beta_list[peaks_id], np.array(Cv_exp)[peaks_id], s=100, color='red')

    ax1.set_ylabel(r'$<X>$')
    ax1.set_xlabel(r'$\beta$')
    ax1.set_xlim(3, 35)
    ax2.set_xlabel(r'$\beta$')
    ax2.set_ylabel(r'$C_v$')

    from matplotlib.ticker import MaxNLocator

    ax2.yaxis.set_major_locator(MaxNLocator(4))  # 自动设置y轴刻度，最多显示4个刻度
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax1.yaxis.set_major_locator(MaxNLocator(5))


    for beta in [10.57, 14.13, 20.30, 27.65]:
        ax1.axvline(beta, color=colors[2], linestyle='--', alpha=1, lw=1.2)
        ax2.axvline(beta, color=colors[2], linestyle='--', alpha=1, lw=1)
        print(f"beta_c={beta:.2f}")

    ax1.legend(loc='upper left', handlelength=1.0, frameon=False, )
    ax2.legend(loc='upper left', handlelength=1.0, frameon=False, )

    ax3 = ax1.inset_axes([0.6, 0., 0.4, 0.55])
    ax4 = ax2.inset_axes([0.6, 0.45, 0.4, 0.55])
    ax3.set_xlim(8, 17.5)
    ax4.set_xlim(8, 17.5)
    """ST 有限尺度数据"""
    for idx, L in enumerate([70, 110, 150]):
        if os.path.exists(f"phase_diag/L{L}_J{J:.2f}_U{U:.2f}.npz"):
            data = np.load(f"phase_diag/L{L}_J{J:.2f}_U{U:.2f}.npz")
            X_exp = data['X_exp']
            Cv_exp = data['Cv_exp']
            Xcor_exp = data['Xcor_exp']
            p = data['p']
        else:
            combination = Omega_matrix(L, J)
            Cv_exp, X_exp, Xcor_exp, v, vv_exp, p = main_st(L, U, J, beta_list, combination, gs_approx, compute_v)
            peaks_id, _ = find_peaks(Cv_exp, height=1E-2)
            if peaks_id.size != 0:
                beta_c = beta_list[peaks_id]
            else:
                beta_c = None
            np.savez(f"phase_diag/L{L}_J{J:.2f}_U{U:.2f}.npz", X_exp=X_exp, Cv_exp=Cv_exp, Xcor_exp=Xcor_exp, v=v, p=p,
                     vv_exp=vv_exp,
                     beta_list=beta_list,
                     beta_c=beta_c)
        ax3.plot(beta_list, X_exp, '-', label=f"L={L}", c=colors[idx])
        ax4.plot(beta_list, Cv_exp, '-', label=f"L={L}", c=colors[idx])
    ax3.legend(loc='lower right', handlelength=1.0, frameon=False, bbox_to_anchor=(1, -0.1), )
    # ax4.legend(loc='upper left', handlelength=1.0, frameon=False, )
    ax3.set_ylabel("$<X>$")
    ax4.set_ylabel("$C_v$")
    ax4.set_xlabel(r'$\beta$')
    ax3.text(0.05, 0.8, r'Exact Sum', transform=ax3.transAxes)
    ax4.text(0.05, 0.8, r'Exact Sum', transform=ax4.transAxes)
    ax3.xaxis.set_major_locator(MaxNLocator(2))
    ax4.xaxis.set_major_locator(MaxNLocator(2))
    # plt.savefig("Fig2.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    exit(0)
