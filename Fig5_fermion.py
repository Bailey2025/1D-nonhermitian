import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from main_st import *

if __name__ == '__main__':
    L = 150
    U = 0.4
    J = 0.0
    gs_approx = False
    compute_v = True
    beta_list = np.arange(0.1, 35, 0.1)
    combination = Omega_matrix(L, J)
    if os.path.exists(f"phase_diag/winding_number_L{L}_J{J:.2f}_U{U:.2f}.npz"):
        data = np.load(f"phase_diag/winding_number_L{L}_J{J:.2f}_U{U:.2f}.npz")
        number = data['number']
        Cv_exp = data['Cv_exp']
        v = data['v']
        p = data['p']
    else:
        Cv_exp, X_exp, Xcor_exp, v, vv_exp, p = main_st(L, U, J, beta_list, combination, gs_approx, compute_v)
        """计算不同beta的winding number """
        number = []
        for i in range(beta_list.shape[0]):
            hist, bin_edges = np.histogram(v[i, :], bins=1000, weights=p[i, :])
            max_weight_idx = np.argmax(hist)  # 找到权重最大的位置
            imv = bin_edges[max_weight_idx]
            winding_number = imv * beta_list[i] / L
            number.append(abs(winding_number))
        number = np.array(number)

    # 全局设置字体为 Arial，大小为 16
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20  # 设置坐标轴标签字体大小
    plt.rcParams['axes.linewidth'] = 1  # 设置坐标轴粗细
    plt.rcParams['lines.linewidth'] = 3  # 设置线宽
    plt.rcParams['lines.markersize'] = 8  # 设置符号大小

    fig = plt.figure(figsize=(11, 11))  # 调整整体图形的尺寸
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 3.5], hspace=0.25, wspace=0.3)  # 设置第四列较窄，用于图例
    ax1 = fig.add_subplot(gs[1, :])
    for i in range(0, 9, 2):
        ax1.axhline(i, linestyle=":", color="#B4D1E7", lw=1.5)
    peaks_id, _ = find_peaks(Cv_exp, height=1E-2)
    # for i in peaks_id:
    for beta in [10.57, 14.13, 20.30, 27.65]:
        ax1.axvline(beta, color='#F07E40', linestyle='--', alpha=1, lw=2)
    ax1.plot(beta_list, number, '-', label=f"L={L}", c='#0249B2', lw=5)

    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    colors = ['#C25365', '#7BC5C8', "#E1D066", "#63AD7F", "#29377F", ]
    v = v / L
    for idx, i in enumerate([66, 121, 171, 239]):  # 在0位置添加索引1
        beta = beta_list[i]
        arrow_length = 0.5
        ax1.arrow(x=beta, y=number[i] + arrow_length, dx=0, dy=-arrow_length + 0.3,
                  head_width=0.8, head_length=0.3,
                  fc=colors[idx], ec=colors[idx], width=0.4)
        ax2.hist(v[i, :], bins=100, weights=p[i, :],
                 label=f'beta={beta_list[i]:.1f}', alpha=1,
                 color=colors[idx], edgecolor=colors[idx])
        if os.path.exists(f"phase_diag/vvcor_L{L}_U{U:.2f}_beta{beta:.2f}.npy"):
            vv_exp_beta = np.load(f"phase_diag/vvcor_L{L}_U{U:.2f}_beta{beta:.2f}.npy")
        else:
            vv_exp_beta = compute_vvcor(L, U, J, beta, combination, gs_approx)
            np.save(f"phase_diag/vvcor_L{L}_U{U:.2f}_beta{beta:.2f}.npy", vv_exp_beta)
        ax3.plot(vv_exp_beta, label=rf'$\beta={beta_list[i]:.1f}$', c=colors[idx])

    ax3.set_ylim(0, 0.2)
    ax3.set_xlim(0, 30)
    ax3.set_xlabel(r'$r$')
    ax3.set_ylabel('Velocity Correlation')
    ax3.legend(loc='upper right', bbox_to_anchor=(1, 1.05), frameon=False, )
    ax1.set_xlabel(r'$\beta$')
    ax1.set_xlim(3, 35)
    ax1.set_ylim(-0.5, 9.5)
    ax1.set_ylabel(r'$|w|$')
    ax2.set_xlabel(r'$Im(v)$')
    ax2.set_ylabel("Prob of " + r'$Im(v)$')
    from matplotlib.ticker import MaxNLocator

    ax1.yaxis.set_major_locator(MaxNLocator(5))  # 自动设置y轴刻度，最多显示4个刻度
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax2.yaxis.set_major_locator(MaxNLocator(3))  # 自动设置y轴刻度，最多显示4个刻度
    ax2.xaxis.set_major_locator(MaxNLocator(3))
    ax3.yaxis.set_major_locator(MaxNLocator(2))  # 自动设置y轴刻度，最多显示4个刻度
    ax3.xaxis.set_major_locator(MaxNLocator(4))


    ax1.text(0.02, 0.95, '(c)', transform=ax1.transAxes,
             va='top', ha='left')
    ax2.text(0.02, 0.95, '(a)', transform=ax2.transAxes,
             va='top', ha='left')
    ax3.text(0.05, 0.95, '(b)', transform=ax3.transAxes,
             va='top', ha='left')
    # plt.savefig("Fig5.svg")
    plt.show()

    exit(0)
