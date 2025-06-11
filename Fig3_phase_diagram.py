from main_st import *


def f(x):
    return 2 * x * np.tanh(x) / np.pi


def phase_diag(L, J, u_list, max_beta_list):
    beta_c = []
    combination = Omega_matrix(L, J)
    for i in range(u_list.size):
        U = u_list[i]
        max_beta = min(max_beta_list[i] * 1.5, max_beta_list[i] + 40)
        beta_list = np.linspace(1, max_beta, 400)
        Cv_exp, X_exp, Xcor_exp, v, vv_exp, p = main_st(L, U, J, beta_list, combination, gs_approx=False,
                                                        compute_v=True)
        np.savez(f"phase_diag/L{L}_J{J:.2f}_U{U:.2f}.npz", X_exp=X_exp, Cv_exp=Cv_exp, Xcor_exp=Xcor_exp, v=v, p=p,
                 vv_exp=vv_exp,
                 beta_list=beta_list)
        # winding number找相变
        n1 = np.argmax(p, axis=1)  # beta_num
        imv = v[np.arange(n1.size), n1]
        winding_number = np.abs(imv * beta_list / L)
        id = np.where(winding_number > 0.4)
        if id[0].size != 0:
            beta_c.append(beta_list[id[0][0]])
        else:
            beta_c.append(None)
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(6, 15))
        ax0.plot(beta_list, X_exp, 'o-', label=f"L={L}")
        ax1.plot(beta_list, Cv_exp, 'o-', label=f"L={L}")
        ax2.plot(beta_list, winding_number, 'o-', label=f"L={L}")
        ax0.set_title(f"L={L}, U={U}, J={J}")
        ax0.set_xlabel(r'$\beta$')
        ax0.set_ylabel(r'$X$')
        ax1.set_xlabel(r'$\beta$')
        ax1.set_ylabel(r'$vvcor$')
        ax1.set_xlabel(r'$\beta$')
        ax1.set_ylabel(r'winding number')
        plt.show()
    return np.array(beta_c)


if __name__ == '__main__':
    u_list = np.arange(0.1, 1, 0.05)
    length = u_list.size
    L = 150
    T0 = f(np.arange(0., 1, 0.05))
    T1 = np.load("phase_diag/MF_J0.05.npz")['T']
    if os.path.exists(f"phase_diag/beta_c_L{L}_J0.00.npy"):
        beta_c0 = np.load(f"phase_diag/beta_c_L{L}_J0.00.npy")[-length:]
    else:
        beta_c0 = phase_diag(L, 0.00, u_list, max_beta_list=1 / T0[-length:])
        np.save(f"phase_diag/beta_c_L{L}_J0.00.npy", beta_c0)

    if os.path.exists(f"phase_diag/beta_c_L{L}_J0.05.npy"):
        beta_c1 = np.load(f"phase_diag/beta_c_L{L}_J0.05.npy")[-length:]
    else:
        beta_c1 = phase_diag(L, 0.05, u_list, max_beta_list=1 / T1[-length:])
        np.save(f"phase_diag/beta_c_L{L}_J0.05.npy", beta_c1)

    # 全局设置字体为 Arial，大小为 16
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20  # 设置坐标轴标签字体大小
    plt.rcParams['axes.linewidth'] = 1  # 设置坐标轴粗细
    plt.rcParams['lines.linewidth'] = 3  # 设置线宽
    plt.rcParams['lines.markersize'] = 8  # 设置符号大小

    fig, ax = plt.subplots(figsize=(11, 8.5))
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(4))  # 自动设置y轴刻度，最多显示4个刻度
    ax.xaxis.set_major_locator(MaxNLocator(4))

    plt.plot(1 / beta_c0, u_list, 'o:', color='#0249B2', label=f"Exact Sum J$=0.00$")  # 实际数据点
    plt.plot(1 / beta_c1, u_list, 'o:', color='#4B7DD1', label=f"Exact Sum J$=0.05$")  # 实际数据点
    u_list = np.arange(0.05, 1, 0.05)
    length = u_list.size
    plt.plot(T0[-length:], u_list, '-', color='#0249B2', label=f"MFT J$=0.00$")
    plt.plot(T1[-length:], u_list, '-', color='#4B7DD1', label=f"MFT J$=0.05$")

    plt.axhline(0.4, color='gray', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(0.03, 0.65, 'Ordered Phases', fontsize=25, color='#F07E40', fontweight='bold', fontname='Arial')
    plt.text(0.25, 0.45, 'Disordered Phase', fontsize=25, color='#F07E40', fontweight='bold', fontname='Arial')

    plt.xlim(0., 0.45)
    plt.ylim(0.0, 1)
    plt.xlabel(r'$T$')
    plt.ylabel(r'$U$')

    plt.legend(frameon=False, loc='lower right')
    # plt.savefig('Fig3.pdf', dpi=300, bbox_inches='tight')
    plt.show()
