from main_mc import *


# 定义对数拟合的函数模型
def fit(x, a, b, c, d):
    return a - b / x + c * x + np.log(x) * d


def curve(L, U, n1):
    tL = (1 + U) ** (n1 / L) * (1 - U) ** (1 - n1 / L)
    tR = (1 - U) ** (n1 / L) * (1 + U) ** (1 - n1 / L)
    E = ((tL + tR - 2)) * (-L / np.pi)
    return E


def energy(L, Beta, n1, nnn, ):
    dimension = 1
    ispbc = True
    isBond = True
    mc = MentoCarlo(L, U, Beta, dimension, ispbc, isBond, nnn)
    mc.make_lattice()
    Nb = mc.Nb
    configuration = np.ones(Nb, dtype=int)
    energy = mc.calculate_weight(configuration, gs_approx=True)[1].real

    indices = list(range(0, n1))
    configuration[indices] = -configuration[indices]
    energy_n1 = mc.calculate_weight(configuration, gs_approx=True)[1].real
    return (energy_n1 - energy)


L_list = np.arange(6, 150)
L_num = L_list.size
J = 0.0
nnn = 0.0
U = 0.4
Beta = 100
if os.path.exists("domain/1E.npz"):
    data = np.load("domain/E.npz")
    E = data["E"]
    E_the = data["E_the"]
else:
    E = np.zeros((6, L_num))
    E_the = np.zeros((3, L_num))
    for i, L in enumerate(L_list):
        E[0, i] = energy(L, Beta, n1=1, nnn=0)
        E[1, i] = energy(L, Beta, n1=int(L / 2), nnn=0)
        E[2, i] = energy(L, Beta, n1=int(L / 10), nnn=0)
        E[3, i] = energy(L, Beta, n1=1, nnn=0.5)
        E[4, i] = energy(L, Beta, n1=int(L / 2), nnn=0.5)
        E[5, i] = energy(L, Beta, n1=int(L / 10), nnn=0.5)
        E_the[0, i] = curve(L, U, n1=1)
        E_the[1, i] = curve(L, U, n1=int(L / 2))
        E_the[2, i] = curve(L, U, n1=int(L / 6))
    np.savez("domain/E.npz", E=E, E_the=E)

# 全局设置字体为 Arial，大小为 16
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20  # 设置坐标轴标签字体大小
plt.rcParams['axes.linewidth'] = 1  # 设置坐标轴粗细
plt.rcParams['lines.linewidth'] = 3  # 设置线宽
plt.rcParams['lines.markersize'] = 8  # 设置符号大小

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(11, 8.5))  # 调整整体图形的尺寸
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0)  # 设置第四列较窄，用于图例
ax = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :], sharex=ax)


colors = ['#C25365', '#7BC5C8', "#E1D066", "#63AD7F", "#29377F", ]
colors = ['#026B93', '#74D0CD', '#E8B4DD', ]
colors = ['#0249B2', '#4B7DD1', '#F07E40', ]
ax.plot(L_list, E[1, :], lw=4, color=colors[0], label=r"$\alpha=1/2$, $t'=0.0$")
ax.scatter(L_list, E[4, :], s=10, color=colors[0], label=r"$\alpha=1/2$, $t'=0.5$", alpha=0.5)
ax.plot(L_list, E[2, :], lw=4, color=colors[1], label=r"$\alpha=1/10$, $t'=0.0$")
ax.scatter(L_list, E[5, :], s=10, color=colors[1], label=r"$\alpha=1/10$, $t'=0.5$", alpha=0.5)

ax2.plot(L_list, E[0, :], lw=4, color=colors[2], label=r"$t'=0.0$")
ax2.scatter(L_list, E[3, :], s=10, color=colors[2], label=r"$t'=0.5$", alpha=0.5)

ax.set_ylim(-0., 12)
ax2.set_ylim(0.13, 0.3)
ax.set_xlim(1, 150)
ax.legend(loc="upper right", frameon=True, handlelength=1.0, ncol=1, facecolor='white', framealpha=0.7,
          edgecolor='none')

ax2.legend(loc="upper right", frameon=False, handlelength=1.0, ncol=1)

ax.set_xlabel('$L$')
ax.set_ylabel("$E(X)-E(X_0)$")
ax2.set_ylabel("$E(X)-E(X_0)$")
ax2.set_xlabel("$L$")
from matplotlib.ticker import MaxNLocator

ax.yaxis.set_major_locator(MaxNLocator(2))  # 自动设置y轴刻度，最多显示4个刻度
ax2.yaxis.set_major_locator(MaxNLocator(2))  # 自动设置y轴刻度，最多显示4个刻度
ax.xaxis.set_major_locator(MaxNLocator(4))
from matplotlib.ticker import FormatStrFormatter

ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置y轴刻度格式为一位小数

# plt.savefig("Fig4.svg")
plt.show()
