"""
读取多个numsymbol的.pt文件，绘制NMSE(dB) vs SNR的多曲线对比图
适配文件：3GPP_numpilots30.0_numsymbols10/30/50/100_TDLA.pt
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# ===================== 1. 基础配置（按需修改） =====================
# 4个.pt文件的路径（请替换为你的实际路径！）
PT_FILE_PATHS = [
    "results_joint_seed4321/3GPP_numpilots10.0_numsymbols30.0_TDLA_S_n_pilot.pt",
    "results_joint_seed4321/3GPP_numpilots30.0_numsymbols30.0_TDLA_S_n_pilot.pt",
    "results_joint_seed4321/3GPP_numpilots50.0_numsymbols30.0_TDLA_S_n_pilot.pt",
    "results_joint_seed4321/3GPP_numpilots70.0_numsymbols30.0_TDLA_S_n_pilot.pt"
]
# 对应4个文件的numsymbol标签（用于图例）
NUMSYMBOL_LABELS = ["10", "30", "50", "70"]
SAVE_FIG_PATH = "figures/nmse_snr_multiple_S_numsybols=30.png"  # 保存路径
PLOT_TITLE = "NMSE (dB) and SNR (joint and single, numsybols=30,sparse,0.1)"  # 标题


# 可选：中文显示（取消注释）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# ===================== 2. 批量读取数据（复用并优化） =====================
def load_nmse_data(pt_file_paths):
    """批量读取4个文件的SNR和NMSE(dB)数据"""
    snr_dB = None
    nmse_dB_list = []

    for idx, pt_path in enumerate(pt_file_paths):
        # 检查文件存在性
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"未找到文件：{pt_path}")

        # 读取.pt文件（CPU兼容）
        try:
            save_dict = torch.load(pt_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"读取文件{pt_path}失败：{e}")

        # 提取核心数据
        current_snr = np.array(save_dict["snr_range"])
        oracle_log = np.array(save_dict["oracle_log"])
        # 取最后一轮迭代的NMSE，加1e-10避免log10(0)
        current_nmse = oracle_log[:, -1] + 1e-10
        current_nmse_dB = 10 * np.log10(current_nmse)

        # 校验SNR一致性
        if snr_dB is None:
            snr_dB = current_snr
        else:
            if not np.array_equal(snr_dB, current_snr):
                raise ValueError(f"文件{pt_path}的SNR范围与其他文件不一致！")

        # 校验维度
        if len(snr_dB) != len(current_nmse_dB):
            raise ValueError(f"文件{pt_path}：SNR长度{len(snr_dB)} ≠ NMSE长度{len(current_nmse_dB)}")

        # 存储数据
        nmse_dB_list.append(current_nmse_dB)

        # 打印数据概览
        print(f"=== NMSE数据概览（numpilots={NUMSYMBOL_LABELS[idx]}）===")
        print(f"SNR (dB)：{snr_dB}")
        print(f"NMSE (dB)：{current_nmse_dB.round(2)}\n")

    return snr_dB, nmse_dB_list


# ===================== 3. 绘制NMSE多曲线 =====================
def plot_nmse_curve(snr_dB, nmse_dB_list, save_path, title):
    """绘制4条NMSE(dB)曲线，符合通信领域可视化规范"""
    # 绘图风格配置（论文级）
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'grid.alpha': 0.3,
        'figure.figsize': (9, 6)
    })

    # 定义4种高区分度的颜色和标记（与SER图保持一致，便于对比）
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']  # 红、绿、蓝、橙
    markers = ['o', 's', '^', '*']  # 圆、方、三角、星号

    # 创建画布
    fig, ax = plt.subplots()

    # 绘制4条NMSE曲线
    for idx, nmse_dB in enumerate(nmse_dB_list):
        ax.plot(
            snr_dB, nmse_dB,
            marker=markers[idx], markersize=7, markeredgecolor='white', markeredgewidth=1.5,
            linewidth=2.5, color=colors[idx],
            label=f'numpilots = {NUMSYMBOL_LABELS[idx]}'
        )

    # 坐标轴配置（NMSE(dB)核心规范）
    ax.set_xlabel('SNR (dB)', fontweight='bold', fontsize=13)
    ax.set_ylabel('NMSE (dB)', fontweight='bold', fontsize=13)
    # NMSE(dB)越小越好，因此y轴可设置反向（可选，更直观）
    # ax.invert_yaxis()
    ax.grid(True, alpha=0.3)  # 全局网格
    ax.tick_params(axis='both', labelsize=11)

    # 图例与标题
    ax.legend(
        loc='lower left', framealpha=1, shadow=True, fontsize=11,
        ncol=2  # 分2列显示，避免遮挡
    )
    ax.set_title(title, fontweight='bold', pad=15, fontsize=14)

    # 保存与显示
    plt.tight_layout()  # 避免标签重叠
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高清保存
    print(f"\nNMSE曲线图已保存至：{os.path.abspath(save_path)}")
    plt.show()


# ===================== 4. 主执行函数 =====================
if __name__ == "__main__":
    try:
        # 步骤1：读取数据
        snr_dB, nmse_dB_list = load_nmse_data(PT_FILE_PATHS)

        # 步骤2：绘制曲线
        plot_nmse_curve(snr_dB, nmse_dB_list, SAVE_FIG_PATH, PLOT_TITLE)

    except Exception as e:
        print(f"\n❌ 执行失败：{e}")
