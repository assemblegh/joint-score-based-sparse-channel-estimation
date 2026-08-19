"""
读取多个numsymbol的.pt文件，在同一张图中绘制4条SER曲线（NMSE可选）
适配文件：3GPP_numpilots30.0_numsymbols10/30/50/100_TDLA.pt
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# ===================== 1. 基础配置（按需修改） =====================
# 4个.pt文件的路径（根据你的实际路径调整！）
PT_FILE_PATHS = [
    "results_joint_seed4321/3GPP_numpilots10.0_numsymbols30.0_TDLA_S_n_pilot.pt",
    "results_joint_seed4321/3GPP_numpilots30.0_numsymbols30.0_TDLA_S_n_pilot.pt",
    "results_joint_seed4321/3GPP_numpilots50.0_numsymbols30.0_TDLA_S_n_pilot.pt",
    "results_joint_seed4321/3GPP_numpilots70.0_numsymbols30.0_TDLA_S_n_pilot.pt"
]
# 对应4个文件的numsymbol标签（用于图例）
NUMSYMBOL_LABELS = ["10", "30", "50", "70"]
SAVE_FIG_PATH = "figures/nmse_snr_multiple_S_numsybols=30.png"  # 保存路径
PLOT_TITLE = "SER (dB) and SNR (numsybols=30,sparse,0.1)"  # 标题


# 可选：中文显示（取消注释）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# ===================== 2. 批量读取并解析.pt文件 =====================
def load_multiple_results(pt_file_paths):
    """
    批量解析多个.pt文件，返回：
    snr_dB: 公共的SNR横坐标
    ser_list: 4个numsymbol对应的SER列表
    nmse_dB_list: 4个numsymbol对应的NMSE(dB)列表（可选）
    """
    snr_dB = None
    ser_list = []
    nmse_dB_list = []

    for idx, pt_path in enumerate(pt_file_paths):
        # 检查文件存在性
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"未找到文件：{pt_path}")

        # 读取.pt文件（兼容CPU）
        try:
            save_dict = torch.load(pt_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"读取文件{pt_path}失败：{e}")

        # 提取核心数据
        current_snr = np.array(save_dict["snr_range"])
        current_ser = np.array(save_dict["SER_langevin"])
        oracle_log = np.array(save_dict["oracle_log"])
        current_nmse = oracle_log[:, -1] + 1e-10  # 加微小值避免log10(0)
        current_nmse_dB = 10 * np.log10(current_nmse)

        # 校验所有文件的SNR维度一致
        if snr_dB is None:
            snr_dB = current_snr
        else:
            if not np.array_equal(snr_dB, current_snr):
                raise ValueError(f"文件{pt_path}的SNR范围与其他文件不一致！")

        # 校验SER维度
        if len(snr_dB) != len(current_ser):
            raise ValueError(f"文件{pt_path}：SNR长度{len(snr_dB)} ≠ SER长度{len(current_ser)}")

        # 存储当前文件的SER和NMSE
        ser_list.append(current_ser)
        nmse_dB_list.append(current_nmse_dB)

        # 打印当前文件的数据概览
        print(f"=== 数据概览（numpilots={NUMSYMBOL_LABELS[idx]}）===")
        print(f"SNR (dB)：{snr_dB}")
        print(f"SER：{current_ser.round(4)}")
        print(f"NMSE (dB)：{current_nmse_dB.round(2)}\n")

    return snr_dB, ser_list, nmse_dB_list


# ===================== 3. 绘制多曲线性能图 =====================
def plot_multiple_ser(snr_dB, ser_list, nmse_dB_list, save_path, title):
    """
    在同一张图中绘制4条SER曲线（核心），可选绘制NMSE曲线
    """
    # 绘图风格配置（论文级美观度）
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'grid.alpha': 0.3,
        'figure.figsize': (9, 6)  # 加宽画布适配4条曲线
    })

    # 定义4种颜色和标记（区分度高）
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']  # 红、绿、蓝、橙
    markers = ['o', 's', '^', '*']  # 圆、方、三角、星号

    # 创建画布（仅绘制SER，如需NMSE可改为双轴）
    fig, ax1 = plt.subplots()

    # -------- 绘制4条SER曲线 --------
    for idx, ser in enumerate(ser_list):
        ax1.plot(
            snr_dB, ser,
            marker=markers[idx], markersize=7, markeredgecolor='white', markeredgewidth=1.5,
            linewidth=2.5, color=colors[idx],
            label=f'numpilots = {NUMSYMBOL_LABELS[idx]}'
        )

    # -------- 坐标轴配置 --------
    ax1.set_xlabel('SNR (dB)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('SER', fontweight='bold', fontsize=13)
    ax1.set_yscale('log')  # SER必须用对数坐标
    ax1.grid(True, which="both", axis="y", alpha=0.3)
    ax1.tick_params(axis='both', labelsize=11)

    # -------- 图例与标题：修改为左下角 --------
    ax1.legend(
        loc='lower left', framealpha=1, shadow=True, fontsize=11,
        ncol=2  # 图例分2列，避免占空间
    )
    ax1.set_title(title, fontweight='bold', pad=15, fontsize=14)

    # -------- 保存与显示 --------
    plt.tight_layout()  # 避免标签重叠
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存至：{os.path.abspath(save_path)}")
    plt.show()

    # 【可选】绘制NMSE多曲线（如需对比NMSE，取消以下注释）
    # plot_multiple_nmse(snr_dB, nmse_dB_list, colors, markers)


# ===================== 可选：绘制NMSE多曲线 =====================
def plot_multiple_nmse(snr_dB, nmse_dB_list, colors, markers):
    """单独绘制NMSE(dB)多曲线"""
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, nmse_dB in enumerate(nmse_dB_list):
        ax.plot(
            snr_dB, nmse_dB,
            marker=markers[idx], markersize=7, markeredgecolor='white', markeredgewidth=1.5,
            linewidth=2.5, color=colors[idx],
            label=f'numsymbol = {NUMSYMBOL_LABELS[idx]}'
        )
    ax.set_xlabel('SNR (dB)', fontweight='bold', fontsize=13)
    ax.set_ylabel('NMSE (dB)', fontweight='bold', fontsize=13)
    ax.set_title("NMSE (dB) vs SNR (Different Number of Symbols, Pilots=30,sparse)", fontweight='bold', pad=15, fontsize=14)
    ax.legend(loc='lower left', framealpha=1, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/nmse_multiple_numsymbols.png", dpi=300, bbox_inches='tight')
    plt.show()


# ===================== 4. 主执行函数 =====================
if __name__ == "__main__":
    try:
        # 步骤1：批量读取并解析4个文件
        snr_dB, ser_list, nmse_dB_list = load_multiple_results(PT_FILE_PATHS)

        # 步骤2：绘制4条SER曲线
        plot_multiple_ser(snr_dB, ser_list, nmse_dB_list, SAVE_FIG_PATH, PLOT_TITLE)

        # 可选：绘制NMSE多曲线
        # colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']
        # markers = ['o', 's', '^', '*']
        # plot_multiple_nmse(snr_dB, nmse_dB_list, colors, markers)

    except Exception as e:
        print(f"\n❌ 执行失败：{e}")
