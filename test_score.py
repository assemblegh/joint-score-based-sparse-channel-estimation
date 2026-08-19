import numpy as np
import torch
import sys
import os
import copy
import argparse
import scipy.io as sio
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm

sys.path.append('./')

from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from data.loaders import Channels
from torch.utils.data import DataLoader
from utils.util import *
from data.sample_generator import *

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)  # 指定使用的GPU编号
parser.add_argument('--channel', type=str, default='3GPP')  # 测试使用的信道类型
parser.add_argument('--save_channels', type=int, default=0)  # 是否保存信道数据
parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[32 / 8])  # 导频alpha参数
parser.add_argument('--noise_boost', nargs='+', type=float, default=0.1)  # 噪声增强系数
parser.add_argument('--batch_size_x_list', nargs='+', type=float, default=[30])  # 符号批次大小列表
parser.add_argument('--pilots_list', nargs='+', type=float, default=[10,30,50,70])  # 导频数列表
parser.add_argument('--sample_joint', type=bool, default=False)  # 是否联合采样

args = parser.parse_args()

# 初始化日志器
logger = get_logger()

# CUDA配置
torch.cuda.empty_cache()  # 清空CUDA缓存
device = 'cuda:0'  # 测试使用的设备
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);

logger.info(f"设备已设置为 {device}.")

# 加载配置和模型权重
test_seed = 4321  # 测试随机种子
result_dir = 'results_joint_seed%d' % test_seed  # 结果保存目录
# 创建结果目录
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
logger.info(f"结果将保存到 {result_dir}.")

# 目标模型权重路径
target_weights = './models/sigmaT30.0/final_model_3gpp_64_TDLA_S_n.pt'
contents = torch.load(target_weights)  # 加载模型文件

# 加载配置
config = contents['config']
config.sampling.steps_each = 3  # 每个sigma的采样步数
config.data.channel = args.channel  # 测试信道类型
config.model.step_size = 1 * 1e-10  # 模型步长
config.data.mod_n = 4  # 调制阶数

# 初始化模型并加载权重
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()  # 将模型移至GPU
diffuser.load_state_dict(contents['model_state'])  # 加载模型参数
diffuser.eval()  # 设置模型为评估模式

# 设置实验参数
snr_range = np.arange(0,15, 0.5)  # SNR范围(dB)
noise_range = 10 ** (-snr_range / 10.)  # 对应的噪声功率

NR = config.data.image_size[0]  # 接收天线数
NT = config.data.image_size[1]  # 发射天线数

M = int(np.sqrt(config.data.mod_n))  # 调制星座维度

num_channels = 50  # 测试信道数量
# 总采样迭代次数
total_iter = int(config.model.num_classes * config.sampling.steps_each)
noise_boost = args.noise_boost  # 噪声增强系数
logger.info(f"信道尺寸: {NR}x{NT}.")
logger.info(f"总迭代次数: {total_iter}.")

# 加载3GPP信道数据并准备符号估计的数据加载器
mat_contents = sio.loadmat('data/TDL_A_sparse_n.mat')  # 加载信道数据
H = mat_contents['H_bank']
# 选取测试信道样本(从100个用户中选取NT个随机用户)
H_test_complex = torch.tensor(H[9500:9500 + num_channels, :, :]).detach().numpy()
# ===== 新增：识别每个信道样本的活跃用户 =====
col_energy = np.linalg.norm(H_test_complex, axis=1)         # shape: (num_channels, NT)
active_mask_np = (col_energy > 1e-8).astype(np.float32)     # shape: (num_channels, NT)
# 构建实数域掩码（实部 NT 个 + 虚部 NT 个）
active_mask_real_np = np.concatenate([active_mask_np, active_mask_np], axis=1)  # (num_channels, 2*NT)
active_mask_real = torch.tensor(active_mask_real_np).unsqueeze(-1).to(device=device)  # (num_channels, 2*NT, 1)
print(f"前5个样本的活跃用户数: {active_mask_np.sum(axis=1)[:5]}")

# 创建符号生成器
generator = sample_generator(num_channels, config.data.mod_n, NR)

# 将复信道矩阵转换为实数表示（需与 Channels 类的归一化保持一致）
H_train_for_norm = torch.tensor(H[:7500, :, :]).detach().numpy()
#norm_std = np.std(H_test_complex)   # 与 Channels(norm='global') 的计算完全相同
norm_std = np.std(H_train_for_norm)  # 用训练集的std
aux = torch.tensor(H_test_complex / norm_std)
H_test_real_repr = torch.empty([num_channels, 2 * NR, 2 * NT])
H_test_real_repr[:, 0:NR, 0:NT] = torch.real(aux)  # 实部
H_test_real_repr[:, 0:NR, NT:] = torch.imag(aux)  # 虚部
H_test_real_repr[:, NR:, 0:NT] = torch.imag(aux)  # 虚部
H_test_real_repr[:, NR:, NT:] = torch.real(aux)  # 实部
H_test_real_repr[:, :NR, NT:] = -H_test_real_repr[:, :NR, NT:]  # 符号调整

logger.info(f"信道加载完成.")

# 主推理循环
for batch_size_x in args.batch_size_x_list:  # 遍历符号批次大小
    for pilots in args.pilots_list:  # 遍历导频数
        logger.info(f"开始实验，导频数: {pilots}.")
        logger.info(f"开始实验，符号数: {batch_size_x}.")

        # 初始化性能指标
        SER_langevin = []  # 符号错误率(SER)
        # 每SNR每迭代的Oracle误差日志
        oracle_log = np.zeros((len(snr_range), total_iter))
        config.data.num_pilots = pilots  # 设置当前导频数
        print(config.data.num_pilots)

        # 加载导频相关数据
        #dataset_pilots = Channels(test_seed, config, H=H_test_complex, norm="global")
        dataset_pilots = Channels(test_seed, config, H=H_test_complex, norm=[0., norm_std])
        batch_size = len(dataset_pilots)
        # 创建导频数据加载器
        loader = DataLoader(dataset_pilots, batch_size=num_channels,
                            shuffle=False, num_workers=0, drop_last=True)

        iter_ = iter(loader)  # 数据迭代器
        samples_pilots = next(iter_)  # 获取导频样本
        # 分离信道、导频和接收信号
        _, pilots, _ = samples_pilots['H'].cuda(), samples_pilots['P'].cuda(), samples_pilots['Y'].cuda()

        # 计算导频的共轭转置
        pilots_conj = torch.conj(torch.transpose(pilots, -1, -2))
        H_herm = samples_pilots['H_herm'].cuda()  # 厄米特信道矩阵
        # 转换为复数形式
        H_herm_complex = H_herm[:, 0] + 1j * H_herm[:, 1]

        # 遍历所有SNR值
        for snr_idx, local_noise in enumerate(noise_range):

            # 为每个SNR设置参数
            iter_lang = 0  # Langevin迭代计数器
            # 创建单位矩阵
            Id = batch_identity_matrix(2 * NR, 2 * NR, batch_size).to(device=device)
            # 根据SNR调整参数
            if snr_range[snr_idx] < 5:
                temp_x = 0.5  # 温度参数
                sigmas_x = np.linspace(0.6, 0.01, config.model.num_classes)  # x的sigma序列
                epsilon = 1E-4  # x的步长系数
            else:
                temp_x = 0.1
                sigmas_x = np.linspace(0.8, 0.01, config.model.num_classes)
                epsilon = 4E-5

            # 准备导频相关的接收数据
            y_pilots = torch.matmul(pilots_conj, H_herm_complex)  # 导频接收信号
            # 添加高斯噪声
            y_pilots = y_pilots + np.sqrt(local_noise) * torch.randn_like(y_pilots)
            H_current = torch.randn_like(H_herm_complex)  # 初始化信道估计值
            oracle = H_herm_complex  # 真实信道值(用于误差计算)
            H_list = []  # 存储信道估计结果

            # 准备符号相关的接收数据
            # 初始化符号估计值
            x_current = torch.randn(num_channels, 2 * NT, batch_size_x).to(device=device)
            x_current = x_current * active_mask_real  # 非活跃用户初始就置零

            # 生成随机符号索引
            indices = generator.random_indices(NT, batch_size_x * num_channels)
            j_indices = generator.joint_indices(indices)  # 联合索引
            x_true = generator.modulate(indices)  # 调制生成真实符号
            # 调整维度
            x_true = torch.reshape(x_true, (num_channels, batch_size_x, 2 * NT))
            x_true = torch.transpose(x_true, -1, -2)
            # ===== 新增：非活跃用户不发送数据，x_true 强制置零 =====
            x_true = x_true.to(device=device) * active_mask_real

            # 计算符号接收信号
            y_x = torch.matmul(H_test_real_repr.double().to(device=device), x_true.double()).float()

            # 添加高斯噪声
            y_x = y_x + np.sqrt(local_noise) * torch.randn_like(y_x).to(device=device)
            H_current_x = torch.zeros([num_channels, 2 * NR, 2 * NT]).to(device=device)   # 初始化实数形式的信道

            # 创建联合测量向量
            y_x_complex = y_x.chunk(2, dim=1)  # 拆分为实部和虚部
            y_x_complex = (y_x_complex[0] - 1j * y_x_complex[1])  # 转换为复数
            y_x_complex = torch.transpose(y_x_complex, -1, -2)  # 转置
            # 拼接导频和符号接收信号
            y_H = torch.cat((y_pilots.to(device=device), y_x_complex.to(device=device)), dim=1)

            with torch.no_grad():  # 禁用梯度计算
                # 遍历每个sigma级别
                for step_idx in tqdm(range(config.model.num_classes)):
                    # 获取当前sigma值
                    current_sigma = diffuser.sigmas[step_idx].item()
                    current_sigma_x = sigmas_x[step_idx]

                    # 生成扩散模型的标签
                    labels = torch.ones(H_current.shape[0]).cuda() * step_idx
                    labels = labels.long()

                    # 计算当前步长(动态调整)
                    step_H = config.model.step_size * \
                             (current_sigma / config.model.sigma_end) ** 2  # 信道步长
                    # 符号步长
                    step_x = epsilon * \
                             (current_sigma_x / sigmas_x[-1]) ** 2

                    # 每个sigma级别内的多次迭代
                    for inner_idx in range(config.sampling.steps_each):

                        # 将复数信道转换为非厄米特实数形式
                        H_current_nonHerm = torch.transpose(torch.conj(H_current), 2, 1).to(device=device)
                        H_current_x[:, 0:NR, 0:NT] = torch.real(H_current_nonHerm)
                        H_current_x[:, 0:NR, NT:] = torch.imag(H_current_nonHerm)
                        H_current_x[:, NR:, 0:NT] = torch.imag(H_current_nonHerm)
                        H_current_x[:, NR:, NT:] = torch.real(H_current_nonHerm)
                        H_current_x[:, :NR, NT:] = -H_current_x[:,:NR, NT:]

                        # ------------------------#
                        # 符号的Langevin采样     #
                        # ------------------------#

                        # 初始化梯度
                        grad = torch.zeros((num_channels, 2 * NT, batch_size_x)).to(device=device)
                        # 先验分布的分数
                        x_gaussian = torch.transpose(x_current, 2, 1)
                        Zi_hat = gaussian(x_gaussian.reshape(batch_size_x * num_channels, 2 * NT), generator,
                                          current_sigma_x ** 2, NT, M, device)
                        Zi_hat = torch.transpose(torch.reshape(Zi_hat, (num_channels, batch_size_x, 2 * NT)), 2, 1)
                        prior = (Zi_hat - x_current) / current_sigma_x ** 2  # 先验分数

                        # 似然分布的分数
                        diff = (y_x - torch.matmul(H_current_x.to(device=device), x_current))  # 接收信号误差
                        # 计算协方差矩阵

                        cov_matrix = (current_sigma_x ** 2) * torch.bmm(H_current_x, H_current_x.transpose(2, 1)) \
                                     + local_noise * Id \
                                     + 1e-4 * Id  # ← 加正则化
                        cov_matrix = torch.inverse(cov_matrix.to(device=device))

                        # 似然梯度
                        grad_likelihood = torch.matmul(cov_matrix, diff.float()).to(device=device)
                        grad_likelihood = torch.matmul(torch.transpose(H_current_x, 2, 1).to(device=device),
                                                       grad_likelihood)
                        del cov_matrix  # 释放内存

                        # 后验分数 = 似然分数 + 先验分数
                        grad = grad_likelihood + prior

                        # 生成Langevin噪声
                        noise = np.sqrt(2 * temp_x * step_x) * torch.randn(num_channels, 2 * NT, batch_size_x).to(
                            device=device)

                        # 更新符号估计值
                        x_current = x_current + step_x * grad + noise
                        x_current = x_current * active_mask_real  # 防止符号先验把非活跃用户拉离零

                        # ------------------------#
                        # 信道的Langevin采样     #
                        # ------------------------#

                        # 先验分布的分数
                        # 将复数信道转换为实数表示
                        current_real = torch.view_as_real(H_current).permute(0, 3, 1, 2)
                        # 获取模型预测的分数
                        score = diffuser(current_real, labels)
                        # 转换回复数形式
                        score = torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())

                        # 似然分布的分数
                        if args.sample_joint == True:  # 联合采样
                            # 将符号转换为复数形式
                            forward_complex = x_current.chunk(2, dim=1)
                            forward_complex = (forward_complex[0] - 1j * forward_complex[1])
                            forward_complex = torch.transpose(forward_complex, -1, -2)
                            # 拼接导频和符号
                            forward_H = torch.cat((pilots_conj.to(device=device), forward_complex.to(device=device)),
                                                  dim=1)
                            forward_herm = torch.conj(torch.transpose(forward_H, -1, -2)).to(device=device)
                            # 计算测量梯度
                            meas_grad = torch.matmul(forward_herm,
                                                     torch.matmul(forward_H, H_current.to(device=device)) - y_H
                                                     )
                        else:  # 仅使用导频
                            meas_grad = torch.matmul(pilots,
                                                     torch.matmul(pilots_conj, H_current.to(device=device)) - y_pilots
                                                     )

                        # 生成信道更新的噪声
                        grad_noise = np.sqrt(2 * step_H * noise_boost) * torch.randn_like(H_current)

                        # 更新信道估计值
                        H_current = H_current.to(device=device) \
                                    + step_H * (score.to(device=device) - meas_grad.to(device=device) / (
                                    local_noise / 2. + current_sigma ** 2)) \
                                    + grad_noise.to(device=device)

                        # 计算并存储信道估计误差
                        oracle_log[snr_idx, iter_lang] = \
                            torch.mean((torch.sum(
                                torch.square(torch.abs(H_current.to(device='cpu') - oracle.to(device='cpu'))),
                                dim=(-1, -2)) / \
                                        torch.sum(torch.square(torch.abs(oracle.to(device='cpu'))),
                                                  dim=(-1, -2)))).cpu().numpy()
                        iter_lang = iter_lang + 1

            # 存储信道估计结果
            H_list.append(H_current_x)
            # 计算符号错误率
            # ===== 只对活跃用户计算 SER =====
            x_est = torch.transpose(x_current, -1, -2).reshape(num_channels * batch_size_x, 2 * NT).to(device='cpu')

            # 将每个样本的活跃用户掩码展开到 (num_channels * batch_size_x, NT)
            # 注意 j_indices 的排列顺序：reshape(num_channels, batch_size_x, NT) 的展平
            active_flat_np = np.tile(active_mask_np[:, None, :],
                                     (1, batch_size_x, 1))  # (num_channels, batch_size_x, NT)
            active_flat_np = active_flat_np.reshape(num_channels * batch_size_x, NT).astype(bool)
            active_flat = torch.tensor(active_flat_np)  # bool 张量

            # 复用 sym_detection 的判决逻辑，但加入掩码
            x_real, x_imag = torch.chunk(x_est, 2, dim=-1)  # 各 (N_total, NT)
            x_real_exp = x_real.unsqueeze(-1).expand(-1, -1, generator.real_QAM_const.numel())
            x_imag_exp = x_imag.unsqueeze(-1).expand(-1, -1, generator.imag_QAM_const.numel())
            x_dist = (x_real_exp - generator.real_QAM_const) ** 2 + (x_imag_exp - generator.imag_QAM_const) ** 2
            x_indices = torch.argmin(x_dist, dim=-1)  # (N_total, NT)

            # 只在活跃位置比较
            correct_mask = (x_indices == j_indices) & active_flat
            n_correct = correct_mask.sum().item()
            n_total = active_flat.sum().item()

            SER_langevin.append(1 - n_correct / max(n_total, 1))

            # 打印当前SNR的最终信道估计误差
            print(snr_range[snr_idx], 10 * np.log10(oracle_log[:, -1]))

        # 清空CUDA缓存
        torch.cuda.empty_cache()

        # 保存实验结果
        save_dict = {
            'snr_range': snr_range,  # SNR范围
            'config': config,  # 配置参数
            'oracle_log': oracle_log,  # 信道估计误差日志
            'H_val_complex': H_test_complex,  # 测试复数信道
            'H_symbols_batch': H_test_real_repr,  # 实数形式的信道
            'H_current_x': H_list,  # 信道估计结果
            'SER_langevin': SER_langevin  # 符号错误率
        }
        # 保存结果文件
        torch.save(save_dict,
                   result_dir + '/%s_numpilots%.1f_numsymbols%.1f_TDLA_S_n_pilot.pt' % (
                   args.channel, config.data.num_pilots, batch_size_x))
