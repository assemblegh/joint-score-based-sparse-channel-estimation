
import numpy as np
import torch
import sys
import os
import argparse
import copy

sys.path.append('./')

from tqdm import tqdm
import scipy.io as sio
import random
from dotmap import DotMap
from torch.utils.data import DataLoader

from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from ncsnv2.losses import get_optimizer
from ncsnv2.losses.dsm import anneal_dsm_score_estimation

from data.loaders import Channels
from data.sample_generator import *

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)  # 指定使用的GPU编号
parser.add_argument('--train', type=str, default='3GPP')  # 训练使用的信道类型
args = parser.parse_args()

# 全局CUDA设置
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
# GPU设备配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

# 模型配置参数
config = DotMap()
config.device = 'cuda:0'  # 训练使用的设备
config.use_amp = True
# 内部模型配置
config.model.ema = True  # 是否使用指数移动平均(EMA)
config.model.ema_rate = 0.999  # EMA的衰减率
config.model.normalization = 'InstanceNorm++'  # 归一化方式
config.model.nonlinearity = 'elu'  # 非线性激活函数
config.model.sigma_dist = 'geometric'  # sigma分布类型
config.model.num_classes = 2311  # 训练sigma的数量和'N'值
config.model.ngf = 32  # 生成器特征图数量

# 优化器配置
config.optim.weight_decay = 0.000  # 权重衰减系数(无衰减)
config.optim.optimizer = 'Adam'  # 优化器类型
config.optim.lr = 0.0001  # 学习率
config.optim.beta1 = 0.9  # Adam优化器的beta1参数
config.optim.amsgrad = False  # 是否使用AMSGrad变体
config.optim.eps = 0.001  # Adam优化器的epsilon参数

# 训练配置
config.training.batch_size = 32  # 批次大小
config.training.num_workers = 4  # 数据加载的线程数
config.training.n_epochs = 40  # 训练轮数
config.training.anneal_power = 2  # 退火幂次
config.training.log_all_sigmas = False  # 是否记录所有sigma的日志
config.training.eval_freq = 50  # 验证频率(按轮数计)

# 数据配置
config.data.channel = args.train  # 训练和验证使用的信道类型
config.data.channels = 2  # 信道维度{实部, 虚部}
config.data.num_pilots = 8  # 导频数
config.data.noise_std = 0.01  # Beta参数(噪声标准差)
config.data.image_size = [8, 100]  # 信道尺寸 = 接收天线数(Nr) x 发射天线数(Nt)
config.data.mixed_channels = False  # 是否混合不同信道
config.data.norm_channels = 'global'  # 信道归一化方式

# 全局随机种子(保证可复现性)
train_seed, val_seed = 1234, 4321

# 加载数据集并创建数据加载器
train_samples = 7500  # 训练样本数量
mat_contents = sio.loadmat('data/TDL_A_sparse_n.mat')  # 加载.mat格式的信道数据
H = mat_contents['H_bank']
# 选取训练样本并转换为numpy数组(从100个用户中选取NT个随机用户)
H_complex = torch.tensor(H[:train_samples, :, :]).detach().numpy()

# 创建训练数据集
dataset = Channels(train_seed, config, H=H_complex, norm=config.data.norm_channels)
# 创建训练数据加载器
dataloader = DataLoader(dataset, batch_size=config.training.batch_size,
                        shuffle=True, num_workers=config.training.num_workers,
                        drop_last=True)

# 验证数据处理
# 选取验证样本(从100个用户中选取NT个随机用户)
H_val_complex = torch.tensor(H[train_samples:9500, :, :]).detach().numpy()
val_samples = H_val_complex.shape[0]  # 验证样本数量

# 复制配置用于验证
val_config = copy.deepcopy(config)
# 创建验证数据集
val_datasets = Channels(val_seed, val_config, H=H_val_complex, norm=config.data.norm_channels)
# 创建验证数据加载器(批次大小为整个验证集)
val_loaders = DataLoader(val_datasets,
                         batch_size=len(val_datasets),
                         shuffle=False,
                         num_workers=0,
                         drop_last=True)
val_iters = iter(val_loaders)  # 验证数据迭代器


# 预定义的sigma参数
config.model.sigma_begin = 30  # sigma初始值(针对CDL-C信道)
config.model.sigma_rate = 0.995  # sigma衰减率(通用)
# sigma终止值(通过衰减率计算)
config.model.sigma_end = config.model.sigma_begin * config.model.sigma_rate ** (config.model.num_classes - 1)

# 根据Song的方法选择步长(epsilon)
candidate_steps = np.logspace(-13, -8, 1000)  # 候选步长范围
step_criterion = np.zeros((len(candidate_steps)))  # 步长选择准则
gamma_rate = 1 / config.model.sigma_rate  # gamma率
for idx, step in enumerate(candidate_steps):
    step_criterion[idx] = (1 - step / config.model.sigma_end ** 2) \
                          ** (2 * config.model.num_classes) * (gamma_rate ** 2 -
                                                               2 * step / (
                                                                           config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
                                                                           1 - step / config.model.sigma_end ** 2) ** 2)) + \
                          2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
            1 - step / config.model.sigma_end ** 2) ** 2)
best_idx = np.argmin(np.abs(step_criterion - 1.))  # 找到最优步长索引
config.model.step_size = candidate_steps[best_idx]  # 设置最优步长

# 初始化模型
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()  # 将模型移至GPU

# 初始化优化器
optimizer = get_optimizer(config, diffuser.parameters())

# 训练计数器
start_epoch = 0
step = 0

# 初始化EMA助手
if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(diffuser)

# 获取sigma值集合
sigmas = get_sigmas(config)

# 为验证准备固定的初始点和数据
val_H_list = []
val_sample = next(val_iters)  # 获取验证样本
val_H_list.append(val_sample['H_herm'].cuda())

# 日志路径配置
config.log_path = 'models/sigmaT%.1f' % (config.model.sigma_begin)

# 创建日志目录(如果不存在)
if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)

# 是否禁用sigma日志
hook = test_hook = None

# 记录的指标
train_loss, val_loss = [], []  # 训练损失和验证损失
val_errors, val_epoch = [], []  # 验证误差和对应的轮数

# 主训练循环
for epoch in tqdm(range(start_epoch, config.training.n_epochs)):
    for i, sample in tqdm(enumerate(dataloader)):
        # 安全检查: 设置模型为训练模式
        diffuser.train()
        step += 1

        # 将数据移至指定设备
        for key in sample:
            sample[key] = sample[key].cuda()

        # 计算厄米特信道的损失
        loss = anneal_dsm_score_estimation(
            diffuser, sample['H_herm'], sigmas, None,
            config.training.anneal_power, hook)#anneal_power=2是平方


        # 计算运行损失(滑动平均)
        if step == 1:
            running_loss = loss.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.item()
        # 记录训练损失
        train_loss.append(loss.item())

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # EMA更新
        if config.model.ema:
            ema_helper.update(diffuser)

        # 每100步进行验证和日志输出
        if step % 100 == 0:
            # 使用EMA模型进行验证(如果启用)
            if config.model.ema:
                val_score = ema_helper.ema_copy(diffuser)
            else:
                val_score = diffuser

            # 计算每个验证配置的损失
            local_val_losses = []
            # 计算验证集损失
            with torch.no_grad():
                val_dsm_loss = anneal_dsm_score_estimation(
                    val_score, val_H_list[0],
                    sigmas, None,
                    config.training.anneal_power,
                    hook=test_hook)
            local_val_losses = [val_dsm_loss.item()]

            # 清理验证模型
            del val_score
            # 记录验证损失
            val_loss.append(local_val_losses)

            # 打印日志
            if len(local_val_losses) == 1:
                print('第%d轮, 第%d步, 训练损失(EMA) %.3f, 验证损失 %.3f' % (epoch, step, running_loss,
                                                                             local_val_losses[0]))
            elif len(local_val_losses) == 2:
                print('第%d轮, 第%d步, 训练损失(EMA) %.3f, 验证损失(拆分) %.3f %.3f' % (epoch, step, running_loss,
                                                                                        local_val_losses[0],
                                                                                        local_val_losses[1]))

# 保存模型快照
torch.save({'model_state': diffuser.state_dict(),  # 模型参数
            'optim_state': optimizer.state_dict(),  # 优化器参数
            'config': config,  # 配置参数
            'train_loss': train_loss,  # 训练损失
            'val_loss': val_loss,  # 验证损失
            'val_errors': val_errors,  # 验证误差
            'val_epoch': val_epoch},  # 验证轮数
           os.path.join(config.log_path, 'final_model_3gpp_64_TDLA_S_n.pt'))  # 保存路径
