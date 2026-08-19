import torch
import numpy as np
import matplotlib.pyplot as plt

# ====================== 你的 .pt 模型路径（改成你自己的） ======================
pt_file = "models/sigmaT30.0/final_model_3gpp_64_TDLA_S_n.pt"
# =============================================================================

# 加载保存的 checkpoint
checkpoint = torch.load(pt_file, map_location="cpu")

# 直接读取里面保存的 loss
train_loss = checkpoint["train_loss"]
val_loss = checkpoint["val_loss"]


# 1. 打印基础信息
print(f"train_loss 长度: {len(train_loss)}, 类型: {type(train_loss)}")
print(f"val_loss 长度: {len(val_loss)}, 类型: {type(val_loss)}")

# 2. 打印前5个val_loss元素（看是否为空/异常）
if len(val_loss) > 0:
    print("val_loss 前5个元素:", val_loss[:5])
else:
    print("val_loss 为空！")

# 3. 检查val_loss_flat的最终状态
val_loss_flat = []
for v in val_loss:
    if isinstance(v, list):
        val_loss_flat.extend(v)
    else:
        val_loss_flat.append(v)
val_loss_flat = np.array(val_loss_flat)

print(f"val_loss_flat 长度: {len(val_loss_flat)}")
print(f"val_loss_flat 是否包含NaN: {np.isnan(val_loss_flat).any()}")
print(f"val_loss_flat 是否包含Inf: {np.isinf(val_loss_flat).any()}")
print(f"val_loss_flat 数值范围: {np.min(val_loss_flat) if len(val_loss_flat)>0 else '无'} ~ {np.max(val_loss_flat) if len(val_loss_flat)>0 else '无'}")
print(f"训练步数：{len(train_loss)}")
print(f"验证点数：{len(val_loss)}")

# 把 val_loss 展平成一维
val_loss_flat = []
for v in val_loss:
    if isinstance(v, list):
        val_loss_flat.extend(v)
    else:
        val_loss_flat.append(v)
val_loss_flat = np.array(val_loss_flat)

# ====================== 画图 ======================
plt.figure(figsize=(14, 6))

# 左图：训练 Loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, color='#4A79E0', linewidth=0.6, label='Train Loss')
plt.title("Training Loss", fontsize=14)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.legend()

# 右图：验证 Loss
plt.subplot(1, 2, 2)
plt.plot(val_loss_flat, color='#FF5A5A', linewidth=2, marker='o', markersize=3, label='Val Loss')
plt.title("Validation Loss", fontsize=14)
plt.xlabel("Val Steps (every 100 train steps)")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=300)
plt.show()
