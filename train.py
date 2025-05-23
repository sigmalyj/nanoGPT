import os
import time
import math
import pickle
from contextlib import nullcontext
from datetime import datetime  # 用于生成日志文件名

import numpy as np
import torch

# from model import ModelConfig, GPT
from model_RoPE import ModelConfig, GPT

# ----------------------------- Configuration ----------------------------------
# 配置部分，定义了训练的超参数和系统设置

# 输出目录，用于保存模型检查点和日志
out_dir = 'out-wikitext'

# 创建日志文件
current_time = datetime.now().strftime("%m%d_%H%M")
log_file_path = os.path.join(out_dir, f"log_{current_time}.txt")
os.makedirs(out_dir, exist_ok=True)

def log_message(message):
    """
    将消息打印到控制台并写入日志文件。
    """
    print(message)
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

# 评估和日志记录的间隔
eval_interval = 250  # 每隔 250 次迭代进行一次评估
log_interval = 10    # 每隔 10 次迭代打印日志

# 评估设置
eval_iters = 200     # 每次评估计算 200 个批次的平均损失
eval_only = False    # 是否只进行评估而不训练
always_save_checkpoint = True  # 是否始终保存检查点

# 初始化方式：'scratch' 表示从头开始训练，'resume' 表示从检查点恢复
init_from = 'resume'

# 数据相关配置
dataset = 'wikitext'  # 数据集名称
gradient_accumulation_steps = 5 * 8  # 梯度累积步数
batch_size = 12  # 每个批次的样本数
block_size = 1024  # 序列长度

# 模型架构配置
n_layer = 8  # Transformer 层数
n_head = 8   # 注意力头数
n_embd = 512 # 嵌入维度
dropout = 0.2  # Dropout 概率
bias = True    # 是否使用偏置

# 优化器设置
learning_rate = 6e-4  # 初始学习率
max_iters = 20000     # 最大迭代次数
weight_decay = 1e-1   # 权重衰减
beta1 = 0.9           # Adam 优化器的 beta1 参数
beta2 = 0.95          # Adam 优化器的 beta2 参数
grad_clip = 1.0       # 梯度裁剪阈值

# 学习率调度
decay_lr = True       # 是否使用学习率衰减
warmup_iters = 2000   # 预热阶段的迭代次数
lr_decay_iters = 20000 # 学习率衰减的总步数
min_lr = 6e-5         # 最小学习率

# 系统设置
device = 'cuda'  # 设备类型，可设置为 'cuda' 使用 GPU
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # 是否编译模型以加速训练

# 从配置文件中动态覆盖配置
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ----------------------------- Initialization ----------------------------------
# 初始化随机种子和设备设置
seed_offset = 0
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
log_message(f"tokens per iteration will be: {tokens_per_iter:,}")

# 创建输出目录
os.makedirs(out_dir, exist_ok=True)

# 设置随机种子
torch.manual_seed(42 + seed_offset)

# 启用 TF32 加速（如果支持）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 确定设备类型和数据类型
device_type = 'cuda' if 'cuda' in device else 'cpu'
print(f"Using device type: {device_type}")  # 打印设备类型
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# ----------------------------- Dataset Loader ----------------------------------
# 数据加载器，负责从磁盘加载训练和验证数据
data_dir = os.path.join('data', dataset)

def get_batch(split):
    """
    从指定的数据集分割（train/val）中采样一个批次的数据。
    """
    data_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')  # 使用内存映射加载数据
    ix = torch.randint(len(data) - block_size, (batch_size,))  # 随机采样起始索引
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])  # 输入序列
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])  # 目标序列
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)

# ----------------------------- Model Initialization ----------------------------
# 模型初始化，根据配置选择从头开始训练或从检查点恢复
iter_num = 0
best_val_loss = 1e9  # 初始化验证集最优损失为一个较大的值
meta_path = os.path.join(data_dir, 'meta.pkl')

# 模型参数
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50304, dropout=dropout)

if init_from == 'scratch':
    # 从头初始化模型
    log_message("Initializing a new model from scratch")
    gptconf = ModelConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    # 从检查点恢复模型
    log_message(f"Resuming training from {out_dir}")
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    for k in model_args:
        if k in checkpoint['model_args']:
            model_args[k] = checkpoint['model_args'][k]
    model = GPT(ModelConfig(**model_args))
    state_dict = checkpoint['model']
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# 如果 block_size 小于模型的最大序列长度，则裁剪模型
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

# 将模型迁移到指定设备
model.to(device)
raw_model = model

# 配置优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# 如果启用了编译，则编译模型以加速训练
if compile:
    log_message("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# ----------------------------- Evaluation and Training --------------------------
@torch.no_grad()
def estimate_loss():
    """
    评估模型在训练集和验证集上的损失。
    """
    out = {}
    model.eval()  # 切换到评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # 恢复到训练模式
    return out

# 主训练循环中的日志记录部分
if iter_num % eval_interval == 0:
    losses = estimate_loss()
    log_message(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

def get_lr(it):
    """
    根据当前迭代数计算学习率（支持预热和余弦退火）。
    """
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 获取一个训练批次
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0

# 主训练循环
while iter_num <= max_iters:
    # 更新学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 梯度累积逻辑
    optimizer.zero_grad()  # 清空梯度
    for _ in range(gradient_accumulation_steps):
        X, Y = get_batch('train')  # 获取一个小 batch
        logits, loss = model(X, Y)  # 前向传播计算损失
        loss = loss / gradient_accumulation_steps  # 将损失按累积步数归一化
        loss.backward()  # 反向传播计算梯度

    # 梯度裁剪（防止梯度爆炸）
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # 更新模型参数
    optimizer.step()

    # 每隔 eval_interval 次迭代进行评估
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # 如果验证集损失降低或启用了 always_save_checkpoint，则保存检查点
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                torch.save({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    t1 = time.time()
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps  # 恢复到未归一化的损失
        if local_iter_num >= 5:
            log_message(f"iter {iter_num}: loss {lossf:.4f}, time {(t1 - t0)*1000:.2f}ms")  # 使用 log_message 记录日志
    t0 = t1
    
    iter_num += 1
    local_iter_num += 1