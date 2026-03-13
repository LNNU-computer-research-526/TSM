# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
import numpy as np
import os
import matplotlib.pyplot as plt

plt.switch_backend('Agg')  # 非交互式后端，适合服务器运行

ROOT_SAVE_DIR = r"E:\map\modeltap\ovavesupqvxian"
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)


# ===================== 新增：Loss/准确率曲线绘制函数 =====================
def normalize_data(data):
    """将数据归一化到0-1范围（min-max归一化）"""
    if len(data) == 0:  # 新增：空数据直接返回空数组
        return np.array([])
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:  # 避免除以0
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)
# ===================== 可视化工具函数 =====================
def create_epoch_dir(base_path, epoch):
    """创建轮次文件夹"""
    epoch_dir = os.path.join(base_path, f"epoch{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    return epoch_dir


import matplotlib.colors as mcolors  # 新增导入


def plot_loss_acc_curve(train_loss_list, val_acc_list, epoch=None, save_path=None):
    """
    Plot normalized training loss and validation accuracy curves

    Parameters:
    - train_loss_list: List of training loss values
    - val_acc_list: List of validation accuracy values (percentage, e.g., 85.5 = 85.5%)
    - epoch: Current training epoch (for file naming)
    - save_path: Custom save path (overrides default)

    Returns:
    - None (saves plot to file)
    """
    # 1. Set save path
    if save_path is None:
        if epoch is not None:
            save_path = os.path.join(ROOT_SAVE_DIR, f"loss_acc_curve_epoch{epoch}.png")
        else:
            save_path = os.path.join(ROOT_SAVE_DIR, "loss_acc_curve_weak.png")

    # 2. Data preprocessing
    loss_np = np.array(train_loss_list)
    val_acc_np = np.array(val_acc_list) / 100.0  # Convert accuracy to 0-1

    # 3. Normalize loss to 0-1
    loss_norm = normalize_data(loss_np)

    # 4. Create x-axis values
    epochs_loss = range(1, len(loss_norm) + 1)
    val_epoch_step = max(1, len(epochs_loss) // len(val_acc_np)) if len(val_acc_np) > 0 else 1
    epochs_acc = range(val_epoch_step, len(val_acc_np) * val_epoch_step + 1, val_epoch_step)
    epochs_acc = epochs_acc[:len(val_acc_np)]

    # 5. Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training loss (red)
    ax.plot(epochs_loss, loss_norm,
            label='Training Loss (Normalized)',
            color='red', linewidth=2, marker='o', markersize=4)

    # Plot validation accuracy (blue)
    ax.plot(epochs_acc, val_acc_np,
            label='Validation Accuracy',
            color='blue', linewidth=2, marker='s', markersize=4)

    # 6. Plot configuration (ALL ENGLISH)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Value (0-1)')

    if epoch is not None:
        ax.set_title(f'Training Loss vs Validation Accuracy (Up to Epoch {epoch})')
    else:
        ax.set_title('Training Loss vs Validation Accuracy (All Epochs)')

    ax.set_xlim(0, len(epochs_loss))
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, len(epochs_loss) + 1, step=max(1, len(epochs_loss) // 10)))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 7. Save and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Explicitly close figure to free memory

    # 8. Log success (ASCII only)
    epoch_str = str(epoch) if epoch else "all"
    print(f"[INFO] Loss/Accuracy curve (up to Epoch {epoch_str}) saved to: {save_path}")

def visualize_similarity_matrix(epoch, sample_idx,
                                vis_feat_before, vis_feat_after,
                                audio_feat_before, audio_feat_after,
                                save_path):
    def compute_row_normalized_similarity(feat):
        feat = feat.cpu().detach().numpy()
        T = min(feat.shape[0], 10)
        feat = feat[:T, :]
        if T < 10:
            feat = np.pad(feat, ((0, 10 - T), (0, 0)), 'constant')

        feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        sim_matrix = np.dot(feat_norm, feat_norm.T)
        # 温度系数增强区分度
        temperature = 0.1
        sim_scaled = sim_matrix / temperature
        sim_exp = np.exp(sim_scaled - sim_scaled.max(axis=1, keepdims=True))
        sim_normalized = sim_exp / (sim_exp.sum(axis=1, keepdims=True) + 1e-8)
        return sim_normalized

    def add_values_to_heatmap(ax, matrix, fontsize=7):
        for i in range(matrix.shape[0]):
            if i < 2:
                row_sum = matrix[i].sum()
                print(f"Sample {sample_idx + 1} - Row {i} Sum: {row_sum:.4f}")
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                # 动态判断文字颜色：深色背景（数值大）→白色，浅色→黑色
                text_color = 'white' if value > 0.3 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                        fontsize=fontsize, color=text_color, fontweight='bold')

    # ========== 1. 计算归一化相似度矩阵 ==========
    vis_sim_before = compute_row_normalized_similarity(vis_feat_before)
    vis_sim_after = compute_row_normalized_similarity(vis_feat_after)
    audio_sim_before = compute_row_normalized_similarity(audio_feat_before)
    audio_sim_after = compute_row_normalized_similarity(audio_feat_after)

    # ========== 2. 配置独立的颜色映射和归一化 ==========
    # 视觉：蓝色系（Blues）+ 0.3为中心增强对比
    cmap_vis = plt.cm.Blues
    norm_vis = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.3, vmax=1)

    # 音频：红色系（Reds）+ 0.3为中心增强对比（独立归一化）
    cmap_audio = plt.cm.Reds
    norm_audio = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.3, vmax=1)

    # ========== 3. 创建子图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Epoch {epoch} - Sample {sample_idx + 1}: Cross-Modal Alignment Comparison\n'
                 f'(Row Sum = 1.0 | Visual: Blue Scale, Audio: Red Scale)',
                 fontsize=14, fontweight='bold')

    # ===================== 视觉模态（蓝色系，独立色条） =====================
    # 视觉-Before GNN
    im1 = axes[0, 0].imshow(vis_sim_before, cmap=cmap_vis, norm=norm_vis)
    axes[0, 0].set_title(f'Visual (Before GNN) | Row Sum: {vis_sim_before[0].sum():.2f}',
                         fontsize=11, color='darkblue')
    axes[0, 0].set_xlabel('Time Step (Target)')
    axes[0, 0].set_ylabel('Time Step (Source)')
    axes[0, 0].set_xticks(range(10))
    axes[0, 0].set_yticks(range(10))
    add_values_to_heatmap(axes[0, 0], vis_sim_before)
    # 独立颜色条
    cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar1.set_label('Visual Attention (Large → Dark Blue)', fontsize=9, color='darkblue')

    # 视觉-After GNN
    im2 = axes[0, 1].imshow(vis_sim_after, cmap=cmap_vis, norm=norm_vis)
    axes[0, 1].set_title(f'Visual (After GNN) | Row Sum: {vis_sim_after[0].sum():.2f}',
                         fontsize=11, color='darkblue')
    axes[0, 1].set_xlabel('Time Step (Target)')
    axes[0, 1].set_ylabel('Time Step (Source)')
    axes[0, 1].set_xticks(range(10))
    axes[0, 1].set_yticks(range(10))
    add_values_to_heatmap(axes[0, 1], vis_sim_after)
    # 独立颜色条
    cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar2.set_label('Visual Attention (Large → Dark Blue)', fontsize=9, color='darkblue')

    # ===================== 音频模态（红色系，独立色条） =====================
    # 音频-Before GNN
    im3 = axes[1, 0].imshow(audio_sim_before, cmap=cmap_audio, norm=norm_audio)
    axes[1, 0].set_title(f'Audio (Before GNN) | Row Sum: {audio_sim_before[0].sum():.2f}',
                         fontsize=11, color='darkred')
    axes[1, 0].set_xlabel('Time Step (Target)')
    axes[1, 0].set_ylabel('Time Step (Source)')
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_yticks(range(10))
    add_values_to_heatmap(axes[1, 0], audio_sim_before)
    # 独立颜色条
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar3.set_label('Audio Attention (Large → Dark Red)', fontsize=9, color='darkred')

    # 音频-After GNN
    im4 = axes[1, 1].imshow(audio_sim_after, cmap=cmap_audio, norm=norm_audio)
    axes[1, 1].set_title(f'Audio (After GNN) | Row Sum: {audio_sim_after[0].sum():.2f}',
                         fontsize=11, color='darkred')
    axes[1, 1].set_xlabel('Time Step (Target)')
    axes[1, 1].set_ylabel('Time Step (Source)')
    axes[1, 1].set_xticks(range(10))
    axes[1, 1].set_yticks(range(10))
    add_values_to_heatmap(axes[1, 1], audio_sim_after)
    # 独立颜色条
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar4.set_label('Audio Attention (Large → Dark Red)', fontsize=9, color='darkred')

    # ========== 调整布局并保存 ==========
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  > 已保存模态独立配色的相似度矩阵: {save_path}")


# ===================== GNN 模块 (纯软监督版) =====================
class TemporalGNN_SoftSupervision(nn.Module):
    """
    纯软监督GNN：
    - 邻接矩阵完全由模型学习（保持原始方式）
    - 伪标签只用于计算软监督Loss，引导学习方向
    - 不会损失任何原始GNN的信息
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()

        # 原始GNN组件（完全保留）
        self.node_linear1 = nn.Linear(in_dim, hidden_dim)
        self.node_linear2 = nn.Linear(hidden_dim, out_dim)
        self.adj_linear = nn.Linear(in_dim * 2, 1)
        self.scale = hidden_dim ** -0.5

        # 伪标签温度参数
        self.pseudo_temperature = nn.Parameter(torch.tensor(0.1))

        # 软监督权重（可学习，控制伪标签的引导强度）
        self.supervision_weight = nn.Parameter(torch.tensor(-2.0))  # sigmoid后约0.12

        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=False)

        # 初始化
        xavier_uniform_(self.adj_linear.weight)
        xavier_uniform_(self.node_linear1.weight)
        xavier_uniform_(self.node_linear2.weight)

        # ★★★ 用于保存GNN输入/输出特征（可视化用）★★★
        self.feat_before = None
        self.feat_after = None

    def build_learned_adj(self, x):
        """原始GNN的可学习邻接矩阵（完全保留原始逻辑）"""
        B, T, C = x.shape
        x_i = x.unsqueeze(2).expand(B, T, T, C)
        x_j = x.unsqueeze(1).expand(B, T, T, C)
        x_pair = torch.cat([x_i, x_j], dim=-1)
        adj = self.adj_linear(x_pair).squeeze(-1) * self.scale
        adj = F.softmax(adj, dim=-1)
        return adj

    def build_pseudo_adj(self, pseudo_labels, target_dtype):
        """从伪标签生成目标邻接矩阵（仅用于计算Loss）"""
        pseudo_probs = F.normalize(pseudo_labels.to(target_dtype), dim=-1)
        adj = torch.bmm(pseudo_probs, pseudo_probs.transpose(1, 2))  # [B, T, T]
        temp = self.pseudo_temperature.abs().clamp(min=0.01)
        adj = F.softmax(adj / temp, dim=-1)
        return adj

    def compute_supervision_loss(self, learned_adj, pseudo_adj):
        """计算软监督损失"""
        pseudo_adj = pseudo_adj.to(learned_adj.dtype)
        mse_loss = F.mse_loss(learned_adj, pseudo_adj.detach())
        weight = torch.sigmoid(self.supervision_weight)
        supervision_loss = mse_loss * weight
        return supervision_loss

    def forward(self, x, pseudo_labels=None):
        """
        x: [Seq, Batch, Dim] 或 [Batch, Seq, Dim]
        pseudo_labels: [Batch, T, Classes] - CLIP/CLAP伪标签

        返回:
            h: 输出特征
            adj: 使用的邻接矩阵（learned_adj，未被修改）
            supervision_loss: 软监督损失
        """
        # ========== 维度处理 ==========
        is_seq_first = False
        if pseudo_labels is not None:
            batch_size = pseudo_labels.shape[0]
            seq_len = pseudo_labels.shape[1]
            if x.shape[0] == seq_len and x.shape[1] == batch_size:
                is_seq_first = True
                x = x.permute(1, 0, 2)
        else:
            if x.dim() == 3 and x.shape[0] < x.shape[1]:
                is_seq_first = True
                x = x.permute(1, 0, 2)

        x_original = x  # [B, T, C]

        # ★★★ 保存GNN输入特征（取第一个样本用于可视化）★★★
        self.feat_before = x_original[0].clone()  # [T, C]

        # ========== 生成邻接矩阵（原始方式，不修改） ==========
        learned_adj = self.build_learned_adj(x_original)  # [B, T, T]
        adj = learned_adj

        # ========== 计算软监督损失 ==========
        supervision_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if pseudo_labels is not None:
            pseudo_adj = self.build_pseudo_adj(pseudo_labels, target_dtype=x.dtype)
            supervision_loss = self.compute_supervision_loss(learned_adj, pseudo_adj)

        # ========== GNN消息传递（原始逻辑，不变） ==========
        h = self.node_linear1(x_original)
        h = self.relu(h)
        h = self.dropout(h)

        h = torch.bmm(adj, h)  # 使用learned_adj

        h = self.node_linear2(h)
        h = self.layer_norm(h + x_original)
        h = self.dropout(h)

        # ★★★ 保存GNN输出特征（取第一个样本用于可视化）★★★
        self.feat_after = h[0].clone()  # [T, C]

        # 恢复原始维度
        if is_seq_first:
            h = h.permute(1, 0, 2)

        return h, adj, supervision_loss


class StructureConsistencyLoss(nn.Module):
    """结构一致性损失"""

    def __init__(self, lambda_cross=1.0, lambda_smooth=0.1, lambda_recon=0.5):
        super(StructureConsistencyLoss, self).__init__()
        self.lambda_cross = lambda_cross
        self.lambda_smooth = lambda_smooth
        self.lambda_recon = lambda_recon

    def forward(self, vis_feat, vis_out, vis_adj, audio_feat, audio_out, audio_adj):
        # 维度对齐
        if vis_feat.shape[0] != vis_out.shape[0]:
            vis_out = vis_out.permute(1, 0, 2)
        if audio_feat.shape[0] != audio_out.shape[0]:
            audio_out = audio_out.permute(1, 0, 2)

        loss_recon_v = F.mse_loss(vis_out, vis_feat)
        loss_recon_a = F.mse_loss(audio_out, audio_feat)
        loss_smooth_v = torch.mean(vis_adj ** 2)
        loss_smooth_a = torch.mean(audio_adj ** 2)
        loss_cross_struct = F.mse_loss(vis_adj, audio_adj) * 100.0

        total_loss = (
                self.lambda_recon * (loss_recon_v + loss_recon_a) +
                self.lambda_smooth * (loss_smooth_v + loss_smooth_a) +
                self.lambda_cross * loss_cross_struct
        )
        return total_loss, loss_cross_struct


# ===================== 基础组件（保持不变） =====================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding="same")
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1d(x)
        out = self.bn(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu2(out)
        return out


class SlidingWindowModel(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, kernel_size=2, stride=1):
        super(SlidingWindowModel, self).__init__()
        layers = []
        for i in range(num_blocks):
            layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FuseModel(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, kernel_size=2, stride=1):
        super(FuseModel, self).__init__()
        layers = []
        for i in range(num_blocks):
            layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ExpendAs(nn.Module):
    def __init__(self, rep):
        super(ExpendAs, self).__init__()
        self.rep = rep

    def forward(self, tensor):
        return tensor.repeat(1, self.rep, 1)


class ChannelBlock(nn.Module):
    def __init__(self, in_channels, filter_size):
        super(ChannelBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_size, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm1d(filter_size)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(in_channels, filter_size, kernel_size=5, padding=2)
        self.batch2 = nn.BatchNorm1d(filter_size)
        self.relu2 = nn.ReLU(inplace=False)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(filter_size * 2, filter_size)
        self.batch3 = nn.BatchNorm1d(filter_size)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(filter_size, filter_size)
        self.conv3 = nn.Conv1d(filter_size * 2, filter_size, kernel_size=1, padding="same")
        self.batch4 = nn.BatchNorm1d(filter_size)
        self.relu4 = nn.ReLU(inplace=False)

    def forward(self, x_2s, x_5s):
        conv1 = self.conv1(x_2s)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)
        conv2 = self.conv2(x_5s)
        batch2 = self.batch2(conv2)
        relu2 = self.relu2(batch2)
        concat = torch.cat([relu1, relu2], dim=1)
        pooled = self.global_pool(concat)
        pooled = pooled.view(pooled.shape[0], -1)
        fc1 = self.fc1(pooled)
        relu3 = self.relu3(fc1)
        fc2 = self.fc2(relu3)
        sig = torch.sigmoid(fc2)
        a = sig.view(sig.size(0), sig.size(1), 1)
        a1 = 1 - sig
        a1 = 0.8 * a1
        a1 = a1.view(a1.size(0), a1.size(1), 1)
        y = relu1 * a
        y1 = relu2 * a1
        concat_y_y1 = torch.cat([y, y1], dim=1)
        conv3 = self.conv3(concat_y_y1)
        batch4 = self.batch4(conv3)
        relu4 = self.relu4(batch4)
        return relu4


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, filter_size, size):
        super(SpatialBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_size, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm1d(filter_size)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size=1, padding="same")
        self.batch2 = nn.BatchNorm1d(filter_size)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv1d(filter_size, 1, kernel_size=1, padding="same")
        self.expend_as = ExpendAs(filter_size)
        self.conv4 = nn.Conv1d(filter_size * 2, filter_size, kernel_size=size, padding="same")
        self.batch4 = nn.BatchNorm1d(filter_size)

    def forward(self, x, channel_data, filter_size, size):
        conv1 = self.conv1(x)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)
        conv2 = self.conv2(relu1)
        batch2 = self.batch2(conv2)
        relu2 = self.relu2(batch2)
        spatil_data = relu2
        data3 = F.relu(channel_data + spatil_data)
        data3 = self.conv3(data3)
        data3 = torch.sigmoid(data3)
        a = self.expend_as(data3)
        y = a * channel_data
        a1 = 1 - data3
        a1 = self.expend_as(a1)
        y1 = a1 * spatil_data
        data_a_a1 = torch.cat([y, y1], dim=1)
        conv3 = self.conv4(data_a_a1)
        batch3 = self.batch4(conv3)
        return batch3


class HAAM(nn.Module):
    def __init__(self, in_channels, filter_size, size):
        super(HAAM, self).__init__()
        self.channel_block = ChannelBlock(in_channels, filter_size)
        self.spatial_block = SpatialBlock(in_channels, filter_size, size)
        self.fc = nn.Conv1d(filter_size, 10, kernel_size=1, stride=1)

    def forward(self, x_1s, x_2s, x_5s, filte, size):
        channel_data = self.channel_block(x_2s, x_5s)
        haam_data = self.spatial_block(x_1s, channel_data, filte, size)
        haam_data = self.fc(haam_data)
        return haam_data


# ===================== 关键修改1：分类头改为67类 =====================
class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Linear(d_model, 1)
        self.event_classifier = nn.Linear(d_model, 67)  # 28 → 67

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.max(1)
        logits = self.classifier(fused_content)
        class_logits = self.event_classifier(max_fused_content)
        class_scores = class_logits
        return logits, class_scores


class Encoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        output = src
        for i in range(self.num_layers):
            output = self.layers[i](output)
        if self.norm:
            output = self.norm(output)
        return output


class Decoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        output = tgt
        for i in range(self.num_layers):
            output = self.layers[i](output, memory)
        if self.norm:
            output = self.norm(output)
        return output


class EncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory):
        memory = torch.cat([memory, tgt], dim=0)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)
        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, feature):
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)
        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()
        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)
        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)
        return output


# ===================== 主模型 (纯软监督版) =====================
class Temp_Model(nn.Module):
    # 可选优化：增加可视化路径参数，避免硬编码
    def __init__(self, in_channels, feature_dim, vis_base_path=r"E:\map\qianhou"):
        super(Temp_Model, self).__init__()
        # 1. 卷积层修改（适配20通道输入）
        self.video_txt_fc = nn.Conv1d(20, 10, kernel_size=1)
        self.audio_txt_fc = nn.Conv1d(20, 64, kernel_size=1)

        # 2. 滑动窗口模型（保持不变）
        self.SlidingWindowModel1s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=1)
        self.SlidingWindowModel2s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=2)
        self.SlidingWindowModel5s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=5)
        self.audio_SlidingWindowModel1s = SlidingWindowModel(5, 64, 64, kernel_size=1)
        self.audio_SlidingWindowModel2s = SlidingWindowModel(5, 64, 64, kernel_size=2)
        self.audio_SlidingWindowModel5s = SlidingWindowModel(5, 64, 64, kernel_size=5)

        # 3. HAAM模块（保持不变）
        self.haam = HAAM(in_channels, filter_size=64, size=3)
        self.audio_haam = HAAM(64, filter_size=64, size=3)

        # 4. 编码器/解码器修改（适配实际输入维度）
        self.video_encoder = InternalTemporalRelationModule(input_dim=512, d_model=256, feedforward_dim=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=512, d_model=256, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=128, d_model=256, feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=128, d_model=256, feedforward_dim=1024)

        # ★★★ 使用纯软监督GNN ★★★
        self.visual_gnn = TemporalGNN_SoftSupervision(
            in_dim=256, hidden_dim=128, out_dim=256, dropout=0.1
        )
        self.audio_gnn = TemporalGNN_SoftSupervision(
            in_dim=256, hidden_dim=128, out_dim=256, dropout=0.1
        )

        # 结构一致性损失
        self.struct_loss_fn = StructureConsistencyLoss(
            lambda_cross=0.5,
            lambda_smooth=0.1,
            lambda_recon=0.5
        )

        # ===================== 关键修改2：分类头改为67类 =====================
        self.video_fc = nn.Linear(256, out_features=67)  # 28 → 67
        self.audio_fc = nn.Linear(256, out_features=67)  # 28 → 67
        self.localize_module = SupvLocalizeModule(256)
        self.vis_localize_module = SupvLocalizeModule(256)
        self.audio_localize_module = SupvLocalizeModule(256)
        self.fuse = FuseModel(5, 512, 256, kernel_size=3)

        # ★★★ 可视化相关配置 ★★★
        self.vis_base_path = vis_base_path  # 从参数传入，避免硬编码
        os.makedirs(self.vis_base_path, exist_ok=True)

    def save_gnn_visualization(self, epoch, sample_idx):
        """保存GNN相似度矩阵可视化图"""
        # 获取GNN前后的特征
        vis_before = self.visual_gnn.feat_before
        vis_after = self.visual_gnn.feat_after
        audio_before = self.audio_gnn.feat_before
        audio_after = self.audio_gnn.feat_after

        if vis_before is None or audio_before is None:
            print(f"Warning: GNN特征未初始化，跳过可视化 epoch {epoch} sample {sample_idx}")
            return

        # 创建轮次文件夹
        epoch_dir = create_epoch_dir(self.vis_base_path, epoch)

        # 图片保存路径（每个样本一张图）
        save_path_v1 = os.path.join(epoch_dir, f"sim_matrix_sample{sample_idx + 1}.png")
        visualize_similarity_matrix(
            epoch=epoch,
            sample_idx=sample_idx,
            vis_feat_before=vis_before,
            vis_feat_after=vis_after,
            audio_feat_before=audio_before,
            audio_feat_after=audio_after,
            save_path=save_path_v1
        )

    def forward(self, feat, text_feat, audio_feat, audio_text_feat,
                clip_pseudo_labels=None, clap_pseudo_labels=None,
                epoch=None, save_vis=False, sample_idx=0):
        """
        新增参数：
        - epoch: 当前轮次（用于保存文件夹命名）
        - save_vis: 是否保存可视化图（测试集时设为True）
        - sample_idx: 当前样本索引（用于命名）
        """
        feat = torch.cat([feat, text_feat], dim=1)
        feat = self.video_txt_fc(feat)
        audio_feat = torch.cat([audio_feat, audio_text_feat], dim=1)
        audio_feat = self.audio_txt_fc(audio_feat)

        feat1s = self.SlidingWindowModel1s(feat)
        feat2s = self.SlidingWindowModel2s(feat)
        feat5s = self.SlidingWindowModel5s(feat)
        feat = self.haam(feat1s, feat2s, feat5s, filte=64, size=3)

        audio_feat1s = self.audio_SlidingWindowModel1s(audio_feat)
        audio_feat2s = self.audio_SlidingWindowModel2s(audio_feat)
        audio_feat5s = self.audio_SlidingWindowModel5s(audio_feat)
        audio_feat = self.audio_haam(audio_feat1s, audio_feat2s, audio_feat5s, filte=64, size=3)

        vis_feat_encode = self.video_encoder(feat)
        audio_feat_encode = self.audio_encoder(audio_feat)

        # ★★★ GNN with Soft Supervision ★★★
        vis_gnn_out, vis_adj, vis_sup_loss = self.visual_gnn(vis_feat_encode, clip_pseudo_labels)
        audio_gnn_out, audio_adj, audio_sup_loss = self.audio_gnn(audio_feat_encode, clap_pseudo_labels)

        # 结构一致性损失
        struct_loss, cross_struct_loss = self.struct_loss_fn(
            vis_feat_encode, vis_gnn_out, vis_adj,
            audio_feat_encode, audio_gnn_out, audio_adj
        )

        # ★★★ 总GNN损失 = 结构损失 + 软监督损失 ★★★
        gnn_total_loss = struct_loss + vis_sup_loss + audio_sup_loss

        # 调试打印
        if self.training and torch.rand(1).item() < 0.02:
            sup_weight_v = torch.sigmoid(self.visual_gnn.supervision_weight).item()
            print(f" > [GNN] Soft Supervision | "
                  f"Sup Weight: {sup_weight_v:.4f} | "
                  f"Vis Sup Loss: {vis_sup_loss.item():.6f} | "
                  f"Audio Sup Loss: {audio_sup_loss.item():.6f} | "
                  f"Struct Loss: {struct_loss.item():.6f}")

        # 残差更新
        if vis_feat_encode.shape[0] != vis_gnn_out.shape[0]:
            vis_gnn_out_aligned = vis_gnn_out.permute(1, 0, 2)
            audio_gnn_out_aligned = audio_gnn_out.permute(1, 0, 2)
        else:
            vis_gnn_out_aligned = vis_gnn_out
            audio_gnn_out_aligned = audio_gnn_out

        vis_feat_encode = vis_feat_encode + 0.1 * vis_gnn_out_aligned
        audio_feat_encode = audio_feat_encode + 0.1 * audio_gnn_out_aligned

        # ★★★ 保存可视化图（测试集时调用）★★★
        if save_vis and epoch is not None:
            self.save_gnn_visualization(epoch, sample_idx)

        # 后续处理
        if vis_feat_encode.shape[0] != feat.shape[0]:
            vis_feat_encode_perm = vis_feat_encode.permute(1, 0, 2)
            audio_feat_encode_perm = audio_feat_encode.permute(1, 0, 2)
        else:
            vis_feat_encode_perm = vis_feat_encode
            audio_feat_encode_perm = audio_feat_encode

        B, f, C = audio_feat_encode_perm.shape
        Fv = vis_feat_encode_perm.reshape(B * f, C)
        Fa = audio_feat_encode_perm.reshape(B * f, C)

        attn_scores = torch.matmul(Fv, Fa.T)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        audio_feat_refined = ((attn_scores @ Fa)).reshape(B, f, C) + audio_feat_encode_perm

        feat_final = torch.cat([vis_feat_encode_perm, audio_feat_refined], dim=-1)
        feat_final = self.fuse(feat_final.permute(0, 2, 1)).permute(0, 2, 1)

        # KL Loss
        video_fc = self.video_fc(vis_feat_encode_perm)
        audio_fc = self.audio_fc(audio_feat_encode_perm)

        video_fcc = nn.ReLU()(video_fc)
        audio_fcc = nn.ReLU()(audio_fc)
        video_sim = nn.Softmax(dim=-1)(video_fcc)
        audio_sim = nn.Softmax(dim=-1)(audio_fcc)

        kl_loss = F.kl_div((audio_sim + 1e-8).log(), video_sim, reduction='sum')

        is_event_scores, event_scores = self.localize_module(feat_final)
        vis_is_event_scores, vis_event_scores = self.vis_localize_module(vis_feat_encode_perm)
        audio_is_event_scores, audio_event_scores = self.audio_localize_module(audio_feat_encode_perm)

        return (is_event_scores, event_scores, kl_loss,
                vis_is_event_scores, vis_event_scores,
                audio_is_event_scores, audio_event_scores, gnn_total_loss)