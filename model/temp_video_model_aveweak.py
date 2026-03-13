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

plt.switch_backend('Agg')


# ===================== 可视化工具函数 =====================
def create_epoch_dir(base_path, epoch):
    epoch_dir = os.path.join(base_path, f"epoch{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    return epoch_dir


import matplotlib.colors as mcolors


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
        temperature = 0.1
        sim_scaled = sim_matrix / temperature
        sim_exp = np.exp(sim_scaled - sim_scaled.max(axis=1, keepdims=True))
        sim_normalized = sim_exp / (sim_exp.sum(axis=1, keepdims=True) + 1e-8)
        return sim_normalized

    def add_values_to_heatmap(ax, matrix, fontsize=7):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text_color = 'white' if value > 0.3 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                        fontsize=fontsize, color=text_color, fontweight='bold')

    vis_sim_before = compute_row_normalized_similarity(vis_feat_before)
    vis_sim_after = compute_row_normalized_similarity(vis_feat_after)
    audio_sim_before = compute_row_normalized_similarity(audio_feat_before)
    audio_sim_after = compute_row_normalized_similarity(audio_feat_after)

    cmap_vis = plt.cm.Blues
    norm_vis = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.3, vmax=1)
    cmap_audio = plt.cm.Reds
    norm_audio = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.3, vmax=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Epoch {epoch} - Sample {sample_idx + 1}', fontsize=14, fontweight='bold')

    im1 = axes[0, 0].imshow(vis_sim_before, cmap=cmap_vis, norm=norm_vis)
    axes[0, 0].set_title('Visual (Before GNN)', fontsize=11)
    add_values_to_heatmap(axes[0, 0], vis_sim_before)
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(vis_sim_after, cmap=cmap_vis, norm=norm_vis)
    axes[0, 1].set_title('Visual (After GNN)', fontsize=11)
    add_values_to_heatmap(axes[0, 1], vis_sim_after)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 0].imshow(audio_sim_before, cmap=cmap_audio, norm=norm_audio)
    axes[1, 0].set_title('Audio (Before GNN)', fontsize=11)
    add_values_to_heatmap(axes[1, 0], audio_sim_before)
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(audio_sim_after, cmap=cmap_audio, norm=norm_audio)
    axes[1, 1].set_title('Audio (After GNN)', fontsize=11)
    add_values_to_heatmap(axes[1, 1], audio_sim_after)
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ===================== GNN 模块 =====================
class TemporalGNN_SoftSupervision(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.node_linear1 = nn.Linear(in_dim, hidden_dim)
        self.node_linear2 = nn.Linear(hidden_dim, out_dim)
        self.adj_linear = nn.Linear(in_dim * 2, 1)
        self.scale = hidden_dim ** -0.5
        self.pseudo_temperature = nn.Parameter(torch.tensor(0.1))
        self.supervision_weight = nn.Parameter(torch.tensor(-2.0))
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=False)
        xavier_uniform_(self.adj_linear.weight)
        xavier_uniform_(self.node_linear1.weight)
        xavier_uniform_(self.node_linear2.weight)
        self.feat_before = None
        self.feat_after = None

    def build_learned_adj(self, x):
        B, T, C = x.shape
        x_i = x.unsqueeze(2).expand(B, T, T, C)
        x_j = x.unsqueeze(1).expand(B, T, T, C)
        x_pair = torch.cat([x_i, x_j], dim=-1)
        adj = self.adj_linear(x_pair).squeeze(-1) * self.scale
        adj = F.softmax(adj, dim=-1)
        return adj

    def build_pseudo_adj(self, pseudo_labels, target_dtype):
        pseudo_probs = F.normalize(pseudo_labels.to(target_dtype), dim=-1)
        adj = torch.bmm(pseudo_probs, pseudo_probs.transpose(1, 2))
        temp = self.pseudo_temperature.abs().clamp(min=0.01)
        adj = F.softmax(adj / temp, dim=-1)
        return adj

    def compute_supervision_loss(self, learned_adj, pseudo_adj):
        pseudo_adj = pseudo_adj.to(learned_adj.dtype)
        mse_loss = F.mse_loss(learned_adj, pseudo_adj.detach())
        weight = torch.sigmoid(self.supervision_weight)
        supervision_loss = mse_loss * weight
        return supervision_loss

    def forward(self, x, pseudo_labels=None):
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

        x_original = x
        self.feat_before = x_original[0].clone()

        learned_adj = self.build_learned_adj(x_original)
        adj = learned_adj

        supervision_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if pseudo_labels is not None:
            pseudo_adj = self.build_pseudo_adj(pseudo_labels, target_dtype=x.dtype)
            supervision_loss = self.compute_supervision_loss(learned_adj, pseudo_adj)

        h = self.node_linear1(x_original)
        h = self.relu(h)
        h = self.dropout(h)
        h = torch.bmm(adj, h)
        h = self.node_linear2(h)
        h = self.layer_norm(h + x_original)
        h = self.dropout(h)

        self.feat_after = h[0].clone()

        if is_seq_first:
            h = h.permute(1, 0, 2)

        return h, adj, supervision_loss


class StructureConsistencyLoss(nn.Module):
    def __init__(self, lambda_cross=1.0, lambda_smooth=0.1, lambda_recon=0.5):
        super(StructureConsistencyLoss, self).__init__()
        self.lambda_cross = lambda_cross
        self.lambda_smooth = lambda_smooth
        self.lambda_recon = lambda_recon

    def forward(self, vis_feat, vis_out, vis_adj, audio_feat, audio_out, audio_adj):
        if vis_feat.shape[0] != vis_out.shape[0]:
            vis_out = vis_out.permute(1, 0, 2)
        if audio_feat.shape[0] != audio_out.shape[0]:
            audio_out = audio_out.permute(1, 0, 2)

        loss_recon_v = F.mse_loss(vis_out, vis_feat)
        loss_recon_a = F.mse_loss(audio_out, audio_feat)
        loss_smooth_v = torch.mean(vis_adj ** 2)
        loss_smooth_a = torch.mean(audio_adj ** 2)

        # 处理邻接矩阵维度不匹配的情况
        if vis_adj.shape != audio_adj.shape:
            # 使用平均池化对齐时间维度
            min_size = min(vis_adj.shape[1], audio_adj.shape[1])
            vis_adj_aligned = F.adaptive_avg_pool2d(vis_adj.unsqueeze(1), (min_size, min_size)).squeeze(1)
            audio_adj_aligned = F.adaptive_avg_pool2d(audio_adj.unsqueeze(1), (min_size, min_size)).squeeze(1)
            loss_cross_struct = F.mse_loss(vis_adj_aligned, audio_adj_aligned) * 100.0
        else:
            loss_cross_struct = F.mse_loss(vis_adj, audio_adj) * 100.0

        total_loss = (
                self.lambda_recon * (loss_recon_v + loss_recon_a) +
                self.lambda_smooth * (loss_smooth_v + loss_smooth_a) +
                self.lambda_cross * loss_cross_struct
        )
        return total_loss, loss_cross_struct


# ===================== 基础组件 =====================
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
    def __init__(self, in_channels, filter_size, size, out_channels=10):
        super(HAAM, self).__init__()
        self.channel_block = ChannelBlock(in_channels, filter_size)
        self.spatial_block = SpatialBlock(in_channels, filter_size, size)
        self.fc = nn.Conv1d(filter_size, out_channels, kernel_size=1, stride=1)

    def forward(self, x_1s, x_2s, x_5s, filte, size):
        channel_data = self.channel_block(x_2s, x_5s)
        haam_data = self.spatial_block(x_1s, channel_data, filte, size)
        haam_data = self.fc(haam_data)
        return haam_data


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Linear(d_model, 1)
        self.event_classifier = nn.Linear(d_model, 29)

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
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
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
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
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


# ===================== 主模型 =====================
class Temp_Model(nn.Module):
    def __init__(self, in_channels, feature_dim, vis_base_path=r"E:\map\qianhou"):
        super(Temp_Model, self).__init__()

        # 视觉: [B, 10, 768] + [B, 1, 768] → [B, 11, 768] → [B, 10, 768]
        self.video_txt_fc = nn.Conv1d(11, 10, kernel_size=1)

        # 音频: 先自适应池化到10个时间步，再处理
        # 不再拼接 audio_text_feat 到时间步维度
        # 改为：在特征维度融合文本信息
        self.audio_pool = nn.AdaptiveAvgPool1d(10)  # 将任意时间步压缩到10
        self.audio_text_proj = nn.Linear(feature_dim, feature_dim)  # 文本特征投影
        self.audio_feat_proj = nn.Linear(feature_dim * 2, feature_dim)  # 拼接后降维

        # 滑动窗口模型
        self.SlidingWindowModel1s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=1)
        self.SlidingWindowModel2s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=2)
        self.SlidingWindowModel5s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=5)

        self.audio_SlidingWindowModel1s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=1)
        self.audio_SlidingWindowModel2s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=2)
        self.audio_SlidingWindowModel5s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=5)

        # HAAM模块
        self.haam = HAAM(in_channels, filter_size=64, size=3, out_channels=10)
        self.audio_haam = HAAM(in_channels, filter_size=64, size=3, out_channels=10)

        # 编码器
        self.video_encoder = InternalTemporalRelationModule(input_dim=feature_dim, d_model=256, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=feature_dim, d_model=256, feedforward_dim=1024)

        # GNN模块
        self.visual_gnn = TemporalGNN_SoftSupervision(in_dim=256, hidden_dim=128, out_dim=256, dropout=0.1)
        self.audio_gnn = TemporalGNN_SoftSupervision(in_dim=256, hidden_dim=128, out_dim=256, dropout=0.1)

        # 结构一致性损失
        self.struct_loss_fn = StructureConsistencyLoss(lambda_cross=0.5, lambda_smooth=0.1, lambda_recon=0.5)

        # 分类层
        self.video_fc = nn.Linear(256, out_features=29)
        self.audio_fc = nn.Linear(256, out_features=29)
        self.localize_module = SupvLocalizeModule(256)
        self.vis_localize_module = SupvLocalizeModule(256)
        self.audio_localize_module = SupvLocalizeModule(256)
        self.fuse = FuseModel(5, 512, 256, kernel_size=3)

        # 可视化配置
        self.vis_base_path = vis_base_path
        os.makedirs(self.vis_base_path, exist_ok=True)

    def save_gnn_visualization(self, epoch, sample_idx):
        vis_before = self.visual_gnn.feat_before
        vis_after = self.visual_gnn.feat_after
        audio_before = self.audio_gnn.feat_before
        audio_after = self.audio_gnn.feat_after

        if vis_before is None or audio_before is None:
            return

        epoch_dir = create_epoch_dir(self.vis_base_path, epoch)
        save_path = os.path.join(epoch_dir, f"sim_matrix_sample{sample_idx + 1}.png")
        visualize_similarity_matrix(
            epoch=epoch,
            sample_idx=sample_idx,
            vis_feat_before=vis_before,
            vis_feat_after=vis_after,
            audio_feat_before=audio_before,
            audio_feat_after=audio_after,
            save_path=save_path
        )

    def forward(self, feat, text_feat, audio_feat, audio_text_feat,
                clip_pseudo_labels=None, clap_pseudo_labels=None,
                epoch=None, save_vis=False, sample_idx=0):
        """
        实际输入维度（根据错误信息确认）：
        - feat:            [B, 10, 768]  视觉特征
        - text_feat:       [B, 1, 768]   CLIP文本特征
        - audio_feat:      [B, 128, 768] 音频特征
        - audio_text_feat: [B, 128, 768] CLAP文本特征（与音频时间步相同）
        """
        # ===================== 类型转换（修复 double/float 不匹配）=====================
        # 确保所有输入都是 float 类型（与模型权重一致）
        if feat.dtype == torch.float64:
            feat = feat.float()
        if text_feat.dtype == torch.float64:
            text_feat = text_feat.float()
        if audio_feat.dtype == torch.float64:
            audio_feat = audio_feat.float()
        if audio_text_feat.dtype == torch.float64:
            audio_text_feat = audio_text_feat.float()
        # ===================== 1. 维度预处理 =====================
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if audio_text_feat.dim() == 2:
            audio_text_feat = audio_text_feat.unsqueeze(1)

        # ===================== 2. 视觉特征处理 =====================
        # [B, 10, 768] + [B, 1, 768] → [B, 11, 768]
        feat = torch.cat([feat, text_feat], dim=1)
        feat = self.video_txt_fc(feat)  # [B, 11, 768] → [B, 10, 768]

        # ===================== 3. 音频特征处理 =====================
        # 方案：先在特征维度融合文本信息，再池化压缩时间步

        # 3a. 文本特征投影
        audio_text_proj = self.audio_text_proj(audio_text_feat)  # [B, 128, 768]

        # 3b. 在特征维度拼接音频和文本
        audio_combined = torch.cat([audio_feat, audio_text_proj], dim=2)  # [B, 128, 1536]

        # 3c. 降维回原始特征维度
        audio_combined = self.audio_feat_proj(audio_combined)  # [B, 128, 768]

        # 3d. 自适应池化：将128个时间步压缩到10个
        # AdaptiveAvgPool1d 需要 [B, C, T] 格式
        audio_feat = audio_combined.permute(0, 2, 1)  # [B, 768, 128]
        audio_feat = self.audio_pool(audio_feat)  # [B, 768, 10]
        audio_feat = audio_feat.permute(0, 2, 1)  # [B, 10, 768]

        # ===================== 4. 滑动窗口特征提取 =====================
        feat1s = self.SlidingWindowModel1s(feat)
        feat2s = self.SlidingWindowModel2s(feat)
        feat5s = self.SlidingWindowModel5s(feat)
        feat = self.haam(feat1s, feat2s, feat5s, filte=64, size=3)

        audio_feat1s = self.audio_SlidingWindowModel1s(audio_feat)
        audio_feat2s = self.audio_SlidingWindowModel2s(audio_feat)
        audio_feat5s = self.audio_SlidingWindowModel5s(audio_feat)
        audio_feat = self.audio_haam(audio_feat1s, audio_feat2s, audio_feat5s, filte=64, size=3)

        # ===================== 5. 编码器处理 =====================
        vis_feat_encode = self.video_encoder(feat)
        audio_feat_encode = self.audio_encoder(audio_feat)

        # ===================== 6. GNN处理 =====================
        vis_gnn_out, vis_adj, vis_sup_loss = self.visual_gnn(vis_feat_encode, clip_pseudo_labels)
        audio_gnn_out, audio_adj, audio_sup_loss = self.audio_gnn(audio_feat_encode, clap_pseudo_labels)

        struct_loss, cross_struct_loss = self.struct_loss_fn(
            vis_feat_encode, vis_gnn_out, vis_adj,
            audio_feat_encode, audio_gnn_out, audio_adj
        )

        gnn_total_loss = struct_loss + vis_sup_loss + audio_sup_loss

        if self.training and torch.rand(1).item() < 0.02:
            sup_weight_v = torch.sigmoid(self.visual_gnn.supervision_weight).item()
            print(f" > [GNN] Sup Weight: {sup_weight_v:.4f} | "
                  f"Vis Sup Loss: {vis_sup_loss.item():.6f} | "
                  f"Audio Sup Loss: {audio_sup_loss.item():.6f} | "
                  f"Struct Loss: {struct_loss.item():.6f}")

        # ===================== 7. 残差更新 =====================
        if vis_feat_encode.shape[0] != vis_gnn_out.shape[0]:
            vis_gnn_out_aligned = vis_gnn_out.permute(1, 0, 2)
            audio_gnn_out_aligned = audio_gnn_out.permute(1, 0, 2)
        else:
            vis_gnn_out_aligned = vis_gnn_out
            audio_gnn_out_aligned = audio_gnn_out

        vis_feat_encode = vis_feat_encode + 0.1 * vis_gnn_out_aligned
        audio_feat_encode = audio_feat_encode + 0.1 * audio_gnn_out_aligned

        if save_vis and epoch is not None:
            self.save_gnn_visualization(epoch, sample_idx)

        # ===================== 8. 特征融合 =====================
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

        # ===================== 9. KL Loss =====================
        video_fc = self.video_fc(vis_feat_encode_perm)
        audio_fc = self.audio_fc(audio_feat_encode_perm)

        video_fcc = nn.ReLU()(video_fc)
        audio_fcc = nn.ReLU()(audio_fc)
        video_sim = nn.Softmax(dim=-1)(video_fcc)
        audio_sim = nn.Softmax(dim=-1)(audio_fcc)

        kl_loss = F.kl_div((audio_sim + 1e-8).log(), video_sim, reduction='sum')

        # ===================== 10. 分类输出 =====================
        is_event_scores, event_scores = self.localize_module(feat_final)
        vis_is_event_scores, vis_event_scores = self.vis_localize_module(vis_feat_encode_perm)
        audio_is_event_scores, audio_event_scores = self.audio_localize_module(audio_feat_encode_perm)

        return (is_event_scores, event_scores, kl_loss + gnn_total_loss,
                vis_is_event_scores, vis_event_scores,
                audio_is_event_scores, audio_event_scores)