# -*- coding: utf-8 -*-
import os
import time
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

# ================== 绘图相关配置 (最顶部，避免导入顺序问题) ==================
plt.switch_backend('Agg')  # 非交互式后端，适合服务器/无桌面环境
ROOT_SAVE_DIR = r"E:\map\modeltap\ovavesupqvxian"
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)

# 解决中文乱码核心配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 兼容不同系统
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 100  # 默认分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图片分辨率

# ================== 导入项目模块 ==================
from configs.opts import parser
from model.modelovavesup import Temp_Model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from dataset.ovavesup import AVEDatasetV2

# ================== 常量定义 ==================
SEED = 43
OVAVE_NUM_CLASSES = 67  # OVAVE 数据集类别数

# ================== 种子固定 ==================
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================== 环境变量配置 ==================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"


# ================== 绘图核心函数（最终汇总版：仅训练结束生成1张图） ==================
def normalize_data(data):
    """Min-Max归一化到0-1区间（避免除以0）"""
    if len(data) < 2:
        return np.array(data) if data else np.array([])
    min_val = np.min(data)
    max_val = np.max(data)
    return (np.array(data) - min_val) / (max_val - min_val + 1e-8)


def plot_final_loss_acc_curve(train_loss_list, val_acc_list, eval_freq, save_dir=ROOT_SAVE_DIR):
    """
    训练结束后，仅生成1张包含所有epoch的Loss和准确率汇总图
    :param train_loss_list: 所有epoch的训练损失列表
    :param val_acc_list: 所有验证轮次的准确率列表
    :param eval_freq: 验证频率（每N轮验证一次）
    :param save_dir: 保存目录
    """
    # 生成最终汇总图的保存路径
    save_path = os.path.join(save_dir, "loss_acc_curve_final.png")

    # 数据预处理（使用所有训练轮次的数据）
    loss_norm = normalize_data(train_loss_list)  # 所有epoch的Loss归一化
    acc_norm = np.array(val_acc_list) / 100.0 if val_acc_list else np.array([])  # 所有验证轮次的Acc转0-1

    # 生成横坐标
    epochs_loss = range(1, len(loss_norm) + 1)  # Loss横坐标：1~总epoch数
    epochs_acc = []
    if len(val_acc_list) > 0:
        # 验证准确率横坐标：按eval_freq生成（如每1轮验证则为1,2,3...）
        epochs_acc = list(range(eval_freq, eval_freq * len(val_acc_list) + 1, eval_freq))
        # 确保最后一个点不超过总epoch数
        epochs_acc = [e for e in epochs_acc if e <= len(loss_norm)]

    # 创建画布（全新画布，避免残留）
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制训练Loss曲线（红色，包含所有epoch）
    if len(epochs_loss) > 0:
        ax.plot(epochs_loss, loss_norm,
                label='训练Loss（归一化）',
                color='#e74c3c',  # 红色系
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.8)

    # 绘制验证准确率曲线（蓝色，包含所有验证轮次）
    if len(epochs_acc) > 0 and len(acc_norm) > 0:
        ax.plot(epochs_acc, acc_norm,
                label='验证准确率',
                color='#3498db',  # 蓝色系
                linewidth=2.5,
                marker='s',
                markersize=4,
                alpha=0.8)

    # 图表样式配置（核心：显示所有epoch的完整趋势）
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=12, fontweight='bold')
    ax.set_ylabel('数值（0-1）', fontsize=12, fontweight='bold')
    ax.set_title('训练损失 vs 验证准确率（所有Epoch汇总）', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(1, len(loss_norm))  # X轴覆盖所有epoch
    ax.set_ylim(0, 1.05)  # Y轴固定0~1.05（留少量余量）

    # X轴刻度：按总epoch数均分，最多显示15个刻度，避免拥挤
    tick_step = max(1, len(loss_norm) // 15)
    ax.set_xticks(np.arange(1, len(loss_norm) + 1, step=tick_step))
    ax.grid(True, alpha=0.3, linestyle='--')  # 网格线
    ax.legend(fontsize=11, loc='best', framealpha=0.9)  # 图例

    # 保存并关闭画布
    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # 强制关闭，释放内存

    print(f"✅ 最终汇总曲线已保存：{save_path}")


# ================== DataLoader 自定义整理函数 ==================
def custom_collate_fn(batch):
    """修复DataLoader数据类型混合导致的崩溃问题"""
    transposed = list(zip(*batch))

    # 强制转换为float32，统一数据类型
    visual_feature = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[0]])
    text_feat = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[1]])
    pseudo_label = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[2]])
    audio_feat = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[3]])
    audio_text_feat = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[4]])
    audio_pseudo_label = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[5]])
    labels = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[6]])

    return visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels


# ================== 训练单轮函数 ==================
def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch, args, logger):
    batch_time = AverageMeter()
    total_losses_with_gnn = AverageMeter()
    total_losses_without_gnn = AverageMeter()
    train_acc = AverageMeter()

    model.train()
    model.double()  # 适配模型double类型
    optimizer.zero_grad()
    end_time = time.time()

    for n_iter, batch_data in enumerate(train_dataloader):
        # 解包数据
        visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels = batch_data
        bs = visual_feature.shape[0]

        # 数据移至GPU
        labels = labels.double().cuda(non_blocking=True)
        visual_feature = visual_feature.double().cuda(non_blocking=True)
        audio_feat = audio_feat.double().cuda(non_blocking=True)
        pseudo_label = pseudo_label.double().cuda(non_blocking=True)
        audio_pseudo_label = audio_pseudo_label.double().cuda(non_blocking=True)

        # 调试：第一轮打印数据信息
        if n_iter == 0 and epoch == 0:
            logger.info(f"\n[数据调试] 标签形状: {labels.shape} | 视觉特征形状: {visual_feature.shape}")
            logger.info(f"[数据调试] 标签总和: {labels.sum().item():.2f} | 最大值: {labels.max().item():.2f}")

        # 模型前向传播
        (is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores,
         audio_is_event_scores, audio_event_scores, gnn_total_loss) = model(
            visual_feature, text_feat, audio_feat, audio_text_feat,
            clip_pseudo_labels=pseudo_label,
            clap_pseudo_labels=audio_pseudo_label,
            epoch=epoch
        )

        # 损失值维度处理（避免多维度导致的错误）
        if gnn_total_loss.dim() > 0:
            gnn_total_loss = gnn_total_loss.mean()
        if kl_loss.dim() > 0:
            kl_loss = kl_loss.mean()

        # 展平存在性得分
        is_event_scores_flat = is_event_scores.reshape(bs, -1)
        vis_is_event_scores_flat = vis_is_event_scores.reshape(bs, -1)
        audio_is_event_scores_flat = audio_is_event_scores.reshape(bs, -1)

        # 标签处理：截取前67类（前景类）
        labels_foreground = labels[:, :, :OVAVE_NUM_CLASSES]
        # 存在性标签 (0/1)
        labels_BCE, _ = labels_foreground.max(-1)
        labels_BCE = labels_BCE.reshape(bs, -1).double()
        # 类别标签
        video_level_vec, _ = labels_foreground.max(1)
        gt_val, gt_idx = video_level_vec.max(-1)
        labels_event = gt_idx.long()

        # 计算存在性损失
        loss_is_event = criterion(is_event_scores_flat, labels_BCE)
        vis_loss_is_event = criterion(vis_is_event_scores_flat, labels_BCE)
        audio_loss_is_event = criterion(audio_is_event_scores_flat, labels_BCE)

        # 计算分类损失（仅对前景样本）
        valid_event_mask = gt_val > 0.5
        if valid_event_mask.sum() > 0:
            loss_event_class = criterion_event(event_scores[valid_event_mask], labels_event[valid_event_mask])
            vis_loss_event_class = criterion_event(vis_event_scores[valid_event_mask], labels_event[valid_event_mask])
            audio_loss_event_class = criterion_event(audio_event_scores[valid_event_mask],
                                                     labels_event[valid_event_mask])
        else:
            loss_event_class = torch.tensor(0.0).cuda()
            vis_loss_event_class = torch.tensor(0.0).cuda()
            audio_loss_event_class = torch.tensor(0.0).cuda()

        # 总损失计算
        loss_without_gnn = (loss_is_event + loss_event_class + kl_loss * 0.1 +
                            vis_loss_is_event + vis_loss_event_class +
                            audio_loss_is_event + audio_loss_event_class)
        loss_with_gnn = loss_without_gnn + gnn_total_loss * 1.0

        # 反向传播
        loss_with_gnn.backward()

        # 梯度裁剪（可选）
        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        # 优化器更新
        optimizer.step()
        optimizer.zero_grad()

        # 计算准确率
        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels_foreground)
        train_acc.update(acc.item(), bs)

        # 更新损失和时间统计
        total_losses_with_gnn.update(loss_with_gnn.item(), bs)
        total_losses_without_gnn.update(loss_without_gnn.item(), bs)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # 打印训练日志
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {total_losses_without_gnn.val:.4f} ({total_losses_without_gnn.avg:.4f})\t'
                f'Acc {train_acc.val:.2f}% ({train_acc.avg:.2f}%)'
            )

    return total_losses_with_gnn.avg


# ================== 验证单轮函数 ==================
@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, args, logger, eval_only=False):
    accuracy = AverageMeter()
    model.eval()
    model.double()

    for n_iter, batch_data in enumerate(test_dataloader):
        # 解包数据
        visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels = batch_data
        bs = visual_feature.shape[0]

        # 数据移至GPU
        labels = labels.double().cuda(non_blocking=True)
        visual_feature = visual_feature.double().cuda(non_blocking=True)
        audio_feat = audio_feat.double().cuda(non_blocking=True)
        pseudo_label = pseudo_label.double().cuda(non_blocking=True)
        audio_pseudo_label = audio_pseudo_label.double().cuda(non_blocking=True)

        # 模型前向传播
        (is_event_scores, event_scores, kl_loss, _, _, _, _, gnn_total_loss) = model(
            visual_feature, text_feat, audio_feat, audio_text_feat,
            clip_pseudo_labels=pseudo_label,
            clap_pseudo_labels=audio_pseudo_label,
            epoch=epoch
        )

        # 标签处理
        labels_foreground = labels[:, :, :OVAVE_NUM_CLASSES]
        # 计算准确率
        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels_foreground)
        accuracy.update(acc.item(), bs)

        # 打印验证日志
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\tAcc {accuracy.val:.2f}% ({accuracy.avg:.2f}%)')

    logger.info(f'===== Test Epoch {epoch} Final Accuracy: {accuracy.avg:.4f}% =====')
    return accuracy.avg


# ================== 准确率计算函数 ==================
def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    """计算监督式准确率（仅对前景样本）"""
    gt_has_event_val, gt_class_indices = labels.max(-1)
    is_foreground_mask = gt_has_event_val > 0.5

    # 无前景样本时返回0
    if is_foreground_mask.sum() == 0:
        return torch.tensor(0.0).cuda()

    # 预测类别
    _, pred_class_idx = event_scores.max(-1)
    pred_class_expand = pred_class_idx.unsqueeze(1).expand_as(gt_class_indices)

    # 仅计算前景样本的准确率
    pred_at_fg = pred_class_expand[is_foreground_mask]
    true_at_fg = gt_class_indices[is_foreground_mask]

    correct = pred_at_fg.eq(true_at_fg)
    if correct.numel() == 0:
        return torch.tensor(0.0).cuda()

    return correct.sum().double() * (100. / correct.numel())


# ================== 主函数 ==================
def main():
    global args, logger, writer
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0.0, 0

    # 1. 初始化记录列表（存储所有epoch的Loss和Acc）
    train_loss_list = []  # 所有训练轮次的损失
    val_acc_list = []  # 所有验证轮次的准确率

    # 2. 解析参数
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()

    # 3. 创建保存目录
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    # 4. 初始化日志
    logger = Prepare_logger(args, eval=args.evaluate)
    if not args.evaluate:
        logger.info(f'\n运行参数\n\n{json.dumps(vars(args), indent=4, ensure_ascii=False)}\n')

    # 5. 加载数据
    logger.info("===== 加载训练数据 =====")
    train_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='trainovave'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    logger.info("===== 加载验证数据 =====")
    test_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='testovave'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # 6. 初始化模型
    logger.info("===== 初始化模型 =====")
    mainModel = main_model(in_channels=10, feature_dim=512)
    mainModel = nn.DataParallel(mainModel).cuda()  # 多GPU训练

    # 7. 优化器和调度器
    optimizer = torch.optim.Adam(mainModel.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=[80, 140], gamma=0.5)

    # 8. 损失函数
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    # 9. 加载预训练模型（可选）
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"加载预训练模型: {args.resume}")
        checkpoint = torch.load(args.resume)
        mainModel.load_state_dict(checkpoint)

    # 10. 仅验证模式
    if args.evaluate:
        logger.info("===== 开始验证 =====")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, args=args, logger=logger,
                       eval_only=True)
        return

    # 11. TensorBoard日志（可选）
    writer = SummaryWriter(args.snapshot_pref)

    # 12. 主训练循环（全程不生成图片，仅记录数据）
    logger.info("===== 开始训练 =====")
    for epoch in range(args.n_epoch):
        # 训练单轮：记录当前轮损失
        train_loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch, args,
                                 logger)
        train_loss_list.append(train_loss)

        # 按验证频率验证：记录当前轮准确率
        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch, args, logger)
            val_acc_list.append(acc)

            # 保存最佳模型
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                torch.save(mainModel.state_dict(), f'{args.snapshot_pref}/best_model_epoch_{epoch + 1}.pth')
                logger.info(f"更新最佳模型：准确率 {best_accuracy:.2f}% (Epoch {best_accuracy_epoch})")

        # 学习率调度
        scheduler.step()

    # 13. 训练结束：仅生成1张包含所有epoch的汇总图
    plot_final_loss_acc_curve(
        train_loss_list=train_loss_list,
        val_acc_list=val_acc_list,
        eval_freq=args.eval_freq
    )

    # 14. 保存最终模型
    torch.save(mainModel.state_dict(), f'{args.snapshot_pref}/final_model.pth')

    # 15. 打印最终结果
    logger.info("===== 训练完成 =====")
    logger.info(f"最佳验证准确率：{best_accuracy:.2f}% (Epoch {best_accuracy_epoch})")
    logger.info(f"最终汇总曲线保存路径：{ROOT_SAVE_DIR}/loss_acc_curve_final.png")


if __name__ == '__main__':
    main()