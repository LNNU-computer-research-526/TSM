# -*- coding: utf-8 -*-
import os
import time
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

# 直接导入你已经改好的、没有h5文件的数据集类
from dataset.ovaveweak import AVEDatasetV2
import torch.nn.functional as F

# ================== 关键修改1：导入模型中的绘图函数 ==================
from model.modelovaveweak import plot_loss_acc_curve, ROOT_SAVE_DIR

# ================== 全局绘图变量 ==================
epoch_history = {
    'epochs': [],
    'train_loss': [],
    'val_acc': []
}

# ================== 常量定义 ==================
OVAVE_NUM_CLASSES = 67  # OVAVE 类别数
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================== 环境设置 ==================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

# ================== 自定义整理函数 ==================
def custom_collate_fn(batch):
    transposed = list(zip(*batch))
    # 强制将所有数据转为 float32，避免 Long/Float 混合导致的崩溃
    visual_feature = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[0]])
    text_feat = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[1]])
    pseudo_label = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[2]])
    audio_feat = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[3]])
    audio_text_feat = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[4]])
    audio_pseudo_label = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[5]])
    labels = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in transposed[6]])

    # 兼容原有代码的video_id输出
    video_ids = [f"class_{i}&video_{idx}" for idx, i in
                 enumerate(torch.argmax(labels[:, 0, :OVAVE_NUM_CLASSES], dim=1))]

    return visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_ids

# ================== 工具类 ==================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

import logging
def Prepare_logger(args, eval=False):
    """创建日志记录器"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 清除原有处理器
    logger.handlers.clear()

    # 设置日志保存路径
    log_dir = args.snapshot_pref
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'Eval.log' if eval else 'Train.log')

    # 文件处理器
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式设置
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# ================== 参数类 ==================
class Args:
    def __init__(self):
        # 核心参数
        self.n_epoch = 100
        self.batch_size = 16
        self.test_batch_size = 16
        self.lr = 0.001
        self.gpu = None
        # 结果保存路径（修改为你的指定路径）
        self.snapshot_pref = r"E:\map\modeltap\ovaveweak"
        self.resume = ""
        self.evaluate = False
        self.clip_gradient = 0.8
        self.loss_weights = 0.5
        self.start_epoch = 0
        self.weight_decay = 0.0005
        self.print_freq = 20
        self.save_freq = None
        self.eval_freq = 1

# ================== 关键修改2：删除重复的plot_training_summary函数 ==================
# （直接使用model中的plot_loss_acc_curve，无需自定义）

# ================== 主函数 ==================
def main():
    # 全局变量初始化
    global args, logger, writer, dataset_configs
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0

    # 加载配置
    config_path = 'configs/main.json'
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as fp:
            config = json.load(fp)
    print("Loaded config:", config)

    # 初始化参数
    args = Args()
    dataset_configs = {}

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 创建保存目录
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    # 初始化日志
    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be saved in: {args.snapshot_pref}/Eval.log')

    # 加载数据集
    logger.info("\nLoading OVAVE dataset...")
    train_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='trainovave'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    test_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='testovave'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    logger.info(f"Dataset loaded - Train samples: {len(train_dataloader)}, Test samples: {len(test_dataloader)}")

    # 初始化模型
    from model.modelovaveweak import Temp_Model as main_model
    mainModel = main_model(in_channels=10, feature_dim=512)  # OVAVE使用512维特征
    mainModel = nn.DataParallel(mainModel).cuda()

    # 优化器和调度器
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[80, 140], gamma=0.5)

    # 损失函数
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    # 加载预训练模型
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")

    # 仅评估模式
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    # Tensorboard初始化
    try:
        writer = SummaryWriter(args.snapshot_pref)
    except Exception as e:
        writer = None
        logger.warning(f"Failed to create SummaryWriter: {e}")

    # 训练循环
    logger.info("\nStarting training...")
    global test_list
    test_list = []

    for epoch in range(args.n_epoch):
        # 训练一个epoch
        train_loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        # 验证并生成图表
        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            test_list.clear()
            val_acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)

            # 记录epoch数据
            epoch_history['epochs'].append(epoch + 1)
            epoch_history['train_loss'].append(train_loss)
            # 关键修改3：val_acc是百分比（如85.5），无需除以100，模型函数会自动处理
            epoch_history['val_acc'].append(val_acc)

            # 关键修改4：调用模型中的plot_loss_acc_curve函数生成每轮曲线
            # 方式1：按epoch命名保存（推荐，保留每轮曲线）
            save_path = os.path.join(ROOT_SAVE_DIR, f"loss_acc_curve_epoch{epoch+1}.png")
            plot_loss_acc_curve(
                train_loss_list=epoch_history['train_loss'],
                val_acc_list=epoch_history['val_acc'],
                save_path=save_path  # 传入按epoch命名的保存路径
            )

            # 方式2：同时保存汇总曲线（覆盖式，始终保留最新完整曲线）
            plot_loss_acc_curve(
                train_loss_list=epoch_history['train_loss'],
                val_acc_list=epoch_history['val_acc']
            )

            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='WeakSupervision',
                    epoch=epoch + 1,
                )

            # 打印最佳结果
            logger.info("=" * 80)
            logger.info(f"Best accuracy: {best_accuracy:.4f}% at Epoch {best_accuracy_epoch + 1}")
            logger.info("=" * 80)
            test_list.clear()

        # 更新学习率
        scheduler.step()

    # 训练结束后生成最终汇总图
    plot_loss_acc_curve(
        train_loss_list=epoch_history['train_loss'],
        val_acc_list=epoch_history['val_acc'],
        save_path=os.path.join(ROOT_SAVE_DIR, "loss_acc_curve_final.png")
    )
    logger.info(f"\nTraining completed! Final summary plot saved to: {os.path.join(ROOT_SAVE_DIR, 'loss_acc_curve_final.png')}")

# ================== 训练函数 ==================
def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)

        # 解析batch数据
        visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_id = batch_data
        bs = visual_feature.shape[0]

        # 数据移至GPU
        labels = labels.double().cuda()
        pseudo_label = pseudo_label.double().cuda()
        visual_feature = visual_feature.double().cuda()
        audio_feat = audio_feat.double().cuda()
        audio_pseudo_label = audio_pseudo_label.double().cuda()

        # 模型前向传播
        try:
            model_outputs = model(
                visual_feature, text_feat, audio_feat, audio_text_feat,
                clip_pseudo_labels=pseudo_label,
                clap_pseudo_labels=audio_pseudo_label
            )
        except Exception as e:
            logger.warning(f"Model forward error: {e}, using fallback forward")
            model_outputs = model(
                visual_feature, text_feat, audio_feat, audio_text_feat
            )

        # 解析模型输出
        if len(model_outputs) == 8:
            is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores, audio_is_event_scores, audio_event_scores, gnn_total_loss = model_outputs
            kl_loss = kl_loss if kl_loss.dim() == 0 else kl_loss.mean()
        elif len(model_outputs) == 7:
            is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores, audio_is_event_scores, audio_event_scores = model_outputs
        else:
            raise ValueError(f"Invalid model output length: {len(model_outputs)}")

        # 处理输出形状
        is_event_scores = is_event_scores.squeeze().contiguous()
        is_event_scores = is_event_scores.reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous()
        vis_is_event_scores = vis_is_event_scores.reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous()
        audio_is_event_scores = audio_is_event_scores.reshape(bs, -1)

        # 标签处理
        labels_foreground = labels[:, :, :OVAVE_NUM_CLASSES]
        labels_BCE, _ = labels_foreground.max(-1)
        labels_BCE = labels_BCE.reshape(bs, -1)

        # 视频级标签提取（弱监督核心）
        video_level_vec, _ = labels_foreground.max(1)
        gt_val, gt_idx = video_level_vec.max(-1)
        labels_event = gt_idx.long()

        # 音频伪标签处理
        audio_labels_BCE, _ = audio_pseudo_label.max(-1)
        audio_labels_BCE = audio_labels_BCE.reshape(bs, -1)
        audio_video_level_vec, _ = audio_pseudo_label.max(1)
        _, audio_gt_idx = audio_video_level_vec.max(-1)
        audio_labels_event = audio_gt_idx.long()

        # 计算损失
        valid_event_mask = gt_val > 0.5

        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores, labels_BCE.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores, audio_labels_BCE.cuda())

        if valid_event_mask.sum() > 0:
            loss_event_class = criterion_event(event_scores[valid_event_mask], labels_event[valid_event_mask])
            vis_loss_event_class = criterion_event(vis_event_scores[valid_event_mask], labels_event[valid_event_mask])
            audio_loss_event_class = criterion_event(audio_event_scores[valid_event_mask],
                                                     audio_labels_event[valid_event_mask])
        else:
            loss_event_class = torch.tensor(0.0).cuda()
            vis_loss_event_class = torch.tensor(0.0).cuda()
            audio_loss_event_class = torch.tensor(0.0).cuda()

        # 总损失
        kl_loss = kl_loss if kl_loss is not None else torch.tensor(0.0).cuda()
        loss = (loss_is_event + loss_event_class + kl_loss +
                vis_loss_is_event + vis_loss_event_class +
                audio_loss_is_event + audio_loss_event_class)

        # 反向传播
        loss.backward()

        # 计算准确率
        acc_result = compute_accuracy_supervised(is_event_scores, event_scores, labels_foreground)
        acc = acc_result[0]
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        # 梯度裁剪
        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        # 更新参数
        optimizer.step()
        optimizer.zero_grad()

        # 更新统计
        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # Tensorboard记录
        if writer is not None:
            writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        # 打印日志
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch + 1}/{args.n_epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Loss {losses.val:.4f} (avg: {losses.avg:.4f})\t'
                f'Acc {train_acc.val:.3f}% (avg: {train_acc.avg:.3f}%)'
            )

    # 记录epoch级损失
    if writer is not None:
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    logger.info(f'\nEpoch {epoch + 1} Train Summary - Avg Loss: {losses.avg:.4f}, Avg Acc: {train_acc.avg:.4f}%')
    return losses.avg

# ================== 验证函数 ==================
@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    model.double()

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        # 解析batch数据
        visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_id = batch_data
        bs = visual_feature.shape[0]

        # 数据移至GPU
        labels = labels.double().cuda()
        pseudo_label = pseudo_label.double().cuda()
        visual_feature = visual_feature.double().cuda()
        audio_feat = audio_feat.double().cuda()
        audio_pseudo_label = audio_pseudo_label.double().cuda()

        # 模型前向
        try:
            model_outputs = model(
                visual_feature, text_feat, audio_feat, audio_text_feat,
                clip_pseudo_labels=pseudo_label,
                clap_pseudo_labels=audio_pseudo_label,
                epoch=epoch
            )
        except:
            model_outputs = model(
                visual_feature, text_feat, audio_feat, audio_text_feat,
                epoch=epoch
            )

        # 解析输出
        if len(model_outputs) == 8:
            is_event_scores, event_scores, kl_loss, _, _, _, _, gnn_total_loss = model_outputs
        elif len(model_outputs) == 7:
            is_event_scores, event_scores, kl_loss, _, _, _, _ = model_outputs
        else:
            raise ValueError(f"Invalid model output length: {len(model_outputs)}")

        # 处理输出形状
        is_event_scores = is_event_scores.squeeze().contiguous()
        is_event_scores = is_event_scores.reshape(bs, -1)

        # 标签处理
        labels_foreground = labels[:, :, :OVAVE_NUM_CLASSES]
        labels_BCE, _ = labels_foreground.max(-1)
        labels_BCE = labels_BCE.reshape(bs, -1)

        # 视频级标签提取
        video_level_vec, _ = labels_foreground.max(1)
        gt_val, gt_idx = video_level_vec.max(-1)
        labels_event = gt_idx.long()

        # 音频标签处理
        audio_labels_BCE, _ = audio_pseudo_label.max(-1)
        audio_labels_BCE = audio_labels_BCE.reshape(bs, -1)

        # 计算损失
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        valid_event_mask = gt_val > 0.5

        if valid_event_mask.sum() > 0:
            loss_event_class = criterion_event(event_scores[valid_event_mask], labels_event[valid_event_mask])
        else:
            loss_event_class = torch.tensor(0.0).cuda()

        kl_loss = kl_loss if kl_loss is not None else torch.tensor(0.0).cuda()
        loss = loss_is_event + loss_event_class + kl_loss

        # 计算视频级准确率
        acc_result = compute_accuracy_supervised(is_event_scores, event_scores, labels_foreground)
        acc, pred, targets = acc_result
        accuracy.update(acc.item(), bs * 10)

        # 更新统计
        losses.update(loss.item(), bs * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # 打印日志
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Val Epoch: [{epoch + 1}/{args.n_epoch}][{n_iter}/{len(test_dataloader)}]\t'
                f'Loss {losses.val:.4f} (avg: {losses.avg:.4f})\t'
                f'Video-Acc {accuracy.val:.3f}% (avg: {accuracy.avg:.3f}%)'
            )

    # Tensorboard记录
    if not eval_only and writer is not None:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Video_Accuracy', accuracy.avg, epoch)

    logger.info(
        f'\nEpoch {epoch + 1} Validation Summary - Avg Loss: {losses.avg:.4f}, Video-level Accuracy: {accuracy.avg:.4f}%')
    return accuracy.avg

# ================== 准确率计算函数（视频级） ==================
def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    """
    计算视频级准确率：判断模型是否正确识别视频包含的事件类别
    """
    # 获取视频级真实标签（弱监督核心）
    gt_has_event_val, gt_class_indices = labels.max(-1)
    is_foreground_mask = gt_has_event_val > 0.5

    if is_foreground_mask.sum() == 0:
        return (torch.tensor(0.0).cuda(), torch.zeros_like(gt_class_indices), gt_class_indices)

    # 计算预测
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0

    _, event_class = event_scores.max(-1)
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]
    pred[scores_mask] = OVAVE_NUM_CLASSES  # 背景类别

    # 计算视频级准确率
    correct = pred.eq(gt_class_indices)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return (acc, pred, gt_class_indices)

# ================== 保存模型函数 ==================
def save_checkpoint(state_dict, top1, task, epoch):
    """保存最佳模型"""
    model_name = f'{args.snapshot_pref}/best_model_epoch{epoch}_acc{top1:.3f}%.pth.tar'
    torch.save(state_dict, model_name)
    logger.info(f"Best model saved to: {model_name}")

# ================== 程序入口 ==================
if __name__ == '__main__':
    main()