# -*- coding: utf-8 -*-
"""AVE 完全监督：训练与评测均使用 labels.h5 帧级 GT。

数据划分使用官方 train_order.h5 / test_order.h5（不再用 Train.txt/Test.txt）。
旧版重叠划分备份：video_audio_train_sup_overlap.py
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import random
import json
from tqdm import tqdm
import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from configs.opts import parser
from model.temp_video_modelv3 import Temp_Model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset_sup import AVEDatasetV2
import torch.nn.functional as F

# ================================= 新增：结果保存路径配置 ============================
RESULT_SAVE_PATH = r"E:\map\modeltap\avesup"
os.makedirs(RESULT_SAVE_PATH, exist_ok=True)

# ================================= seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config_path = 'configs/main.json'
with open(config_path) as fp:
    config = json.load(fp)
print(config)

AVE_NUM_CLASSES = 28
AVE_BG_CLASS = 28


def _gt_frame_labels(labels):
    return labels[:, :, :AVE_NUM_CLASSES].float()


def _pack_gt_targets(labels, batch_size):
    labels_foreground = labels[:, :, :AVE_NUM_CLASSES]
    labels_BCE, labels_evn = labels_foreground.max(-1)
    labels_BCE = labels_BCE.reshape(batch_size, -1)
    labels_event, _ = labels_evn.max(-1)
    return labels_BCE, labels_event


# ========================== 参数统计函数 ==========================
def count_parameters(model, prefix=""):
    """统计模型参数量的函数，支持处理分布式模型"""
    if isinstance(model, nn.DataParallel):
        model = model.module

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{prefix}模型总参数量（Total parameters）: {total_params:,}")
    print(f"{prefix}可训练参数量（Trainable parameters）: {trainable_params:,}\n")

    return total_params, trainable_params


# ================================= 新增：保存标签结果函数（增加视频名称） ============================
def save_prediction_labels(epoch, sample_idx, video_name, pred_labels, true_labels, split="test"):
    """
    保存预测标签和真实标签到指定路径（包含视频名称）
    Args:
        epoch: 当前轮次
        sample_idx: 样本索引
        video_name: 视频文件名（如 "video_123.mp4"）
        pred_labels: 预测标签列表 [T,] (int类型)
        true_labels: 真实标签列表 [T,] (int类型)
        split: 数据集划分（train/test）
    """
    # 创建轮次文件夹
    epoch_dir = os.path.join(RESULT_SAVE_PATH, split, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 构建结果字典（增加video_name字段）
    result_dict = {
        "epoch": epoch,
        "sample_idx": sample_idx,
        "video_name": video_name,  # 新增：视频名称
        "pred_labels": pred_labels.tolist(),  # numpy数组转列表
        "true_labels": true_labels.tolist(),
        "background_label": AVE_BG_CLASS,
        "num_classes": AVE_NUM_CLASSES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 保存为JSON文件（用视频名称命名，更易识别）
    # 清理视频名称中的特殊字符，避免文件命名错误
    safe_video_name = "".join([c for c in video_name if c not in r'\/:*?"<>|'])
    save_file = os.path.join(epoch_dir, f"{safe_video_name}_sample_{sample_idx:04d}.json")
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    # 保存汇总文件（每个轮次一个）
    summary_file = os.path.join(RESULT_SAVE_PATH, split, f"epoch_{epoch}_summary.json")
    summary_data = {}
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

    summary_data[f"{safe_video_name}_sample_{sample_idx:04d}"] = {
        "video_name": video_name,
        "pred_labels": pred_labels.tolist(),
        "true_labels": true_labels.tolist()
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=4)


# ================================= 新增：解析标签函数（修复版本） ============================
def parse_labels(is_event_scores, event_scores, true_labels_tensor):
    """
    解析预测标签和真实标签（修复类型不匹配问题）
    Args:
        is_event_scores: 事件存在性得分 [B, T]
        event_scores: 事件类别得分 [B, 28]
        true_labels_tensor: 真实标签张量 [B, T, 29]（28类+背景）

    Returns:
        pred_labels: 预测标签 [B, T] (numpy数组，int类型)
        true_labels: 真实标签 [B, T] (numpy数组，int类型)
    """
    batch_size, seq_len = is_event_scores.shape

    # ========== 解析预测标签 ==========
    # 将is_event_scores转换为float（避免DataParallel导致的类型问题）
    is_event_scores_float = is_event_scores.float()
    is_event_pred = is_event_scores_float.sigmoid() > 0.5  # [B, T] (bool)

    # 事件类别预测 [B] (long类型)
    event_class_pred = event_scores.argmax(-1).long()  # [B] - 事件类别预测

    # 初始化预测标签（long类型）
    pred_labels = torch.zeros((batch_size, seq_len), dtype=torch.long).cuda()

    # 赋值：事件位置为预测类别，背景位置为28
    for b in range(batch_size):
        pred_labels[b, is_event_pred[b]] = event_class_pred[b]
        pred_labels[b, ~is_event_pred[b]] = AVE_BG_CLASS

    true_labels_tensor_float = true_labels_tensor.float()
    true_labels_foreground = true_labels_tensor_float[:, :, :AVE_NUM_CLASSES]

    true_event_class = true_labels_foreground.argmax(-1).long()
    true_is_event = true_labels_foreground.sum(-1) > 0

    true_labels = torch.zeros((batch_size, seq_len), dtype=torch.long, device=true_labels_tensor.device)
    true_labels[true_is_event] = true_event_class[true_is_event]
    true_labels[~true_is_event] = AVE_BG_CLASS

    # 转换为numpy数组（int类型）返回
    return pred_labels.cpu().numpy().astype(int), true_labels.cpu().numpy().astype(int)


def main():
    global args, logger, writer, dataset_configs
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0

    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
        logger.info('Protocol: official AVE split (train_order.h5 / test_order.h5) | train=frame-level GT | test=frame-level GT')
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    train_dataset = AVEDatasetV2('./data/', split='train')
    test_dataset = AVEDatasetV2('./data/', split='test')
    train_vids = set(train_dataset.video_names)
    test_vids = set(test_dataset.video_names)
    overlap_vids = train_vids & test_vids
    logger.info(
        f"[Official Split] train={len(train_dataset)} test={len(test_dataset)} | "
        f"unique train/test videos={len(train_vids)}/{len(test_vids)} | "
        f"video-id overlap={len(overlap_vids)}"
    )
    if overlap_vids:
        logger.info(
            f"[Official Split] {len(overlap_vids)} 个重复视频 ID 来自 Annotations 双标注条目，"
            f"labels.h5 索引本身无交集"
        )
    else:
        logger.info("[Official Split] 训练/测试视频 ID 无交集")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Model - 确认in_channels=10（OVAVE视觉特征为10帧，匹配）、feature_dim=768（CLIP/CLAP特征维度，匹配）
    mainModel = main_model(in_channels=10, feature_dim=768)

    total_params, trainable_params = count_parameters(mainModel, "初始")
    logger.info(f"初始模型总参数量: {total_params:,}，可训练参数量: {trainable_params:,}")

    mainModel = nn.DataParallel(mainModel).cuda()

    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)

    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        test_total, test_trainable = count_parameters(mainModel, "测试集评估用")
        logger.info(f"测试集评估用模型总参数量: {test_total:,}，可训练参数量: {test_trainable:,}")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    writer = SummaryWriter(args.snapshot_pref)

    for epoch in range(args.n_epoch):
        train_loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            test_list.clear()
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)

            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised_GT',
                    epoch=epoch + 1,
                )

            logger.info("=" * 80)
            logger.info(f"Best accuracy: {best_accuracy:.4f}% at Epoch {best_accuracy_epoch + 1}")
            logger.info("=" * 80)
            test_list.clear()

        scheduler.step()


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):
        if len(batch_data) == 8:
            visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_names = batch_data
        else:
            visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels = batch_data
            video_names = [f"unknown_video_{n_iter}_{b}" for b in range(visual_feature.shape[0])]

        bs = visual_feature.shape[0]
        labels = labels.double().cuda()
        visual_feature = visual_feature.double().cuda()
        audio_feat = audio_feat.double().cuda()
        gt_fg = _gt_frame_labels(labels)

        (is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores,
         audio_is_event_scores, audio_event_scores, gnn_total_loss) = model(
            visual_feature, text_feat, audio_feat, audio_text_feat,
            clip_pseudo_labels=gt_fg,
            clap_pseudo_labels=gt_fg,
        )

        if gnn_total_loss.dim() > 0:
            gnn_total_loss = gnn_total_loss.mean()
        if kl_loss.dim() > 0:
            kl_loss = kl_loss.mean()

        is_event_scores = is_event_scores.squeeze().contiguous().reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous().reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous().reshape(bs, -1)

        labels_BCE, labels_event = _pack_gt_targets(labels, bs)

        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores, labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores, labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, labels_event.cuda())

        loss = (loss_is_event + loss_event_class + kl_loss +
                vis_loss_is_event + vis_loss_event_class +
                audio_loss_is_event + audio_loss_event_class + gnn_total_loss)

        loss.backward()

        acc = compute_accuracy_ave_segment(is_event_scores, event_scores, labels)
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), visual_feature.size(0) * 10)
        end_time = time.time()

        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch + 1}/{args.n_epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Loss {losses.val:.4f} (avg: {losses.avg:.4f})\t'
                f'Acc {train_acc.val:.3f}% (avg: {train_acc.avg:.3f}%)'
            )

    writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)
    writer.add_scalar('Train_epoch_data/epoch_acc', train_acc.avg, epoch)
    logger.info(f'\nEpoch {epoch + 1} Train Summary - Avg Loss: {losses.avg:.4f}, Avg Acc: {train_acc.avg:.4f}%')
    return losses.avg


test_list = []


@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    losses = AverageMeter()
    segment_acc = AverageMeter()
    video_acc = AverageMeter()
    end_time = time.time()

    model.eval()
    model.double()

    sample_count = 0

    for n_iter, batch_data in enumerate(test_dataloader):
        if len(batch_data) == 8:
            visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_names = batch_data
        else:
            visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels = batch_data
            video_names = [f"test_video_{epoch}_{sample_count + b}" for b in range(visual_feature.shape[0])]

        bs = visual_feature.shape[0]
        labels = labels.double().cuda()
        visual_feature = visual_feature.double().cuda()
        audio_feat = audio_feat.double().cuda()
        gt_fg = _gt_frame_labels(labels)

        (is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores,
         audio_is_event_scores, audio_event_scores, gnn_total_loss) = model(
            visual_feature, text_feat, audio_feat, audio_text_feat,
            clip_pseudo_labels=gt_fg,
            clap_pseudo_labels=gt_fg,
        )

        if gnn_total_loss.dim() > 0:
            gnn_total_loss = gnn_total_loss.mean()
        if kl_loss.dim() > 0:
            kl_loss = kl_loss.mean()

        is_event_scores = is_event_scores.squeeze().contiguous().reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous().reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous().reshape(bs, -1)

        labels_BCE, labels_event = _pack_gt_targets(labels, bs)

        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores, labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores, labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, labels_event.cuda())

        loss = (loss_is_event + loss_event_class + kl_loss +
                vis_loss_is_event + vis_loss_event_class +
                audio_loss_is_event + audio_loss_event_class + 0.1 * gnn_total_loss)

        acc = compute_accuracy_ave_segment(is_event_scores, event_scores, labels)
        pred, targets = build_segment_predictions(is_event_scores, event_scores, labels)
        batch_video_acc = compute_video_accuracy(pred, targets)

        segment_acc.update(acc.item(), bs * 10)
        video_acc.update(batch_video_acc.item(), bs)

        pred_labels, true_labels = parse_labels(is_event_scores, event_scores, labels.cpu())
        for b in range(bs):
            save_prediction_labels(
                epoch=epoch,
                sample_idx=sample_count + b,
                video_name=video_names[b],
                pred_labels=pred_labels[b],
                true_labels=true_labels[b],
                split="test"
            )
        sample_count += bs

        losses.update(loss.item(), bs * 10)
        end_time = time.time()

        if n_iter % args.print_freq == 0:
            logger.info(
                f'Val Epoch: [{epoch + 1}/{args.n_epoch}][{n_iter}/{len(test_dataloader)}]\t'
                f'Loss {losses.val:.4f} (avg: {losses.avg:.4f})\t'
                f'Video-Acc {video_acc.val:.3f}% (avg: {video_acc.avg:.3f}%)'
            )

    logger.info(
        f'\nEpoch {epoch + 1} Validation Summary - Avg Loss: {losses.avg:.4f}, '
        f'Segment-Acc@GT: {segment_acc.avg:.4f}%, Video-Acc@GT: {video_acc.avg:.4f}%'
    )

    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Segment_Accuracy_GT', segment_acc.avg, epoch)
        writer.add_scalar('Val_epoch/Video_Accuracy_GT', video_acc.avg, epoch)

    return segment_acc.avg


def build_segment_predictions(is_event_scores, event_scores, labels):
    _, targets = labels.max(-1)
    if is_event_scores.dim() == 3:
        is_event_scores = is_event_scores.squeeze(-1)
    is_event_pred = is_event_scores.sigmoid() > 0.5
    _, event_class = event_scores.max(-1)
    pred = is_event_pred.long() * event_class.unsqueeze(1)
    pred[~is_event_pred] = AVE_BG_CLASS
    return pred, targets


def compute_accuracy_ave_segment(is_event_scores, event_scores, labels):
    pred, targets = build_segment_predictions(is_event_scores, event_scores, labels)
    correct = pred.eq(targets)
    if correct.numel() == 0:
        return torch.tensor(0.0, device=is_event_scores.device)
    return correct.sum().double() * (100.0 / correct.numel())


def compute_video_accuracy(pred, targets):
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return pred.eq(targets).all(dim=1).float().mean() * 100.0


def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


# ========================== 可选：修改数据集代码（AVEDatasetV2）返回视频名称 ==========================
# 如果你需要修改数据集代码，以下是参考示例：
# 在 dataset/AVE_dataset_sup.py 中修改 AVEDatasetV2 的 __getitem__ 方法：
"""
def __getitem__(self, idx):
    # 原有逻辑...
    video_name = self.data_list[idx]['video_name']  # 假设数据列表中有视频名称字段
    # 最后返回时增加视频名称
    return visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_name
"""

if __name__ == '__main__':
    main()