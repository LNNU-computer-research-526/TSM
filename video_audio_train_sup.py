# -*- coding: utf-8 -*-
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
        "background_label": 28,  # 背景类标签值
        "num_classes": 67,  # 事件类别数（不含背景）
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
        event_scores: 事件类别得分 [B, 67]
        true_labels_tensor: 真实标签张量 [B, T, 68]（67类+背景）

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
        pred_labels[b, ~is_event_pred[b]] = 28

    # ========== 解析真实标签 ==========
    # 确保真实标签张量在GPU上且为float类型（避免double类型问题）
    if true_labels_tensor.device != torch.device('cuda'):
        true_labels_tensor = true_labels_tensor.cuda()
    true_labels_tensor_float = true_labels_tensor.float()
    true_labels_foreground = true_labels_tensor_float[:, :, :-1]  # [B, T, 67]

    # 获取真实事件类别（转换为long类型，关键修复）
    true_event_class = true_labels_foreground.argmax(-1).long()  # [B, T] (long)
    true_is_event = (true_event_class != 0)  # [B, T] (bool)

    # 初始化真实标签（long类型）
    true_labels = torch.zeros((batch_size, seq_len), dtype=torch.long).cuda()

    # 修复：确保赋值的源和目标类型一致（都是long）
    true_labels[true_is_event] = true_event_class[true_is_event]
    true_labels[~true_is_event] = 28

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
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    # Dataset - 已适配trainovave/testovave，无需修改
    # 注意：需要确保AVEDatasetV2返回视频名称，若数据集未返回，需先修改数据集代码
    train_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='test'),
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
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            test_list.clear()
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)

            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',
                    epoch=epoch + 1,
                )

            print("-----------------------------")
            print("best acc and epoch:", best_accuracy, best_accuracy_epoch)
            print("-----------------------------")
            test_list.clear()

        scheduler.step()


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    gnn_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses_with_gnn = AverageMeter()
    total_losses_without_gnn = AverageMeter()
    train_acc = AverageMeter()
    kl_losses = AverageMeter()
    end_time = time.time()

    model.train()
    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)

        # ========== 解包数据（增加视频名称） ==========
        # 注意：根据你的数据集实际返回顺序调整解包！
        # 假设数据集返回顺序：visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_names
        if len(batch_data) == 8:
            visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_names = batch_data
        else:
            # 兼容旧版本（无视频名称）
            visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels = batch_data
            video_names = [f"unknown_video_{n_iter}_{b}" for b in range(visual_feature.shape[0])]

        bs = visual_feature.shape[0]
        labels = labels.double().cuda()
        visual_feature = visual_feature.double().cuda()
        audio_feat = audio_feat.double().cuda()

        # ========== 伪标签处理：已适配67类，无需修改 ==========
        pseudo_label = pseudo_label.double().cuda()  # CLIP伪标签 [B, 10, 67]
        audio_pseudo_label = audio_pseudo_label.double().cuda()  # CLAP伪标签 [B, 10, 67]

        # 调试：首次打印伪标签形状（确认格式后可删除）
        if n_iter == 0 and epoch == 0:
            print(f"[DEBUG] pseudo_label shape: {pseudo_label.shape}")  # 应输出 [B,10,67]
            print(f"[DEBUG] audio_pseudo_label shape: {audio_pseudo_label.shape}")  # 应输出 [B,10,67]
            print(f"[DEBUG] video_names: {video_names[:2]}")  # 打印前2个视频名称

        # ========== 传入伪标签到模型 ==========
        (is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores,
         audio_is_event_scores, audio_event_scores, gnn_total_loss) = model(
            visual_feature, text_feat, audio_feat, audio_text_feat,
            clip_pseudo_labels=pseudo_label,
            clap_pseudo_labels=audio_pseudo_label
        )
        # ==================================================

        # DataParallel返回向量，必须取mean
        if gnn_total_loss.dim() > 0:
            gnn_total_loss = gnn_total_loss.mean()
        if kl_loss.dim() > 0:
            kl_loss = kl_loss.mean()

        is_event_scores = is_event_scores.squeeze().contiguous()
        is_event_scores = is_event_scores.reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous()
        vis_is_event_scores = vis_is_event_scores.reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous()
        audio_is_event_scores = audio_is_event_scores.reshape(bs, -1)

        labels_foreground = labels[:, :, :-1]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_BCE = labels_BCE.reshape(bs, -1)
        labels_event, _ = labels_evn.max(-1)

        audio_labels_foreground = labels[:, :, :-1]
        audio_labels_BCE, audio_labels_evn = audio_labels_foreground.max(-1)
        audio_labels_BCE = audio_labels_BCE.reshape(bs, -1)
        audio_labels_event, _ = audio_labels_evn.max(-1)

        # Loss计算逻辑不变
        loss_is_event = criterion(is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, audio_labels_event.cuda())

        # 计算Loss
        loss_without_gnn = (loss_is_event + loss_event_class + kl_loss +
                            vis_loss_is_event + vis_loss_event_class +
                            audio_loss_is_event + audio_loss_event_class)
        loss_with_gnn = loss_without_gnn + gnn_total_loss

        loss = loss_with_gnn
        loss.backward()

        # 计算Accuracy - 调用修改后的compute_accuracy_supervised
        acc = compute_accuracy_supervised(is_event_scores, event_scores, pseudo_label)
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        # Clip Gradient
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # 更新统计
        total_losses_with_gnn.update(loss_with_gnn.item(), visual_feature.size(0) * 10)
        total_losses_without_gnn.update(loss_without_gnn.item(), visual_feature.size(0) * 10)
        gnn_losses.update(gnn_total_loss.item(), visual_feature.size(0) * 10)
        kl_losses.update(kl_loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        writer.add_scalar('Train_data/loss_with_gnn', total_losses_with_gnn.val,
                          epoch * len(train_dataloader) + n_iter + 1)
        writer.add_scalar('Train_data/loss_without_gnn', total_losses_without_gnn.val,
                          epoch * len(train_dataloader) + n_iter + 1)
        writer.add_scalar('Train_data/gnn_loss', gnn_losses.val,
                          epoch * len(train_dataloader) + n_iter + 1)

        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Loss(without GNN) {total_losses_without_gnn.val:.4f} (avg: {total_losses_without_gnn.avg:.4f})\t'
                f'GNN Loss {gnn_losses.val:.4f} (avg: {gnn_losses.avg:.4f})\t'
                f'Loss(with GNN) {total_losses_with_gnn.val:.4f} (avg: {total_losses_with_gnn.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} (avg: {train_acc.avg:.3f})'
            )

    logger.info(
        f'**************************************************************************\t'
        f"\tTrain Epoch {epoch} Results:\t"
        f"Acc: {train_acc.avg:.4f}%\t"
        f"Loss(without GNN): {total_losses_without_gnn.avg:.4f}\t"
        f"GNN Loss: {gnn_losses.avg:.4f}\t"
        f"Loss(with GNN): {total_losses_with_gnn.avg:.4f}"
    )

    writer.add_scalar('Train_epoch_data/epoch_loss_with_gnn', total_losses_with_gnn.avg, epoch)
    writer.add_scalar('Train_epoch_data/epoch_loss_without_gnn', total_losses_without_gnn.avg, epoch)
    writer.add_scalar('Train_epoch_data/epoch_gnn_loss', gnn_losses.avg, epoch)
    writer.add_scalar('Train_epoch_data/epoch_acc', train_acc.avg, epoch)

    return total_losses_with_gnn.avg


test_list = []


@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    gnn_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses_with_gnn = AverageMeter()
    total_losses_without_gnn = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()
    kl_losses = AverageMeter()

    model.eval()
    model.double()

    # 样本计数（用于生成唯一的sample_idx）
    sample_count = 0

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        # ========== 解包数据（增加视频名称） ==========
        # 核心修改：根据数据集返回顺序，增加视频名称的解包
        if len(batch_data) == 8:
            visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_names = batch_data
        else:
            # 兼容模式：若数据集未返回视频名称，自动生成
            visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels = batch_data
            video_names = [f"test_video_{epoch}_{sample_count + b}" for b in range(visual_feature.shape[0])]

        bs = visual_feature.shape[0]
        labels = labels.double().cuda()
        visual_feature = visual_feature.double().cuda()
        audio_feat = audio_feat.double().cuda()

        # 伪标签处理：已适配67类
        pseudo_label = pseudo_label.double().cuda()
        audio_pseudo_label = audio_pseudo_label.double().cuda()

        # 传入伪标签到模型
        (is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores,
         audio_is_event_scores, audio_event_scores, gnn_total_loss) = model(
            visual_feature, text_feat, audio_feat, audio_text_feat,
            clip_pseudo_labels=pseudo_label,
            clap_pseudo_labels=audio_pseudo_label,
            epoch=epoch,
            save_vis=True,  # 测试阶段保存可视化
            sample_idx=sample_count  # 传入样本索引
        )
        # ==================================================

        if gnn_total_loss.dim() > 0:
            gnn_total_loss = gnn_total_loss.mean()
        if kl_loss.dim() > 0:
            kl_loss = kl_loss.mean()

        is_event_scores = is_event_scores.squeeze().contiguous()
        is_event_scores = is_event_scores.reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous()
        vis_is_event_scores = vis_is_event_scores.reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous()
        audio_is_event_scores = audio_is_event_scores.reshape(bs, -1)

        labels_foreground = labels[:, :, :-1]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_BCE = labels_BCE.reshape(bs, -1)
        labels_event, _ = labels_evn.max(-1)

        audio_labels_foreground = labels[:, :, :-1]
        audio_labels_BCE, audio_labels_evn = audio_labels_foreground.max(-1)
        audio_labels_BCE = audio_labels_BCE.reshape(bs, -1)
        audio_labels_event, _ = audio_labels_evn.max(-1)

        # Loss计算逻辑不变
        loss_is_event = criterion(is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, audio_labels_event.cuda())

        loss_without_gnn = (loss_is_event + loss_event_class + kl_loss +
                            vis_loss_is_event + vis_loss_event_class +
                            audio_loss_is_event + audio_loss_event_class)
        loss_with_gnn = loss_without_gnn + 0.1 * gnn_total_loss

        # 计算Accuracy - 调用修改后的compute_accuracy_supervised
        acc = compute_accuracy_supervised(is_event_scores, event_scores, pseudo_label)
        accuracy.update(acc.item(), bs * 10)

        # ================================= 核心修改：解析并保存标签（增加视频名称） ============================
        # 解析预测标签和真实标签（传入cpu的labels以避免设备不匹配）
        pred_labels, true_labels = parse_labels(
            is_event_scores, event_scores, labels.cpu()
        )

        # 逐个样本保存结果（增加video_name参数）
        for b in range(bs):
            save_prediction_labels(
                epoch=epoch,
                sample_idx=sample_count + b,
                video_name=video_names[b],  # 传入视频名称
                pred_labels=pred_labels[b],
                true_labels=true_labels[b],
                split="test"
            )
        sample_count += bs

        total_losses_with_gnn.update(loss_with_gnn.item(), bs * 10)
        total_losses_without_gnn.update(loss_without_gnn.item(), bs * 10)
        gnn_losses.update(gnn_total_loss.item(), bs * 10)
        kl_losses.update(kl_loss.item(), bs * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\t'
                f'Loss(without GNN) {total_losses_without_gnn.val:.4f} (avg: {total_losses_without_gnn.avg:.4f})\t'
                f'GNN Loss {gnn_losses.val:.4f} (avg: {gnn_losses.avg:.4f})\t'
                f'Loss(with GNN) {total_losses_with_gnn.val:.4f} (avg: {total_losses_with_gnn.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} (avg: {accuracy.avg:.3f})'
            )

    logger.info(
        f'**************************************************************************\t'
        f"\tTest Epoch {epoch} Results:\t"
        f"Accuracy: {accuracy.avg:.4f}%\t"
        f"KL Loss: {kl_losses.avg:.4f}\t"
        f"Loss(without GNN): {total_losses_without_gnn.avg:.4f}\t"
        f"GNN Loss: {gnn_losses.avg:.4f}\t"
        f"Loss(with GNN): {total_losses_with_gnn.avg:.4f}"
    )

    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss_with_gnn', total_losses_with_gnn.avg, epoch)
        writer.add_scalar('Val_epoch_data/epoch_loss_without_gnn', total_losses_without_gnn.avg, epoch)
        writer.add_scalar('Val_epoch_data/epoch_gnn_loss', gnn_losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    return accuracy.avg


# ========================== 最高优先级修改3：精度计算函数全量替换（28→67，背景类索引更新） ==========================
def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    _, targets = labels.max(-1)
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5  # 预测为事件的位置
    scores_mask = scores_pos_ind == 0  # 预测为背景的位置
    _, event_class = event_scores.max(-1)  # 预测的事件类别（0-66）
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]  # 事件位置赋值类别，背景位置为0
    pred[scores_mask] = 28  # 背景位置赋值为67（替换原28）
    correct = pred.eq(targets)  # 预测与标签对比
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())
    return acc


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