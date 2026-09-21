# -*- coding: utf-8 -*-
import json
import os
import random
import time

import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from configs.opts import parser
from model.temp_video_model_aveweak import Temp_Model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from dataset.AVE_dataset_aveweak import AVEDatasetV2

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


def _pack_pseudo_targets(pseudo_label, audio_pseudo_label, batch_size):
    labels_foreground = pseudo_label
    labels_BCE, labels_evn = labels_foreground.max(-1)
    labels_BCE = labels_BCE.reshape(batch_size, -1)
    labels_event, _ = labels_evn.max(-1)

    audio_labels_foreground = audio_pseudo_label
    audio_labels_BCE, audio_labels_evn = audio_labels_foreground.max(-1)
    audio_labels_BCE = audio_labels_BCE.reshape(batch_size, -1)
    audio_labels_event, _ = audio_labels_evn.max(-1)
    return labels_BCE, labels_event, audio_labels_BCE, audio_labels_event

ROW1_SNAPSHOT = os.path.join('.', 'Exps', 'aveweak_table6', 'row1_kl_even')


def loss_combo_name(a):
    parts = []
    if a.use_loss_kl:
        parts.append('L_KL')
    if a.use_loss_va:
        parts.append('L^(v/a)')
    if a.use_loss_even:
        parts.append('L^(even)')
    if a.use_loss_class:
        parts.append('L^(class)')
    if a.use_loss_st:
        parts.append('L_st')
    return ' + '.join(parts) if parts else '(none)'


def is_table6_row1(a):
    return (a.use_loss_kl and a.use_loss_even
            and (not a.use_loss_va) and (not a.use_loss_class)
            and (not a.use_loss_st) and (not a.enable_gnn))


def compose_train_loss(loss_is_event, loss_event_class, kl_loss, gnn_loss,
                       vis_loss_is_event, vis_loss_event_class,
                       audio_loss_is_event, audio_loss_event_class):
    loss = kl_loss.new_tensor(0.0)
    if args.use_loss_even:
        loss = loss + loss_is_event
    if args.use_loss_class:
        loss = loss + loss_event_class
    if args.use_loss_va:
        loss = loss + vis_loss_is_event + audio_loss_is_event
        if args.use_loss_class:
            loss = loss + vis_loss_event_class + audio_loss_event_class
    if args.use_loss_kl:
        loss = loss + kl_loss
    if args.use_loss_st:
        loss = loss + gnn_loss
    return loss


def main():
    global args, logger, writer, dataset_configs
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0

    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()

    # Table VI 第1行默认不要覆盖原来的 80% full 实验目录
    snap = str(args.snapshot_pref).replace('\\', '/').rstrip('/')
    if is_table6_row1(args) and snap in ('.', './exp/debug/123', 'exp/debug/123'):
        args.snapshot_pref = ROW1_SNAPSHOT

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
        logger.info('Protocol: train=frame-level pseudo | test=frame-level GT')
        logger.info('Table VI loss combo: {}'.format(loss_combo_name(args)))
        logger.info('enable_gnn={}'.format(args.enable_gnn))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    train_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    mainModel = main_model(in_channels=10, feature_dim=768)
    mainModel = nn.DataParallel(mainModel).cuda()
    optimizer = torch.optim.Adam(mainModel.parameters(), lr=args.lr)
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
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    writer = SummaryWriter(args.snapshot_pref)

    for epoch in range(args.n_epoch):
        train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)
        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='PseudoLabel_EvalGT',
                    epoch=epoch + 1,
                )
                write_train_summary()
            print("-----------------------------")
            print("best acc and epoch:", best_accuracy, best_accuracy_epoch)
            print("loss combo:", loss_combo_name(args))
            print("-----------------------------")

    write_train_summary({'finished': True})


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)

        visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, _ = batch_data
        bs = visual_feature.shape[0]
        labels = labels.float().cuda()
        pseudo_label = pseudo_label.float().cuda()
        visual_feature = visual_feature.float().cuda()
        text_feat = text_feat.float().cuda()
        audio_feat = audio_feat.float().cuda()
        audio_text_feat = audio_text_feat.float().cuda()
        audio_pseudo_label = audio_pseudo_label.float().cuda()

        (is_event_scores, event_scores, kl_loss, gnn_loss,
         vis_is_event_scores, vis_event_scores,
         audio_is_event_scores, audio_event_scores) = model(
            visual_feature, text_feat, audio_feat, audio_text_feat,
            clip_pseudo_labels=pseudo_label if args.enable_gnn else None,
            clap_pseudo_labels=audio_pseudo_label if args.enable_gnn else None,
            enable_gnn=args.enable_gnn,
        )

        if n_iter == 0:
            print("=== train: frame-level pseudo ===")
            print("pseudo_label:", tuple(pseudo_label.shape), "GT(test only):", tuple(labels.shape))
            print("loss combo:", loss_combo_name(args), "| enable_gnn:", args.enable_gnn)

        is_event_scores = is_event_scores.squeeze().contiguous().reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous().reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous().reshape(bs, -1)

        (labels_BCE, labels_event,
         audio_labels_BCE, audio_labels_event) = _pack_pseudo_targets(
            pseudo_label, audio_pseudo_label, bs)

        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores, labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores, audio_labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, audio_labels_event.cuda())

        loss = compose_train_loss(
            loss_is_event, loss_event_class, kl_loss, gnn_loss,
            vis_loss_is_event, vis_loss_event_class,
            audio_loss_is_event, audio_loss_event_class,
        )

        if not torch.isfinite(loss):
            logger.info('Skip non-finite loss at train iter {}'.format(n_iter))
            optimizer.zero_grad()
            continue

        loss.backward()

        acc_result = compute_accuracy_supervised(is_event_scores, event_scores, pseudo_label)
        train_acc.update(acc_result[0].item(), bs * 10)

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), bs * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@pseudo {train_acc.val:.3f} ({train_acc.avg:.3f})'
            )

    writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)
    logger.info(
        f'**************************************************************************\t'
        f"\tTrain results (acc vs pseudo): {train_acc.avg:.4f}%."
    )
    return losses.avg


@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()
    model.eval()

    class_event_stats = {}
    class_video_stats = {}
    debug_total_samples = 0

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_id = batch_data
        bs = visual_feature.shape[0]
        debug_total_samples += bs

        labels = labels.float().cuda()
        pseudo_label = pseudo_label.float().cuda()
        visual_feature = visual_feature.float().cuda()
        text_feat = text_feat.float().cuda()
        audio_feat = audio_feat.float().cuda()
        audio_text_feat = audio_text_feat.float().cuda()
        audio_pseudo_label = audio_pseudo_label.float().cuda()

        (is_event_scores, event_scores, kl_loss, gnn_loss,
         vis_is_event_scores, vis_event_scores,
         audio_is_event_scores, audio_event_scores) = model(
            visual_feature, text_feat, audio_feat, audio_text_feat,
            clip_pseudo_labels=pseudo_label if args.enable_gnn else None,
            clap_pseudo_labels=audio_pseudo_label if args.enable_gnn else None,
            enable_gnn=args.enable_gnn,
        )

        is_event_scores = is_event_scores.squeeze().contiguous().reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous().reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous().reshape(bs, -1)

        (labels_BCE, labels_event,
         audio_labels_BCE, audio_labels_event) = _pack_pseudo_targets(
            pseudo_label, audio_pseudo_label, bs)

        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores, labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores, audio_labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, audio_labels_event.cuda())

        loss = compose_train_loss(
            loss_is_event, loss_event_class, kl_loss, gnn_loss,
            vis_loss_is_event, vis_loss_event_class,
            audio_loss_is_event, audio_loss_event_class,
        )

        acc_result = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        acc, pred, targets = acc_result
        accuracy.update(acc.item(), bs * 10)

        for i in range(bs):
            current_category = video_id[i].split('&')[0]
            sample_pred = pred[i].cpu().numpy()
            sample_target = targets[i].cpu().numpy()

            if current_category not in class_event_stats:
                class_event_stats[current_category] = [0, 0]
            class_event_stats[current_category][0] += int(np.sum(sample_pred == sample_target))
            class_event_stats[current_category][1] += len(sample_target)

            is_video_correct = bool(np.all(sample_pred == sample_target))
            if current_category not in class_video_stats:
                class_video_stats[current_category] = [0, 0]
            class_video_stats[current_category][0] += 1 if is_video_correct else 0
            class_video_stats[current_category][1] += 1

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@GT {accuracy.val:.3f} ({accuracy.avg:.3f})'
            )

    total_video_correct = sum(v[0] for v in class_video_stats.values())
    total_video_count = sum(v[1] for v in class_video_stats.values())
    overall_video_acc = (total_video_correct / total_video_count * 100) if total_video_count else 0.0

    logger.info(f"\n===== 数据完整性校验 =====")
    logger.info(f"test samples: {debug_total_samples}")
    logger.info(f"整体视频级准确率(全部帧正确@GT): {overall_video_acc:.2f}%")
    logger.info(f"整体事件级准确率(GT): {accuracy.avg:.4f}%")

    output_dir = os.path.join(args.snapshot_pref, "test_results_per_epoch")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"epoch_{epoch}_test_stats.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"===== Epoch {epoch} PseudoTrain / GT Eval =====\n")
        f.write(f"整体事件级准确率(segment acc @ GT): {accuracy.avg:.4f}%\n")
        f.write(f"整体视频级准确率(全部帧正确): {overall_video_acc:.2f}%\n")
        f.write("\n--- Per-category segment acc ---\n")
        for cat in sorted(class_event_stats.keys()):
            c, t = class_event_stats[cat]
            f.write(f"{cat}\t{c}/{t}\t{100.0*c/t:.2f}%\n")

    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy_GT', accuracy.avg, epoch)
        writer.add_scalar('Val_epoch/Video_Accuracy_GT', overall_video_acc, epoch)

    logger.info(
        f'**************************************************************************\t'
        f"\tEvaluation results (acc @ GT): {accuracy.avg:.4f}%."
    )
    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    _, targets = labels.max(-1)
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0
    _, event_class = event_scores.max(-1)
    pred = scores_pos_ind.long()
    pred = pred * event_class[:, None]
    pred[scores_mask] = AVE_BG_CLASS
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())
    return acc, pred, targets


def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':
    main()
