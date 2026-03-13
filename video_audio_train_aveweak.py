# -*- coding: utf-8 -*-
import os
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
from model.temp_video_model_aveweak import Temp_Model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset_aveweak import AVEDatasetV2
import torch.nn.functional as F

# =================================  seed config ============================
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


# =============================================================================
#
# def AVPSLoss(av_simm, soft_label):
#     """audio-visual pair similarity loss for fully supervised setting,
#     please refer to Eq.(8, 9) in our paper.
#     """
#     # av_simm: [bs, 10]
#     relu_av_simm = F.relu(av_simm)
#     sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
#     avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
#     loss = nn.MSELoss()(avg_av_simm, soft_label)
#     return loss


def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    '''Create snapshot_pred dir for copying code and saving models '''
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

    '''Dataset'''
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
    '''model setting'''
    mainModel = main_model(in_channels=10,feature_dim=768)
    # mainModel.register_debug_hook()
    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    # recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    # recorder.writeopt(args)

    '''Training and Testing'''
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
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    kl_losses = AverageMeter()
    end_time = time.time()

    model.train()
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.
    #model.double()
    optimizer.zero_grad()


    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        # 更改 原
        # visual_feature, text_feat, pseudo_label, audio_feat , audio_text_feat, audio_pseudo_label, labels = batch_data
        visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, _ = batch_data
        bs = visual_feature.shape[0]
        labels = labels.float().cuda()
        pseudo_label = pseudo_label.float().cuda()
        visual_feature = visual_feature.float().cuda()
        audio_feat = audio_feat.float().cuda()
        audio_pseudo_label = audio_pseudo_label.float().cuda()
        is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores, audio_is_event_scores, audio_event_scores  = model(visual_feature, text_feat,
                                                                                                                                          audio_feat, audio_text_feat)
        # ✅ 调试打印放在这里（解包之后，只打印第一个batch）
        if n_iter == 0:
            print(f"=== 特征维度调试 ===")
            print(f"visual_feature shape: {visual_feature.shape}")
            print(f"text_feat shape: {text_feat.shape}")
            print(f"audio_feat shape: {audio_feat.shape}")
            print(f"audio_text_feat shape: {audio_text_feat.shape}")
        is_event_scores = is_event_scores.squeeze().contiguous()
        is_event_scores = is_event_scores.reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous()
        vis_is_event_scores = vis_is_event_scores.reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous()
        audio_is_event_scores = audio_is_event_scores.reshape(bs, -1)
        # audio_visual_gate = audio_visual_gate.transpose(1, 0).squeeze().contiguous()

        labels_foreground = pseudo_label
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_BCE = labels_BCE.reshape(bs,-1)
        labels_event, _ = labels_evn.max(-1)

        audio_labels_foreground = audio_pseudo_label
        audio_labels_BCE, audio_labels_evn = audio_labels_foreground.max(-1)
        audio_labels_BCE = audio_labels_BCE.reshape(bs, -1)
        audio_labels_event, _ = audio_labels_evn.max(-1)

        loss_is_event = criterion(is_event_scores.reshape(bs,-1), labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores.reshape(bs, -1), audio_labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, audio_labels_event.cuda())

        loss = loss_is_event + loss_event_class + kl_loss + vis_loss_is_event + vis_loss_event_class + audio_loss_is_event + audio_loss_event_class

        loss.backward()
#更改 原
        # '''Compute Accuracy'''
        # acc = compute_accuracy_supervised(is_event_scores, event_scores, pseudo_label)
        # train_acc.update(acc.item(), visual_feature.size(0) * 10)
        '''Compute Accuracy'''
        acc_result = compute_accuracy_supervised(is_event_scores, event_scores, pseudo_label)
        acc = acc_result[0]
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()


        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})'
            )

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)
    logger.info(
            f'**************************************************************************\t'
            f"\tTrain results (acc): {train_acc.avg:.4f}%."

        )
    return losses.avg

test_list = []
@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()
    kl_losses = AverageMeter()
    model.eval()
    # 更改 增加 核心：类别统计字典（键：类别名，值：[正确样本数, 总样本数]）
    class_event_stats = {}
    # 新增：视频级统计（修改逻辑：至少一个时间步预测正确即为视频正确）
    class_video_stats = {}  # {类别名: [正确视频数, 总视频数]}
    # 调试用：记录所有类别和总事件数，用于对比test集
    debug_total_samples = 0  # 所有类别的总事件数（代码统计）
    debug_total_events = 0  # 加载的总样本数
    debug_all_categories = set()  # 解析到的所有类别名（用于校验类别匹配）
    # 第一步：先获取所有解析的类别名（用于后续校验）
    for batch_data in test_dataloader:
        _, _, _, _, _, _, _, video_id = batch_data
        for id in video_id:
            category = id.split('&')[0]
            debug_all_categories.add(category)
    logger.info(f"\n===== 类别名校验 =====")
    logger.info(f"解析到的所有类别（共{len(debug_all_categories)}个）：{sorted(debug_all_categories)}")

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels, video_id = batch_data
        bs = visual_feature.shape[0]
        debug_total_samples += bs

        labels = labels.float().cuda()
        pseudo_label = pseudo_label.float().cuda()
        visual_feature = visual_feature.float().cuda()
        audio_feat = audio_feat.float().cuda()
        audio_pseudo_label = audio_pseudo_label.float().cuda()
        is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores, audio_is_event_scores, audio_event_scores = model(
            visual_feature, text_feat, audio_feat, audio_text_feat)
        is_event_scores = is_event_scores.squeeze().contiguous()
        is_event_scores = is_event_scores.reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous()
        vis_is_event_scores = vis_is_event_scores.reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous()
        audio_is_event_scores = audio_is_event_scores.reshape(bs, -1)

        labels_foreground = pseudo_label
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_BCE = labels_BCE.reshape(bs, -1)
        labels_event, _ = labels_evn.max(-1)

        audio_labels_foreground = audio_pseudo_label
        audio_labels_BCE, audio_labels_evn = audio_labels_foreground.max(-1)
        audio_labels_BCE = audio_labels_BCE.reshape(bs, -1)
        audio_labels_event, _ = audio_labels_evn.max(-1)

        loss_is_event = criterion(is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores.reshape(bs, -1), audio_labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, audio_labels_event.cuda())

        loss = loss_is_event + loss_event_class + kl_loss + vis_loss_is_event + vis_loss_event_class + audio_loss_is_event + audio_loss_event_class

        acc_result = compute_accuracy_supervised(is_event_scores, event_scores, pseudo_label)
        acc, pred, targets = acc_result

        accuracy.update(acc.item(), bs * 10)

        # ===================== 核心：替换这里 =====================
        for i in range(bs):
            # 1. 正确解析类别
            current_id = video_id[i]
            current_category = current_id.split('&')[0]  # 直接取&分割后的第一个字段，保留完整后缀

            sample_pred = pred[i].cpu().numpy()
            sample_target = targets[i].cpu().numpy()

            # 事件级统计（不动）
            total_event_in_sample = len(sample_target)
            correct_event_in_sample = np.sum(sample_pred == sample_target)
            if current_category not in class_event_stats:
                class_event_stats[current_category] = [0, 0]
            class_event_stats[current_category][0] += correct_event_in_sample
            class_event_stats[current_category][1] += total_event_in_sample
            debug_total_events += total_event_in_sample

            # ===================== 视频级正确逻辑 =====================
            # 只要视频里【有一个时刻预测正确】就算对
            correct_mask = (sample_pred == sample_target)
            is_video_correct = np.any(correct_mask)  #  这行是你真正需要的

            # 更新视频级统计
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
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
            )

    # 打印数据完整性校验信息
    logger.info(f"\n===== 数据完整性校验 =====")
    logger.info(f"1. test_dataloader实际加载样本数：{debug_total_samples}")
    logger.info(f"2. 代码统计的所有类别总事件数：{debug_total_events}")

    # 保存当前epoch的测试结果（优化输出格式，方便对比CMBS）
    output_dir = os.path.join(args.snapshot_pref, "test_results_per_epoch")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"epoch_{epoch}_test_stats.txt")

    # 整理28类的准确率（确保按固定顺序输出）
    # 先定义28类的标准名称（根据AVE数据集的类别顺序，可根据你的实际类别调整）
    standard_categories = [
        # 日志中解析到的28类（Test1.txt实际存在，按顺序排列）
        'Accordion',
        'Acoustic guitar',
        'Baby cry, infant cry',
        'Banjo',
        'Bark',
        'Bus',
        'Cat',
        'Chainsaw',
        'Church bell',
        'Clock',
        'Female speech, woman speaking',
        'Fixed-wing aircraft, airplane',
        'Flute',
        'Frying (food)',
        'Goat',
        'Helicopter',
        'Horse',
        'Male speech, man speaking',
        'Mandolin',
        'Motorcycle',
        'Race car, auto racing',
        'Rodents, rats, mice',
        'Shofar',
        'Toilet flush',
        'Train horn',
        'Truck',
        'Ukulele',
        'Violin, fiddle'
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"===== Epoch {epoch} 评估统计结果（对比CMBS）=====\n")
        f.write(f"整体事件级准确率: {accuracy.avg:.4f}%\n")
        f.write(f"总视频数: {sum([v[1] for v in class_video_stats.values()])}\n")
        f.write(
            f"整体视频级准确率: {sum([v[0] for v in class_video_stats.values()]) / sum([v[1] for v in class_video_stats.values()]) * 100:.2f}%\n")
        f.write("-" * 100 + "\n")
        # 表头：优化格式，方便复制到表格对比
        f.write("类别名\t视频总数\t正确视频数\t视频准确率(%)\t事件总数\t正确事件数\t事件准确率(%)\n")
        f.write("-" * 100 + "\n")

        # 先输出标准28类的结果
        f.write("\n【28类标准顺序（用于CMBS对比）】\n")
        total_video_correct = 0
        total_video_count = 0
        for category in standard_categories:
            if category in class_video_stats:
                # 视频级统计
                video_correct = class_video_stats[category][0]
                video_total = class_video_stats[category][1]
                video_acc = (video_correct / video_total * 100) if video_total > 0 else 0.0
                # 事件级统计
                event_correct = class_event_stats[category][0] if category in class_event_stats else 0
                event_total = class_event_stats[category][1] if category in class_event_stats else 0
                event_acc = (event_correct / event_total * 100) if event_total > 0 else 0.0

                f.write(
                    f"{category}\t{video_total}\t{video_correct}\t{video_acc:.2f}\t{event_total}\t{event_correct}\t{event_acc:.2f}\n")
                total_video_correct += video_correct
                total_video_count += video_total
            else:
                f.write(f"{category}\t0\t0\t0.00\t0\t0\t0.00\n")

        # 输出其他类别（如果有）
        other_categories = [c for c in class_video_stats.keys() if c not in standard_categories]
        if other_categories:
            f.write("\n【其他类别】\n")
            for category in sorted(other_categories):
                video_correct = class_video_stats[category][0]
                video_total = class_video_stats[category][1]
                video_acc = (video_correct / video_total * 100) if video_total > 0 else 0.0
                event_correct = class_event_stats[category][0] if category in class_event_stats else 0
                event_total = class_event_stats[category][1] if category in class_event_stats else 0
                event_acc = (event_correct / event_total * 100) if event_total > 0 else 0.0
                f.write(
                    f"{category}\t{video_total}\t{video_correct}\t{video_acc:.2f}\t{event_total}\t{event_correct}\t{event_acc:.2f}\n")

    # 打印到日志（优化格式，突出28类结果）
    logger.info("\n===== 28类视频级准确率（用于CMBS对比）=====")
    logger.info("类别名\t\t视频准确率(%)")
    logger.info("-" * 40)
    for category in standard_categories:
        if category in class_video_stats:
            video_correct = class_video_stats[category][0]
            video_total = class_video_stats[category][1]
            video_acc = (video_correct / video_total * 100) if video_total > 0 else 0.0
            # 对齐输出，方便阅读
            logger.info(f"{category:<15}\t{video_acc:.2f}")
        else:
            logger.info(f"{category:<15}\t0.00")

    # 打印整体视频级准确率
    total_video_correct = sum([v[0] for v in class_video_stats.values()])
    total_video_count = sum([v[1] for v in class_video_stats.values()])
    overall_video_acc = (total_video_correct / total_video_count * 100) if total_video_count > 0 else 0.0
    logger.info("-" * 40)
    logger.info(f"整体视频级准确率: {overall_video_acc:.2f}%")
    logger.info(f"整体事件级准确率: {accuracy.avg:.4f}%")

    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)
        # 新增：记录整体视频级准确率到Tensorboard
        writer.add_scalar('Val_epoch/Video_Accuracy', overall_video_acc, epoch)

    logger.info(
        f'**************************************************************************\t'
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
        f"\t results (kl_loss): {kl_losses.avg:.4f}%."
    )
    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    _, targets = labels.max(-1)
    # pos pred
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0
    _, event_class = event_scores.max(-1)  # foreground classification
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]
    # add mask
    pred[scores_mask] = 28  # 28 denotes bg
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())
#更改 原
    # return acc
    return (acc, pred, targets)

def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)



if __name__ == '__main__':
    main()





