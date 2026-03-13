# -*- coding: utf-8 -*-
import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class AVEDatasetV2(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDatasetV2, self).__init__()
        self.split = split
        self.data_root = data_root  # 保存数据根目录，方便后续使用

        # 视觉特征和伪标签路径
        self.visual_feature_dir = os.path.join(data_root, 'CLIP_fix/features')
        self.visual_feature_path = []
        self.text_feature_path = []
        self.pseudo_label_dir = os.path.join(data_root, 'CLIP_fix/segment_pseudo_labels')
        self.pseudo_label_path = []

        # 音频特征和伪标签路径
        self.audio_feature_dir = os.path.join(data_root, 'CLAP_fix/features')
        self.audio_feature_path = []
        self.audio_txt_feature_path = []
        self.audio_pseudo_label_dir = os.path.join(data_root, 'CLAP_fix/segment_pseudo_labels')
        self.audio_pseudo_label_path = []

        # 新增：存储每个样本的视频名称
        self.video_names = []

        # 监督任务所需文件路径（核心修改：适配实际文件名）
        self.labels_path = os.path.join(data_root, 'labels.h5')
        # 关键：使用实际存在的 train_order.h5，而非 trainovave_order.h5
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.sample_match_order_path = os.path.join(data_root, f'{split}_order_match.h5')
        self.annotations_path = os.path.join(data_root, f'{split}.txt')
        # 修复硬编码路径：使用相对路径（Annotations.txt 放在 data_root 下）
        self.annotations_full_path = os.path.join(data_root, 'Annotations.txt')

        # 检查所有必要文件是否存在（新增：提前报错，方便排查）
        self._check_files_exist()

        # 加载文件顺序映射
        self.order = {}
        with open(self.annotations_full_path, 'r', encoding='utf-8') as f:
            for order_index, line in enumerate(f.readlines()):
                self.order[line.strip('\n').strip('').split('&')[1]] = order_index

        # 加载特征路径列表
        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                parts = line.split("&")
                base_name = parts[1]  # 这个就是真正的视频名（不带后缀）
                video_name = base_name  # 直接用它！

                vis_line = f'{base_name}.npy'
                text_line = f'{base_name}_text.npy'

                self.visual_feature_path.append(vis_line)
                self.text_feature_path.append(text_line)
                self.pseudo_label_path.append(vis_line)
                self.audio_feature_path.append(vis_line)
                self.audio_pseudo_label_path.append(vis_line)
                self.audio_txt_feature_path.append(vis_line)

                # 保存视频名称
                self.video_names.append(video_name)

        # 提前打开 H5 文件（修复延迟打开问题，避免多线程冲突）
        self._load_h5_files()

    def _check_files_exist(self):
        """检查必要文件是否存在，不存在则抛出明确错误"""
        required_files = [
            self.labels_path,
            self.sample_order_path,
            self.sample_match_order_path,
            self.annotations_path,
            self.annotations_full_path
        ]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"必要文件缺失：{file_path}\n请确认文件路径或生成该文件！")

    def _load_h5_files(self):
        """提前加载 H5 文件，避免在 __getitem__ 中重复打开"""
        try:
            self.labels_h5 = h5py.File(self.labels_path, 'r')
            self.labels = self.labels_h5['avadataset']

            self.sample_order_h5 = h5py.File(self.sample_order_path, 'r')
            self.sample_order = self.sample_order_h5['order']

            # 修复变量名重复问题：用 sample_match_order 存储 h5 数据，而非覆盖路径
            self.sample_match_order_h5 = h5py.File(self.sample_match_order_path, 'r')
            self.sample_match_order = self.sample_match_order_h5['order']
        except Exception as e:
            raise RuntimeError(f"加载 H5 文件失败：{e}")

    def __getitem__(self, index):
        # 加载视觉特征
        visual_feat = np.load(os.path.join(self.visual_feature_dir, self.visual_feature_path[index]))
        text_feat = np.load(os.path.join(self.visual_feature_dir, self.text_feature_path[index]))
        pseudo_label = np.load(os.path.join(self.pseudo_label_dir, self.pseudo_label_path[index]))

        # 加载音频特征
        audio_feat = np.load(os.path.join(self.audio_feature_dir, self.audio_feature_path[index]))
        audio_text_feat = np.load(os.path.join(self.audio_feature_dir, self.audio_txt_feature_path[index]))
        audio_pseudo_label = np.load(os.path.join(self.audio_pseudo_label_dir, self.audio_pseudo_label_path[index]))

        # 加载标签
        feat_name = self.visual_feature_path[index][:-4]  # 去掉 .npy 后缀
        label = self.labels[self.order[feat_name]]

        # 转换为 torch tensor（可选：如果后续需要用 torch 计算）
        visual_feat = torch.from_numpy(visual_feat).float()
        text_feat = torch.from_numpy(text_feat).float()
        pseudo_label = torch.from_numpy(pseudo_label).float()
        audio_feat = torch.from_numpy(audio_feat).float()
        audio_text_feat = torch.from_numpy(audio_text_feat).float()
        audio_pseudo_label = torch.from_numpy(audio_pseudo_label).float()
        label = torch.from_numpy(label).float()

        # 核心修改：获取当前样本的视频名称
        video_name = self.video_names[index]

        # 新增返回视频名称（作为第8个返回值）
        return visual_feat, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, label, video_name

    def __len__(self):
        return len(self.visual_feature_path)

    def __del__(self):
        """析构函数：关闭 H5 文件，避免资源泄漏"""
        if hasattr(self, 'labels_h5'):
            self.labels_h5.close()
        if hasattr(self, 'sample_order_h5'):
            self.sample_order_h5.close()
        if hasattr(self, 'sample_match_order_h5'):
            self.sample_match_order_h5.close()


# 测试代码（可选：验证数据集是否能正常加载，包含视频名称）
if __name__ == "__main__":
    # 替换为你的实际数据根目录
    data_root = r"E:\myy\CMBS-main\data"
    dataset = AVEDatasetV2(data_root, split='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 测试加载一个 batch（新增接收视频名称）
    for batch in dataloader:
        visual_feat, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, label, video_names = batch
        print("视觉特征形状：", visual_feat.shape)
        print("标签形状：", label.shape)
        print("视频名称：", video_names)  # 打印视频名称
        break