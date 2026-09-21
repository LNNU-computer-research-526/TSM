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
        self.data_root = data_root

        self.visual_feature_dir = os.path.join(data_root, 'CLIP_fix/features')
        self.visual_feature_path = []
        self.text_feature_path = []
        self.pseudo_label_dir = os.path.join(data_root, 'CLIP_fix/segment_pseudo_labels')
        self.pseudo_label_path = []

        self.audio_feature_dir = os.path.join(data_root, 'CLAP_fix/features')
        self.audio_feature_path = []
        self.audio_txt_feature_path = []
        self.audio_pseudo_label_dir = os.path.join(data_root, 'CLAP_fix/segment_pseudo_labels')
        self.audio_pseudo_label_path = []

        self.video_names = []
        self.label_indices = []

        self.labels_path = os.path.join(data_root, 'labels.h5')
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.annotations_full_path = os.path.join(data_root, 'Annotations.txt')

        self._check_files_exist()
        self._load_h5_files()
        self._build_official_split()

        print(
            f"[AVE official] split={split} | samples={len(self.video_names)} | "
            f"source={os.path.basename(self.sample_order_path)}"
        )

    def _check_files_exist(self):
        required_files = [
            self.labels_path,
            self.sample_order_path,
            self.annotations_full_path,
        ]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"必要文件缺失：{file_path}")

    def _load_h5_files(self):
        try:
            self.labels_h5 = h5py.File(self.labels_path, 'r')
            self.labels = self.labels_h5['avadataset']
            self.sample_order_h5 = h5py.File(self.sample_order_path, 'r')
            self.sample_order = np.array(self.sample_order_h5['order'][:], dtype=np.int64)
        except Exception as e:
            raise RuntimeError(f"加载 H5 文件失败：{e}")

    def _load_annotation_lines(self):
        lines = []
        with open(self.annotations_full_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def _build_official_split(self):
        ann_lines = self._load_annotation_lines()
        n_ann = len(ann_lines)
        n_labels = int(self.labels.shape[0])
        if n_ann != n_labels:
            raise RuntimeError(
                f"Annotations.txt 行数 ({n_ann}) 与 labels.h5 样本数 ({n_labels}) 不一致"
            )

        missing = []
        for sample_index in self.sample_order:
            sample_index = int(sample_index)
            if sample_index < 0 or sample_index >= n_ann:
                raise IndexError(f"official index {sample_index} 超出 Annotations 范围")
            parts = ann_lines[sample_index].split('&')
            if len(parts) < 2:
                raise ValueError(f"Annotations 第 {sample_index} 行格式错误")
            base_name = parts[1].strip()
            vis_line = f'{base_name}.npy'
            text_line = f'{base_name}_text.npy'
            vis_path = os.path.join(self.visual_feature_dir, vis_line)
            if not os.path.isfile(vis_path):
                missing.append(base_name)
                continue
            self.visual_feature_path.append(vis_line)
            self.text_feature_path.append(text_line)
            self.pseudo_label_path.append(vis_line)
            self.audio_feature_path.append(vis_line)
            self.audio_pseudo_label_path.append(vis_line)
            self.audio_txt_feature_path.append(vis_line)
            self.video_names.append(base_name)
            self.label_indices.append(sample_index)

        if missing:
            raise FileNotFoundError(
                f"{self.split} 官方划分中有 {len(missing)} 个视频缺少 CLIP 特征，"
                f"例如: {missing[:8]}"
            )

    def __getitem__(self, index):
        visual_feat = np.load(os.path.join(self.visual_feature_dir, self.visual_feature_path[index]))
        text_feat = np.load(os.path.join(self.visual_feature_dir, self.text_feature_path[index]))
        pseudo_label = np.load(os.path.join(self.pseudo_label_dir, self.pseudo_label_path[index]))
        audio_feat = np.load(os.path.join(self.audio_feature_dir, self.audio_feature_path[index]))
        audio_text_feat = np.load(os.path.join(self.audio_feature_dir, self.audio_txt_feature_path[index]))
        audio_pseudo_label = np.load(os.path.join(self.audio_pseudo_label_dir, self.audio_pseudo_label_path[index]))
        label = np.array(self.labels[self.label_indices[index]], dtype=np.float32)
        video_name = self.video_names[index]

        visual_feat = torch.from_numpy(visual_feat).float()
        text_feat = torch.from_numpy(text_feat).float()
        pseudo_label = torch.from_numpy(pseudo_label).float()
        audio_feat = torch.from_numpy(audio_feat).float()
        audio_text_feat = torch.from_numpy(audio_text_feat).float()
        audio_pseudo_label = torch.from_numpy(audio_pseudo_label).float()
        label = torch.from_numpy(label).float()
        return visual_feat, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, label, video_name

    def __len__(self):
        return len(self.visual_feature_path)

    def __del__(self):
        if hasattr(self, 'labels_h5'):
            try:
                self.labels_h5.close()
            except Exception:
                pass
        if hasattr(self, 'sample_order_h5'):
            try:
                self.sample_order_h5.close()
            except Exception:
                pass


if __name__ == "__main__":
    data_root = r"E:\myy\CMBS-main\data"
    train_set = AVEDatasetV2(data_root, split='train')
    test_set = AVEDatasetV2(data_root, split='test')
    overlap = set(train_set.video_names) & set(test_set.video_names)
    print("train", len(train_set), "test", len(test_set), "video-id overlap", len(overlap))
    dataloader = DataLoader(train_set, batch_size=4, shuffle=True)
    for batch in dataloader:
        visual_feat, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, label, video_names = batch
        print("视觉特征形状：", visual_feat.shape)
        print("标签形状：", label.shape)
        print("视频名称：", video_names)
        break
