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

        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'labels.h5')
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.sample_match_order_path = os.path.join(data_root, f'{split}_order_match.h5')
        self.annotations_path = os.path.join(data_root, f'{split}.txt')
#更改 增加
        # 新增：存储完整视频ID（用于解析类别名）
        self.video_ids = []

##
        with open(self.annotations_path, 'r') as f:
            for line in f.readlines():
                vis_line = line.strip().split("&")[1]+'.npy'
                text_line = line.strip().split("&")[1] + '_text.npy'
                self.visual_feature_path.append(vis_line)
                self.text_feature_path.append(text_line)
                self.pseudo_label_path.append(vis_line)
                self.audio_feature_path.append(vis_line)
                self.audio_pseudo_label_path.append(vis_line)
                self.audio_txt_feature_path.append(vis_line)

#更改 增加
                # 保存完整视频ID（原始行，用于提取类别名）
                self.video_ids.append(line.strip())
##

        self.h5_isOpen = False

    def __getitem__(self, index):
        if not self.h5_isOpen:
            self.labels = h5py.File(self.labels_path, 'r')['avadataset']
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.sample_match_order_path = h5py.File(self.sample_match_order_path, 'r')['order']

            self.h5_isOpen = True
        visual_feat = np.load(os.path.join(self.visual_feature_dir, self.visual_feature_path[index]))
        text_feat = np.load(os.path.join(self.visual_feature_dir, self.text_feature_path[index]))
        pseudo_label = np.load(os.path.join(self.pseudo_label_dir, self.pseudo_label_path[index]))
        audio_feat = np.load(os.path.join(self.audio_feature_dir, self.audio_feature_path[index]))
        audio_text_feat = np.load(os.path.join(self.audio_feature_dir, self.audio_txt_feature_path[index]))
        audio_pseudo_label = np.load(os.path.join(self.audio_pseudo_label_dir, self.audio_pseudo_label_path[index]))
        label = self.labels[index]
#更改 增加
        # 新增：返回视频ID（用于解析类别名）
        video_id = self.video_ids[index]
##
        #更改 原
        #return visual_feat, text_feat,  pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, label
        return visual_feat, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, label, video_id

    def __len__(self):
        return len(self.visual_feature_path)






