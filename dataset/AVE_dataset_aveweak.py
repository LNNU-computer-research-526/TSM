# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
from torch.utils.data import Dataset


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

        self.labels_path = os.path.join(data_root, 'labels.h5')
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.sample_match_order_path = os.path.join(data_root, f'{split}_order_match.h5')
        self.annotations_path = os.path.join(data_root, f'{split}.txt')
        self.annotations_full_path = os.path.join(data_root, 'Annotations.txt')

        self.video_ids = []

        # labels.h5 与 Annotations.txt 行号对齐
        self.order = {}
        with open(self.annotations_full_path, 'r', encoding='utf-8') as f:
            for order_index, line in enumerate(f):
                parts = line.strip().split('&')
                if len(parts) >= 2:
                    self.order[parts[1].strip()] = order_index

        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = line.split('&')[1].strip()
                vis_line = vid + '.npy'
                text_line = vid + '_text.npy'
                self.visual_feature_path.append(vis_line)
                self.text_feature_path.append(text_line)
                self.pseudo_label_path.append(vis_line)
                self.audio_feature_path.append(vis_line)
                self.audio_pseudo_label_path.append(vis_line)
                self.audio_txt_feature_path.append(vis_line)
                self.video_ids.append(line)

        with h5py.File(self.labels_path, 'r') as f:
            self.label_array = f['avadataset'][:].astype(np.float32)

        print(
            f"[AVE aveweak] split={split} | samples={len(self.video_ids)} | "
            f"train_supervise=frame-pseudo | test_eval=frame-GT"
        )

    def __getitem__(self, index):
        visual_feat = np.load(os.path.join(self.visual_feature_dir, self.visual_feature_path[index]))
        text_feat = np.load(os.path.join(self.visual_feature_dir, self.text_feature_path[index]))
        pseudo_label = np.load(os.path.join(self.pseudo_label_dir, self.pseudo_label_path[index]))
        audio_feat = np.load(os.path.join(self.audio_feature_dir, self.audio_feature_path[index]))
        audio_text_feat = np.load(os.path.join(self.audio_feature_dir, self.audio_txt_feature_path[index]))
        audio_pseudo_label = np.load(os.path.join(self.audio_pseudo_label_dir, self.audio_pseudo_label_path[index]))

        feat_name = self.visual_feature_path[index][:-4]
        if feat_name not in self.order:
            raise KeyError(f"video {feat_name} not found in Annotations.txt")
        label = self.label_array[self.order[feat_name]]  # [10, 29] GT
        video_id = self.video_ids[index]

        return (
            visual_feat,
            text_feat,
            pseudo_label,
            audio_feat,
            audio_text_feat,
            audio_pseudo_label,
            label,
            video_id,
        )

    def __len__(self):
        return len(self.visual_feature_path)
