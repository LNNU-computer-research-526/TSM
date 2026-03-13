# -*- coding: utf-8 -*-
import os
import json
import csv
import torch
from torch.utils.data import Dataset
import numpy as np
import re


class AVEDatasetV2(Dataset):
    def __init__(self, data_root, split='train'):  # 移除ratio参数
        super(AVEDatasetV2, self).__init__()
        self.split = split

        # ====================== 1. 路径设置 ======================
        self.visual_feature_dir = os.path.join(data_root, 'OVAVE/CLIP2/features')
        self.pseudo_label_dir = os.path.join(data_root, 'OVAVE/CLIP2/segment_pseudo_labels')
        self.audio_feature_dir = os.path.join(data_root, 'OVAVE/CLAP2/features')
        self.audio_pseudo_label_dir = os.path.join(data_root, 'OVAVE/CLAP2/segment_pseudo_labels')

        # 直接使用固定路径（如果有不同ratio的文件，手动修改这里）
        self.meta_csv = os.path.join(data_root, 'OVAVE/ovave_trainratio0.751_meta.csv')
        self.anno_json = os.path.join(data_root, 'OVAVE/ovavel_dataset_anno.json')

        # ====================== 2. 定义 67 类 (标准顺序) ======================
        self.categories = [
            'baby laughter', 'people whistling', 'female speech', 'people booing', 'people crowd',
            'child singing', 'female singing', 'child speech', 'people sniggering', 'people burping',
            'people cheering', 'people clapping', 'typing on typewriter', 'vacuum cleaner cleaning floors',
            'basketball bounce', 'roller coaster running', 'bowling impact', 'chicken crowing',
            'lions roaring', 'gibbon howling', 'bird chirping', 'goose honking', 'dog barking',
            'mynah bird singing', 'woodpecker pecking tree', 'frog croaking', 'cricket chirping',
            'horse clip-clop', 'crow cawing', 'cattle mooing', 'cat purring', 'turkey gobbling',
            'playing mandolin', 'playing bass guitar', 'playing banjo', 'playing drum kit',
            'playing glockenspiel', 'playing synthesizer', 'playing harp', 'playing bassoon',
            'playing cymbal', 'playing saxophone', 'playing sitar', 'playing acoustic guitar',
            'playing electric guitar', 'playing violin', 'singing bowl', 'playing bass drum',
            'playing piano', 'race car', 'train horning', 'engine accelerating', 'helicopter',
            'driving buses', 'airplane flyby', 'motorboat', 'ambulance siren', 'fire truck siren',
            'slot machine', 'electric shaver', 'chainsawing trees', 'arc welding', 'volcano explosion',
            'fireworks banging', 'ocean burbling', 'church bell ringing', 'missile launch'
        ]
        # 建立模糊匹配字典：只保留字母和数字，转小写
        self.cat_to_id = {}
        for i, cat in enumerate(self.categories):
            clean_key = re.sub(r'[^a-z0-9]', '', cat.lower())
            self.cat_to_id[clean_key] = i

        # ====================== 3. 加载元数据 ======================
        try:
            self.video_meta = self._load_meta_csv()
            print(f"成功加载 CSV 元数据，共 {len(self.video_meta)} 个视频")
        except Exception as e:
            raise Exception(f"加载 CSV 失败：{e}")

        try:
            self.video_anno = self._load_anno_json()
            print(f"成功加载 JSON 标注，共 {len(self.video_anno)} 个视频")
        except Exception as e:
            raise Exception(f"加载 JSON 失败：{e}")

        self.video_list = self._load_video_list_from_csv()

        # 预生成路径
        self.visual_feature_path = [f"{v}.npy" for v in self.video_list]
        self.text_feature_path = [f"{v}_text.npy" for v in self.video_list]
        self.video_names = self.video_list

        # ====================== 4. 自检：打印前几个视频的类别匹配情况 ======================
        print("\n[Dataset Self-Check] 正在检查类别匹配...")
        match_count = 0
        for i in range(min(5, len(self.video_list))):
            vid = self.video_list[i]
            if vid in self.video_meta:
                raw_cat = self.video_meta[vid]['category']
                idx = self._get_event_index(raw_cat)
                status = f"Matched (ID: {idx})" if idx is not None else "FAILED (Mapped to BG)"
                print(f"  Video {vid}: '{raw_cat}' -> {status}")
                if idx is not None: match_count += 1

        if match_count == 0:
            print("  警告：前5个视频均未匹配到类别！请检查 categories 列表是否正确。")
        print("--------------------------------------------------\n")

    def _load_meta_csv(self):
        meta = {}
        with open(self.meta_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['vid_name'].strip()
                meta[video_id] = {
                    'split': row['split'].strip(),
                    'category': row['cls_name'].strip(),
                }
        return meta

    def _load_anno_json(self):
        with open(self.anno_json, 'r', encoding='utf-8') as f:
            anno = json.load(f)
        if 'videos' in anno:
            new_anno = {}
            for video in anno['videos']:
                video_id = video.get('video_id', video.get('id', '')).strip()
                if video_id: new_anno[video_id] = video
            anno = new_anno
        return anno

    def _load_video_list_from_csv(self):
        split_mapping = {'trainovave': 'train', 'testovave': 'test'}
        target_split = split_mapping.get(self.split, self.split)
        video_list = [vid for vid, info in self.video_meta.items() if info['split'] == target_split]
        return video_list

    def _get_event_index(self, event_name):
        """ 模糊匹配查找索引 """
        clean_name = re.sub(r'[^a-z0-9]', '', event_name.lower())
        return self.cat_to_id.get(clean_name, None)

    def _force_shape(self, feat, target_len=10, target_dim=512):
        """
        强制调整特征形状到 (10, target_dim)
        解决 RuntimeError: expected 20 channels, got 74 channels
        """
        # 1. 处理维度 Dim
        if feat.ndim == 1:
            feat = feat[np.newaxis, :]  # (1, D)

        if feat.shape[1] > target_dim:
            feat = feat[:, :target_dim]
        elif feat.shape[1] < target_dim:
            # Padding Dim
            pad = np.zeros((feat.shape[0], target_dim - feat.shape[1]), dtype=feat.dtype)
            feat = np.concatenate([feat, pad], axis=1)

        # 2. 处理时序 Len
        curr_len = feat.shape[0]
        if curr_len == target_len:
            return feat
        elif curr_len > target_len:
            # 均匀采样 (Downsample)
            idxs = np.linspace(0, curr_len - 1, target_len).astype(int)
            return feat[idxs]
        else:
            # 填充 (Padding)
            pad_len = target_len - curr_len
            pad = np.zeros((pad_len, target_dim), dtype=feat.dtype)
            return np.concatenate([feat, pad], axis=0)

    def _get_label(self, video_name):
        label = np.zeros((10, 68), dtype=np.float32)

        # 策略1：优先使用 JSON 细粒度标注
        if video_name in self.video_anno:
            video_info = self.video_anno[video_name]
            events = video_info.get('events', video_info.get('annotations', []))

            # OVAVE 视频默认为 10s
            segment_len = 1.0

            for event in events:
                event_label = event.get('label', '').strip()
                start_t = float(event.get('start_time', event.get('start', 0.0)))
                end_t = float(event.get('end_time', event.get('end', 10.0)))

                event_idx = self._get_event_index(event_label)
                if event_idx is not None:
                    for i in range(10):
                        seg_start = i * segment_len
                        seg_end = (i + 1) * segment_len
                        intersection = max(0, min(end_t, seg_end) - max(start_t, seg_start))
                        if intersection > 0.1:
                            label[i, event_idx] = 1.0

        # 策略2：如果 JSON 没标出前景，尝试使用 CSV 的类别作为整个视频的标签
        if label[:, :67].sum() == 0 and video_name in self.video_meta:
            cat_name = self.video_meta[video_name]['category']
            idx = self._get_event_index(cat_name)
            if idx is not None:
                label[:, idx] = 1.0  # 整个视频都标为该类

        # 策略3：如果还是全0，设为背景
        for i in range(10):
            if label[i, :67].sum() == 0:
                label[i, 67] = 1.0  # Background

        return label

    def __getitem__(self, index):
        video_name = self.video_names[index]

        # 1. 视觉特征 (10, 512)
        try:
            visual_feat = np.load(os.path.join(self.visual_feature_dir, self.visual_feature_path[index]))
        except:
            visual_feat = np.zeros((10, 512), dtype=np.float32)
        visual_feat = self._force_shape(visual_feat, 10, 512)

        # 2. 文本特征 (10, 512)
        try:
            text_feat = np.load(os.path.join(self.visual_feature_dir, self.text_feature_path[index]))
            # 如果是 (1, D)，先扩展
            if text_feat.ndim == 2 and text_feat.shape[0] == 1:
                text_feat = np.repeat(text_feat, 10, axis=0)
        except:
            text_feat = np.zeros((10, 512), dtype=np.float32)
        text_feat = self._force_shape(text_feat, 10, 512)

        # 3. CLIP 伪标签 (10, 67) - 注意这里维度是 67
        try:
            pseudo_label = np.load(os.path.join(self.pseudo_label_dir, self.visual_feature_path[index]))
        except:
            pseudo_label = np.zeros((10, 67), dtype=np.float32)
        pseudo_label = self._force_shape(pseudo_label, 10, 67)

        # 4. 音频特征 (10, 128)
        try:
            audio_feat = np.load(os.path.join(self.audio_feature_dir, self.visual_feature_path[index]))
        except:
            audio_feat = np.zeros((10, 128), dtype=np.float32)
        audio_feat = self._force_shape(audio_feat, 10, 128)

        # 5. 音频文本 (10, 128)
        try:
            audio_text_feat = np.load(os.path.join(self.audio_feature_dir, self.text_feature_path[index]))
            if audio_text_feat.ndim == 2 and audio_text_feat.shape[0] == 1:
                audio_text_feat = np.repeat(audio_text_feat, 10, axis=0)
        except:
            audio_text_feat = np.zeros((10, 128), dtype=np.float32)
        audio_text_feat = self._force_shape(audio_text_feat, 10, 128)

        # 6. CLAP 伪标签 (10, 67)
        try:
            audio_pseudo_label = np.load(os.path.join(self.audio_pseudo_label_dir, self.visual_feature_path[index]))
        except:
            audio_pseudo_label = np.zeros((10, 67), dtype=np.float32)
        audio_pseudo_label = self._force_shape(audio_pseudo_label, 10, 67)

        # 7. 真实标签 (10, 68)
        label = self._get_label(video_name)

        # 类型转换
        return (visual_feat.astype(np.float32),
                text_feat.astype(np.float32),
                pseudo_label.astype(np.float32),
                audio_feat.astype(np.float32),
                audio_text_feat.astype(np.float32),
                audio_pseudo_label.astype(np.float32),
                label.astype(np.float32))

    def __len__(self):
        return len(self.video_list)