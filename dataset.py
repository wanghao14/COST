"""
Build dataset and loader for captioning task.

References:
    Copyright (c) 2017 Jie Lei
    Licensed under The MIT License, see https://choosealicense.com/licenses/mit/
    @inproceedings{lei2020mart,
        title={MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning},
        author={Lei, Jie and Wang, Liwei and Shen, Yelong and Yu, Dong and Berg, Tamara L and Bansal, Mohit},
        booktitle={ACL},
        year={2020}
    }

    Copyright (c) 2020 Simon Ging
    Licensed under Apache2 (Copyright 2021 S. Ging)
    @inproceedings{ging2020coot,
        title={COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning},
        author={Simon Ging and Mohammadreza Zolfaghari and Hamed Pirsiavash and Thomas Brox},
        booktitle={Advances on Neural Information Processing Systems (NeurIPS)},
        year={2020}
    }
"""
import copy
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from nntrainer.utils import is_main_process


class CaptionDataset(data.Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"  # denoting rare token
    DET_TOKEN = "[DET]"  # add "[DET]" token for detection feature
    ACT_TOKEN = "[ACT]"  # add "[ACT]" token for action feature
    CLS_DET_TOKEN = "[CLS_DET]"  # imitating CLS_TOKEN 
    CLS_ACT_TOKEN = "[CLS_ACT]"

    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    DET = 992
    ACT = 993
    CLS_DET = 994
    CLS_ACT = 995
    # DET = 4408
    # ACT = 4409
    # CLS_DET = 4410
    # CLS_ACT = 4411
    IGNORE = -1  # used to calculate loss

    """
    recurrent: if True, return recurrent data
    """

    def __init__(self, dset_name: str, max_t_len, max_v_len, max_n_sen, max_d_len, max_a_len, 
                 mode="train", recurrent=True, video_feature_dir: Optional[str] = None,
                 annotations_dir: str = "annotations", coot_feat_dir="data/coot_video_feature",
                 dataset_max: Optional[int] = None):
        # metadata settings
        self.dset_name = dset_name
        self.annotations_dir = Path(annotations_dir)

        # define the path of detection feats for specific category
        if self.dset_name == "youcook2":
            if mode == "train":
                self.detection_feat_dir = "data/yc2_detect_feature/training_aggre"
                with open("annotations/youcook2/extract_verbs/vidname2index_train.json", "r") as f:
                    self.action_label = json.load(f)
            else:
                self.detection_feat_dir = "data/yc2_detect_feature/validation_aggre"
                with open("annotations/youcook2/extract_verbs/vidname2index_val.json", "r") as f:
                    self.action_label = json.load(f)
        elif self.dset_name == "activitynet":
            self.detection_feat_dir = "data/anet_detect_feature/fc6_feat_100rois"
            with open("annotations/activitynet/vid_index_feat_all.json", "r") as f:
                self.vid2feat = json.load(f)
            if mode == "train":
                with open("annotations/activitynet/extract_verbs/vidname2index_train.json", "r") as f:
                    self.action_label = json.load(f)
            elif mode == 'val':
                with open("annotations/activitynet/extract_verbs/vidname2index_val1.json", "r") as f:
                    self.action_label = json.load(f)
            else:
                with open("annotations/activitynet/extract_verbs/vidname2index_test_1.json") as f:
                    self.action_label = json.load(f)
        else:
            raise ValueError(f"Unknown dataset {self.dset_name}")

        # Video feature settings
        self.video_feature_dir = Path(video_feature_dir) / self.dset_name
        self.duration_file = self.annotations_dir / self.dset_name / "captioning_video_feat_duration.csv"
        self.word2idx_file = self.annotations_dir / self.dset_name / "mart_word2idx.json"
        self.word2idx = json.load(self.word2idx_file.open("rt", encoding="utf8"))
        self.idx2word = {int(v): k for k, v in list(self.word2idx.items())}
        if is_main_process():
            print(f"WORD2IDX: {self.word2idx_file} len {len(self.word2idx)}")
        
        # Parameters for sequence lengths
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sen
        self.max_n_sen = max_n_sen
        self.max_d_len = max_d_len
        self.max_a_len = max_a_len

        # Train or val mode
        self.mode = mode

        # Recurrent or not, different data styles for different models
        self.recurrent = recurrent

        # ---------- Load metadata ----------
        # determine metadata file
        if self.dset_name == "activitynet":
            if mode == "train":  # 10000 videos
                data_path = self.annotations_dir / self.dset_name / "train.json"
            elif mode == "val":  # 2500 videos
                data_path = self.annotations_dir / self.dset_name / "captioning_val_1.json"
            elif mode == "test":  # 2500 videos
                data_path = self.annotations_dir / self.dset_name / "captioning_test_1.json"
            else:
                raise ValueError(f"Mode must be [train, val, test] for {self.dset_name}, got {mode}")
        elif self.dset_name == "youcook2":
            if mode == "train":  # 1333 videos
                data_path = self.annotations_dir / self.dset_name / "captioning_train.json"
            elif mode == "val":  # 457 videos
                data_path = self.annotations_dir / self.dset_name / "captioning_val.json"
            else:
                raise ValueError(f"Mode must be [train, val] for {self.dset_name}, got {mode}")

        # load and process captions and video data
        raw_data = json.load(data_path.open("rt", encoding="utf8"))
        coll_data = []
        for i, (k, line) in enumerate(tqdm(list(raw_data.items()))):
            if dataset_max is not None and i >= dataset_max > 0:
                break
            line["name"] = k
            line["timestamps"] = line["timestamps"][:self.max_n_sen]
            line["sentences"] = line["sentences"][:self.max_n_sen]
            coll_data.append(line)

        if self.recurrent:  # recurrent
            self.data = coll_data
        else:  # non-recurrent single sentence
            single_sentence_data = []
            for d in coll_data:
                num_sen = min(self.max_n_sen, len(d["sentences"]))
                single_sentence_data.extend([
                    {
                        "duration": d["duration"],
                        "name": d["name"],
                        "timestamp": d["timestamps"][idx],
                        "sentence": d["sentences"][idx],
                        "idx": idx,
                        "action_label": self.action_label[d["name"]][idx]
                    } for idx in range(num_sen)])
            self.data = single_sentence_data

        # check for missing detection features
        self.missing_video_names = []
        for e in self.data:
            video_name = e["name"][2:] if self.dset_name == "activitynet" else e["name"]
            cur_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
            cur_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
            for p in [cur_path_bn, cur_path_resnet]:
                if not os.path.exists(p):
                    self.missing_video_names.append(video_name)
        if is_main_process():
            print(f"Missing {len(self.missing_video_names)} features (clips/sentences) "
                  f"from {len(set(self.missing_video_names))} videos")
            print(f"Missing {set(self.missing_video_names)}")
            
        if self.dset_name == "activitynet":
            self.data = [e for e in self.data if e["name"][2:] not in self.missing_video_names]
        elif self.dset_name == "youcook2":
            self.data = [e for e in self.data if e["name"] not in self.missing_video_names]

        # ---------- Load video data ----------
        # load video duration
        # Original note: Since the features are extracted not at the exact 0.5 secs. To get the
        # real time for each feature, use `(idx + 1) * frame_to_second[vid_name] `
        frame_to_second = {}
        sampling_sec = 0.5  # hard coded, only support 0.5
        with open(self.duration_file, "r") as f:
            for line in f:
                vid_name, vid_dur, vid_frame = [entry.strip() for entry in line.split(",")]
                if self.dset_name == "activitynet":
                    frame_to_second[vid_name] = float(vid_dur) * int(float(vid_frame) * 1. / int(
                        float(vid_dur)) * sampling_sec) * 1. / float(vid_frame)
                elif self.dset_name == "youcook2":
                    frame_to_second[vid_name] = float(vid_dur) * math.ceil(float(vid_frame) * 1. / float(
                        vid_dur) * sampling_sec) * 1. / float(vid_frame)
                    if vid_name == 'T_fPNAK5Ecg':
                        print(frame_to_second[vid_name])
        if self.dset_name == "activitynet":
            frame_to_second["_0CqozZun3U"] = sampling_sec  # a missing video in anet
        self.frame_to_second = frame_to_second

        # Video features
        assert len(self.data) > 0, "No data was found! Video features directory may not be setup correctly."

        if is_main_process():
            print(f"Dataset {self.dset_name} #{len(self)} {self.mode} input video_feat")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def _load_mart_video_feature(self, raw_name: str, mot_only: bool = False) -> np.array:
        """
        Load given mart video feature

        Args:
            raw_name: Video ID
            mot_only: whether to load motion feature

        Returns:
            Mart video feature with shape (len_sequence, 3072)
        """
        video_name = raw_name[2:] if self.dset_name == "activitynet" else raw_name

        feat_app_path = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
        feat_mot_path = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
        if mot_only:
            feat_mot = np.load(feat_mot_path)
            return feat_mot
        else:
            feat_vid = np.load(feat_app_path)
            feat_mot = np.load(feat_mot_path)
            return feat_vid, feat_mot

    def _load_det_feature(self, raw_name: str) -> np.array:
        """
        Load given detection feature
        Args:
            raw_name: Video ID

        Returns:
            Detection feature with shape (len_sequence, 2048)

        """
        feat_det = np.load(os.path.join(self.detection_feat_dir, '{}.npz'.format(raw_name)))
        return feat_det

    def _load_det_feature_anet(self, raw_name: str, segment_num: int) -> np.array:
        feat_det = np.load(os.path.join(self.detection_feat_dir, 
                                        '{}_segment_{:>02d}.npy'.format(raw_name, segment_num)))
        return feat_det

    def _load_motion_cate(self, raw_name: str, idx: int) -> np.array:
        """
        Load action feature

        Args:
            raw_name: Video ID
            idx: The index of the clip

        Returns:
            Action feature with shape (3, 2048)
        """
        motion_cate = np.load(os.path.join(self.action_feat_dir, "{}_{}.npz".format(raw_name, idx)))['score']
        return motion_cate

    def convert_example_to_features(self, example):
        """
        example single snetence
        {"name": str,
         "duration": float,
         "timestamp": [st(float), ed(float)],
         "sentence": str
        } or
        {"name": str,
         "duration": float,
         "timestamps": list([st(float), ed(float)]),
         "sentences": list(str)
        }
        """
        raw_name = example["name"]

        feat_vid, feat_mot = self._load_mart_video_feature(raw_name)
        if not self.recurrent:
            idx = example["idx"]
            if self.dset_name == 'activitynet':
                feat_det = self._load_det_feature_anet(raw_name, idx)
            else:
                feat_det = self._load_det_feature(raw_name)

        if self.recurrent:
            # recurrent
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []
            for clip_idx in range(num_sen):
                if self.dset_name == 'activitynet':
                    feat_det = self._load_det_feature_anet(raw_name, clip_idx)
                else:
                    feat_det = self._load_det_feature(raw_name)
                cur_data, cur_meta = self.clip_sentence_to_feature(
                    example["name"], example["timestamps"][clip_idx], example["sentences"][clip_idx],
                    self.action_label[example["name"]][clip_idx], clip_idx, feat_vid, feat_det, feat_mot
                )
                single_video_features.append(cur_data)
                single_video_meta.append(cur_meta)
            return single_video_features, single_video_meta
        # single sentence not untied
        cur_data, cur_meta = self.clip_sentence_to_feature(
            example["name"], example["timestamp"], example["sentence"], example["action_label"], 
            example["idx"], feat_vid, feat_det, feat_mot
        )
        return cur_data, cur_meta

    def clip_sentence_to_feature(self, name, timestamp, sentence, motion_cate, clip_idx: int, 
                                 appearance_feature, detection_feature, motion_feature):
        """
        make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            clip_idx: clip number in the video (needed to loat COOT features)
            appearance_feature: Either np.array of rgb features or Dict[str, np.array] of COOT embeddings
            detection_feature: Detection features are extracted by Faster RCNN
            motion_feature: np.array of flow features
        """
        
        frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "activitynet" else self.frame_to_second[name]
        if self.dset_name == "activitynet":
            detect_info = self.vid2feat['{}_segment_{:>02d}'.format(name, clip_idx)]
        else:
            detect_info = None          # set None for youcook2 dataset

        # video + text tokens
        feat, video_tokens, video_mask, detect_feat, detect_tokens, detect_mask, detect_cates, \
            action_feat, action_tokens, action_mask,  action_cates = \
                self._load_indexed_video_feature(appearance_feature, detection_feature, timestamp, 
                                                 frm2sec, motion_cate, clip_idx, motion_feature, detect_info)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        input_tokens = video_tokens + text_tokens
        input_detect_tokens = detect_tokens + text_tokens
        input_action_tokens = action_tokens + text_tokens

        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]
        input_detect_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_detect_tokens]
        input_action_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_action_tokens]
                
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        input_labels = [self.IGNORE] * len(video_tokens) + [self.IGNORE if m == 0 else tid for tid, m in zip(
            input_ids[-len(text_mask):], text_mask)][1:] + [self.IGNORE]

        input_mask = video_mask + text_mask
        input_detect_mask = detect_mask + text_mask
        input_action_mask = action_mask + text_mask

        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len
        detect_token_type_ids = [2] * self.max_d_len + [1] * self.max_t_len       # use 2 to indicate det!!
        action_token_type_ids = [3] * self.max_a_len + [1] * self.max_t_len

        coll_data = dict(
            name=f'{name}_{clip_idx}', input_tokens=input_tokens, input_ids=np.array(input_ids).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64), input_mask=np.array(input_mask).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64), video_feature=feat.astype(np.float32),
            detect_tokens=input_detect_tokens, detect_ids=np.array(input_detect_ids).astype(np.int64),
            detect_mask=np.array(input_detect_mask).astype(np.float32),
            detect_token_type_ids=np.array(detect_token_type_ids).astype(np.int64),
            detect_feature=detect_feat.astype(np.float32), detect_cates=np.array(detect_cates).astype(np.int64),
            action_tokens=input_action_tokens, action_ids=np.array(input_action_ids).astype(np.int64),
            action_mask=np.array(input_action_mask).astype(np.float32),
            action_token_type_ids=np.array(action_token_type_ids).astype(np.int64),
            action_feature=action_feat.astype(np.float32),
            action_cates=np.array(action_cates).astype(np.float32))
        meta = dict(
            name=name, timestamp=timestamp, sentence=sentence, index=clip_idx)
        return coll_data, meta

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        """
        convert wall time st_ed to feature index st_ed
        """
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len - 1)
        st = min(st, ed - 1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(
            st, ed, feat_len)
        return st, ed

    def _get_vt_features(self, video_feat_tuple, clip_idx, max_v_l):
        vid_feat, vid_ctx_feat, clip_feats = video_feat_tuple
        clip_feat = clip_feats[clip_idx]
        if self.coot_mode == "clip":
            # only clip (1, 384)
            valid_l = 0
            feat = np.zeros((max_v_l, self.coot_dim_clip))
            feat[valid_l] = clip_feat
            valid_l += 1
        elif self.coot_mode == "vidclip":
            # stack vid + clip vertically (1, 1152)
            feat = np.zeros((max_v_l, self.coot_dim_vid + self.coot_dim_clip))
            valid_l = 0
            feat[valid_l, :self.coot_dim_vid] = vid_feat
            feat[valid_l, self.coot_dim_vid:self.coot_dim_vid + self.coot_dim_clip] = clip_feat
            valid_l += 1
        elif self.coot_mode == "vidclipctx":
            # stack vid + ctx + clip vertically (1, 1536)
            feat = np.zeros((max_v_l, self.coot_dim_vid + self.coot_dim_clip * 2))
            valid_l = 0
            feat[valid_l, :self.coot_dim_vid] = vid_feat
            feat[valid_l, self.coot_dim_vid:self.coot_dim_vid + self.coot_dim_clip] = vid_ctx_feat
            feat[valid_l, self.coot_dim_vid + self.coot_dim_clip:self.coot_dim_vid + self.coot_dim_clip * 2] = clip_feat
            valid_l += 1
        elif self.coot_mode == "vid":
            # only video (1, 768)
            feat = np.zeros((max_v_l, self.coot_dim_vid))
            valid_l = 0
            feat[valid_l, :] = vid_feat
            valid_l += 1
        else:
            raise NotImplementedError(f"Unknown: opt.vtmode = {self.coot_mode}")

        assert valid_l == max_v_l, f"valid {valid_l} max {max_v_l}"
        return feat, valid_l

    def _load_indexed_video_feature(self, appearance_feature, detection_feature, timestamp, frm2sec, motion_cate,
                                    clip_idx, motion_feature=None, detect_info=None):
        """
        [CLS], [VID], ..., [VID], [SEP], [PAD], ..., [PAD],
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat is padded to length of (self.max_v_len + self.max_t_len,)
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        
        raw_feat = appearance_feature
        if detect_info is not None:
            detection_labels = np.array(detect_info['detections'])
            detection_indexes = detect_info['indexes']
        else:
            detection_labels = []
            detection_indexes = []
            
        if len(detection_labels) > 0:
            raw_detect_feat = detection_feature[tuple(detection_indexes)].copy()       # for anet
        else:
            raw_detect_feat = detection_feature["x"][:, :5, :].copy()
            detection_labels = detection_feature["scores"][:, :5].copy()
        del detection_feature

        action_cate = np.zeros(50) # 50 for yc2 while 500 for anet
        if len(motion_cate) > 0:
            for al in motion_cate:
                action_cate[al] = 1

        # Regular video features
        max_v_l = self.max_v_len - 2
        max_d_l = self.max_d_len - 2
        max_a_l = self.max_a_len - 2
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1
        feat = np.zeros((self.max_v_len + self.max_t_len, raw_feat.shape[-1]))  # includes [CLS], [SEP]
        action_feat = np.zeros((self.max_a_len + self.max_t_len, motion_feature.shape[-1]))
        detect_feat = np.zeros((self.max_d_len + self.max_t_len,  2048))
        detect_cate = np.zeros(self.max_d_len + self.max_t_len) - 1

        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int32).tolist()
            assert max(downsamlp_indices) < feat_len
            feat[1: max_v_l + 1] = raw_feat[downsamlp_indices]
            action_feat[1: max_v_l + 1] = motion_feature[downsamlp_indices]
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            mask = [1] * (max_v_l + 2)
            action_tokens = [self.CLS_ACT_TOKEN] + [self.ACT_TOKEN] * max_a_l + [self.SEP_TOKEN]
            action_mask = [1] * (max_a_l + 2)

            # index the detection feature of yc2
            if max(downsamlp_indices) >= len(raw_detect_feat):
                st_d, ed_d = self._convert_to_feat_index_st_ed(len(raw_detect_feat), timestamp, frm2sec)
                downsamlp_indices = np.linspace(st_d, ed_d, max_d_l, endpoint=True).astype(np.int32).tolist()
            detect_feat[1: max_d_l + 1] = raw_detect_feat[downsamlp_indices].copy().reshape(-1, raw_detect_feat.shape[-1])
            detect_tokens = [self.CLS_DET_TOKEN] + [self.DET_TOKEN] * max_d_l + [self.SEP_TOKEN]
            detect_mask = [1] * (max_d_l + 2)
            detect_cate[1: max_d_l + 1] = detection_labels[downsamlp_indices].copy().reshape(-1)

        else:
            valid_l = ed - st + 1
            feat[1: valid_l + 1] = raw_feat[st: ed + 1].copy()
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + [self.SEP_TOKEN] + [self.PAD_TOKEN] * (
                    max_v_l - valid_l)
            mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)

            action_feat[1: valid_l + 1] = motion_feature[st:ed + 1].copy()
            action_tokens = [self.CLS_ACT_TOKEN] + [self.ACT_TOKEN] * valid_l + [self.SEP_TOKEN] + [self.PAD_TOKEN] * (
                max_a_l - valid_l)
            action_mask = [1] * (valid_l + 2) + [0] * (max_a_l - valid_l)
            
            # index the detection feature of yc2
            indexed_detect_feat = raw_detect_feat[st:ed+1].copy().reshape(-1, raw_detect_feat.shape[-1])
            indexed_detect_feat_len = len(indexed_detect_feat)
            detect_feat[1: indexed_detect_feat_len+1] = indexed_detect_feat
            detect_tokens = [self.CLS_DET_TOKEN] + [self.DET_TOKEN] * indexed_detect_feat_len + [self.SEP_TOKEN] + \
                [self.PAD_TOKEN] * (max_d_l - indexed_detect_feat_len)
            detect_mask = [1] * (indexed_detect_feat_len + 2) + [0] * (max_d_l - indexed_detect_feat_len)
            raw_detect_scores_pick = detection_labels[st: ed+1].copy().reshape(-1)
            detect_cate[1: indexed_detect_feat_len + 1] = raw_detect_scores_pick

        return feat, video_tokens, mask, detect_feat, detect_tokens, detect_mask, detect_cate, action_feat, action_tokens, \
               action_mask, action_cate

    def _tokenize_pad_sentence(self, sentence):
        """
        [BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD],
            len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True,
                                return_sentence_only=True) -> str:
        """
        A list of token ids
        """
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)

    def collate_fn(self, batch):
        """
        Args:
            batch:

        Returns:
        """
        if self.recurrent:
            # recurrent collate function. original docstring:
            # HOW to batch clip-sentence pair? 1) directly copy the last sentence, but do not count them in when
            # back-prop OR put all -1 to their text token label, treat

            # collect meta
            raw_batch_meta = [e[1] for e in batch]
            batch_meta = []
            for e in raw_batch_meta:
                cur_meta = dict(
                    name=None,
                    timestamp=[],
                    gt_sentence=[]
                )
                for d in e:
                    cur_meta["name"] = d["name"]
                    cur_meta["timestamp"].append(d["timestamp"])
                    cur_meta["gt_sentence"].append(d["sentence"])
                batch_meta.append(cur_meta)

            batch = [e[0] for e in batch]
            # Step1: pad each example to max_n_sen
            max_n_sen = max([len(e) for e in batch])
            raw_step_sizes = []

            padded_batch = []
            padding_clip_sen_data = copy.deepcopy(
                batch[0][0])  # doesn"t matter which one is used
            padding_clip_sen_data["input_labels"][:] = CaptionDataset.IGNORE
            for ele in batch:
                cur_n_sen = len(ele)
                if cur_n_sen < max_n_sen:
                    # noinspection PyAugmentAssignment
                    ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
                raw_step_sizes.append(cur_n_sen)
                padded_batch.append(ele)

            # Step2: batching each steps individually in the batches
            collated_step_batch = []
            for step_idx in range(max_n_sen):
                collated_step = step_collate([e[step_idx] for e in padded_batch])
                collated_step_batch.append(collated_step)
            return collated_step_batch, raw_step_sizes, batch_meta

        # single sentences / untied

        # collect meta
        batch_meta = [{
            "name": e[1]["name"],
            "timestamp": e[1]["timestamp"],
            "gt_sentence": e[1]["sentence"],
            'index': e[1]['index']
        } for e in batch]  # change key
        padded_batch = step_collate([e[0] for e in batch])
        return padded_batch, None, batch_meta

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def prepare_batch_inputs(batch, use_cuda: bool, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in list(batch.items()):
        assert bsz == len(v), (bsz, k, v)
        if use_cuda:
            if isinstance(v, torch.Tensor):
                v = v.cuda(non_blocking=non_blocking)
        batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """
    The same step (clip-sentence pair) from each example
    """
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch

def create_datasets_and_loaders(cfg):
    cfg_d = cfg.data
    # create the dataset
    train_dataset = CaptionDataset(
        cfg_d.name, cfg_d.max_t_len, cfg_d.max_v_len, cfg_d.max_n_sen, cfg_d.max_d_len, 
        cfg_d.max_a_len, mode="train", recurrent=cfg_d.recurrent, video_feature_dir=cfg_d.video_feature_dir, 
        annotations_dir=cfg_d.annotations_dir, dataset_max=cfg_d.max_datapoints
    )
    # add 10 at max_n_sen to make the inference stage use all the segments
    max_n_sen_val = cfg_d.max_n_sen + 10
    val_dataset = CaptionDataset(
        cfg_d.name, cfg_d.max_t_len, cfg_d.max_v_len, max_n_sen_val, cfg_d.max_d_len, cfg_d.max_a_len, 
        mode=cfg_d.val_split, recurrent=cfg_d.recurrent, video_feature_dir=cfg_d.video_feature_dir, 
        annotations_dir=cfg_d.annotations_dir, dataset_max=cfg_d.max_datapoints
    )

    # build loader
    if cfg.distributed:
        train_sampler= torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = SequentialDistributedSampler(val_dataset, batch_size=cfg_d.val_batch_size_per_gpu)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = data.DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn, num_workers=cfg_d.num_workers,
        batch_size=cfg_d.train_batch_size_per_gpu, sampler=train_sampler, pin_memory=cfg_d.pin_memory
    )
    val_loader = data.DataLoader(
        val_dataset, collate_fn=val_dataset.collate_fn, num_workers=cfg_d.num_workers, 
        batch_size=cfg_d.val_batch_size_per_gpu, shuffle=cfg_d.val_shuffle, sampler = val_sampler, 
        pin_memory=cfg_d.pin_memory
    )

    return train_dataset, train_loader, val_loader
