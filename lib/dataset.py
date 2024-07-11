from torch.utils.data import Dataset
import pickle
import os
import torch
from transformers import GPT2Tokenizer
import numpy as np
from tqdm import tqdm, trange
import cv2
from PIL import Image
import lib.utils as utils
# 图像读取预处理单元
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp

class ReConstructDataset_infer(Dataset):
    def __init__(self, vocab_path, data_path, max_seq_len):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.imgid = []
        self.sentid = []
        self.filename = []
        self.image_embedding = []
        self.caption = {}
        self.caption_embedding = []

        id_finished = []
        for i in range(len(data)):
            # 是否已经存在
            if data[i]["imgid"] in id_finished:
                self.caption[data[i]["imgid"]].append(data[i]['caption'])
                continue
            else:
                self.caption[data[i]["imgid"]] = [data[i]['caption']]
                id_finished.append(data[i]["imgid"])

            self.imgid.append(data[i]["imgid"])
            self.sentid.append(data[i]["sentid"])
            self.filename.append(data[i]["filename"])
            self.image_embedding.append(data[i]["image_embedding"])
            self.caption_embedding.append(data[i]["caption_embedding"])

        self.image_embedding = torch.cat(self.image_embedding, dim=0)
        self.caption_embedding = torch.cat(self.caption_embedding, dim=0)


        self.vocab = utils.load_vocab(vocab_path)
        self.max_seq_len = max_seq_len

        # 构建图像预处理单元
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=_pil_interp('bicubic')),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        )

    

    def __getitem__(self, item: int):
        imgid = self.imgid[item]
        sentid = self.sentid[item]
        filename = self.filename[item]
        image_embedding = self.image_embedding[item]
        caption_embedding = self.caption_embedding[item]
        # 图像特征
        img = cv2.imread(filename)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        att_feats = self.transform(img)  # [3, 384, 384]，图像

        return imgid, sentid, filename, image_embedding, caption_embedding, att_feats
    
    def __len__(self) -> int:
        return len(self.imgid)

class RCStyleDataset_infer(Dataset):
    def __init__(self, vocab_path, data_path, max_seq_len, teststyle):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.imgid = []
        self.filename = []
        self.image_embedding = []
        self.caption = {}
        self.style = []

        # 当时做数据集的时候没想过imagid会有用
        if teststyle in ['romantic', 'humorous']:
            for i in range(len(data)):
                data[i]['imgid'] = data[i]['idx']


        for i in range(len(data)):
            if data[i]['style'] == teststyle:
                self.imgid.append(data[i]["imgid"])
                self.filename.append(data[i]["filename"])
                self.image_embedding.append(data[i]["prefix"])
                self.caption[data[i]["imgid"]] = data[i]['caption']

                if data[i]['style'] in ["positive", "negative"]:
                    self.style.append(data[i]['style'])
                elif data[i]['style'] in ["humorous", "romantic", "factual"]:
                    self.style.append(" "+data[i]['style'])

        self.image_embedding = torch.cat(self.image_embedding, dim=0)
        self.vocab = utils.load_vocab(vocab_path)
        self.max_seq_len = max_seq_len

        # 构建图像预处理单元
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=_pil_interp('bicubic')),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        )

    

    def __getitem__(self, item: int):
        imgid = self.imgid[item]
        filename = self.filename[item]
        image_embedding = self.image_embedding[item]
        # 图像特征
        img = cv2.imread(filename)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        att_feats = self.transform(img)  # [3, 384, 384]，图像
        style = self.style[item]

        return imgid, filename, image_embedding, att_feats, style
    
    def __len__(self) -> int:
        return len(self.imgid)