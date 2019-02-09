#encoding:utf-8
import os
import csv
import re
import random
from tqdm import tqdm
from ..utils.utils import pkl_write,text_write
class DataTransformer(object):
    def __init__(self,
                 logger,
                 label_to_id,
                 train_file,
                 valid_file,
                 valid_size,
                 vocab_path,
                 skip_header,
                 preprocess,
                 data_path,
                 shuffle,
                 alphabet,
                 seed
                 ):
        self.seed       = seed
        self.logger     = logger
        self.valid_size = valid_size
        self.train_file = train_file
        self.valid_file = valid_file
        self.vocab_path = vocab_path
        self.data_path  = data_path
        self.skip_header= skip_header
        self.label_to_id= label_to_id
        self.preprocess = preprocess
        self.shuffle    = shuffle
        self.alphabet   = alphabet
        self.build_vocab()

    # 将原始数据集分割成train和valid
    def train_val_split(self,X, y):
        self.logger.info('train val split')
        train, valid = [], []
        bucket = [[] for _ in self.label_to_id]
        for data_x, data_y in tqdm(zip(X, y), desc='bucket'):
            bucket[int(data_y)].append((data_x, data_y))
        del X, y
        for bt in tqdm(bucket, desc='split'):
            N = len(bt)
            if N == 0:
                continue
            test_size = int(N * self.valid_size)
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(bt)
            valid.extend(bt[:test_size])
            train.extend(bt[test_size:])
        # 混洗train数据集
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(train)
        return train, valid

    # char与id的映射
    def build_vocab(self):
        # 词典到编号的映射
        word2id = {k: v for k, v in zip(self.alphabet, range(0, len(self.alphabet)))}
        # 写入文件中
        pkl_write(data = word2id,filename=self.vocab_path)
        self.vocab = word2id

    # 读取原始数据集
    def read_data(self):
        targets,data = [],[]
        with open(self.data_path, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                txt = ""
                for s in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                if self.preprocess:
                    txt = self.preprocess(txt)
                if txt:
                    txt = [str(self.vocab[w]) for w in txt if w in self.vocab]
                    targets.append(int(row[0]) - 1)
                    data.append(txt)
        # 保存数据
        if self.valid_size:
            train,valid = self.train_val_split(X = data,y = targets)
            text_write(filename = self.train_file,data = train)
            text_write(filename = self.valid_file,data = valid)
        else:
            data = list(zip(data, targets))
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(data)
            if self.train_file:
                text_write(filename=self.train_file, data=data)
            elif self.valid_file:
                text_write(filename=self.valid_file, data=data)
            else:
                raise ValueError('unknow data type')


