from pathlib import Path

import re
from itertools import chain

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import WeightedRandomSampler

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def read_annotation_csv(csv_path):
    '''
    args
        csv_path: annotation 파일 하나의 경로
    return
        df: header를 소문자로 통일
    참고
        멀티 인덱싱 읽기: https://stackoverflow.com/a/65786767
    '''
    headers = pd.read_csv(csv_path, header=None, nrows=2,
                          index_col=0, keep_default_na=False).values.tolist()
    headers = [list(map(str.lower, l)) for l in headers]
    col_name = headers[0]
    new_headers = []
    for col in headers[0]:
        if len(col.strip()) != 0:
            col_name = col
        new_headers.append(col_name)
    headers[0] = new_headers
    
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    df.columns = pd.MultiIndex.from_arrays(headers)
    
    return df

def make_weighted_random_sampler(dataset, n_classes=7):
    count = [0] * n_classes
    for label in dataset:                                                         
        count[label[2]] += 1
    weight_per_class = [0.] * n_classes
    N = float(sum(count))
    for i in range(n_classes):
        weight_per_class[i] = N / float(count[i])
    weights = [0] * len(dataset)
    for idx, label in enumerate(dataset):
        weights[idx] = weight_per_class[label[2]]

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


def pad_with_attention_mask(seq):
    slen = torch.LongTensor([len(s) for s in seq])
    mask = (torch.arange(slen.max())[None, :] < slen[:, None]).type(torch.LongTensor)
    pad_seq = pad_sequence(seq, batch_first=True)

    return_dict = {'input': pad_seq, 'attention_mask': mask}

    return return_dict


def collate_fn(batch):
    batch_speech, batch_text, labels = zip(*batch)

    speech_inputs = pad_with_attention_mask(batch_speech)
    speech_inputs['input_values'] = speech_inputs.pop('input')

    text_inputs = pad_with_attention_mask(batch_text)
    text_inputs['input_ids'] = text_inputs.pop('input')
    
    labels = torch.LongTensor(labels)

    return speech_inputs, text_inputs, labels



class FoldDataset(object):
    
    def __init__(self, dataset, k=5, fold=0, seed=12, split='train'):
        '''
        args
            dataset: KEMDy19 또는 KEMDy20
            k = 폴드 개수
            fold = 몇번째폴드
            seed
            split: train, valid, test중 하나
        '''
        self.dataset = dataset
        self.k = k
        self.fold = fold
        self.seed = seed
        self.split = split
        
        self.indices = self._build_fold(k, fold, seed, split)
    
    def _build_fold(self, k, fold, seed, split):

        session_dict = {}

        for index, data in enumerate(self.dataset.data_list):
            segment_id = data['segment_id']
            session_id = re.search(r'(Sess\d{2})_', segment_id).group(1)
            
            if session_id in session_dict:
                session_dict[session_id].append(index)
            else:
                session_dict[session_id] = [index]

        num_sessions = len(session_dict)
        session_keys = np.array(sorted(session_dict.keys()))
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        train_sessions, test_sessions = list(kf.split(np.zeros(num_sessions), np.arange(num_sessions)))[fold]
        valid_sessions = np.random.RandomState(seed + fold).choice(train_sessions, 2, replace=False)
        train_sessions = np.array([i for i in train_sessions if i not in valid_sessions])
        
        if split == 'train':
            sessions = session_keys[train_sessions]
        elif split == 'valid':
            sessions = session_keys[valid_sessions]
        elif split == 'test':
            sessions = session_keys[test_sessions]
        
        indices = list(chain.from_iterable([session_dict[session_id] for session_id in sessions ]))
        return indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        index = self.indices[index]
        return self.dataset[index]


class Dataset(object):

    def __init__(self, data_root, tokenizer, processor, max_text_len=256, max_seq_len=5):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_text_len = max_text_len
        self.max_seq_len = max_seq_len

        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.data_list = self.load_dataset(self.data_root)
        
    def load_dataset(self, data_root):
        annot_csv_list = sorted(Path(data_root).joinpath('annotation').glob('*.csv'))
        wav_list = { p.name.replace('.wav', ''): str(p) for p in Path(data_root).joinpath('wav').glob('**/*.wav') }
        txt_list = { p.name.replace('.txt', ''): str(p) for p in Path(data_root).joinpath('wav').glob('**/*.txt') }
        
        data_list = []

        for csv_path in annot_csv_list:
            df = read_annotation_csv(csv_path)
            df = df[['wav', 'segment id', 'total evaluation']]
            df.columns = ['start', 'end', 'segment_id', 'emotion', 'valence', 'arousal']
            
            # 화자 평가 이슈: https://nanum.etri.re.kr/share/kjnoh/KEMDy19/shareBBSDetail?id=45&lang=ko_KR
            if 'F' in csv_path.name:
                df = df[df['segment_id'].str.contains('F')]
            elif 'M' in csv_path.name:
                df = df[df['segment_id'].str.contains('M')]

            # KEMDy20 disqust 라벨 이슈
            df['emotion'] = df['emotion'].str.replace('disqust', 'disgust')
                
            # 감정 라벨 2개 이상 제거
            df = df[~df['emotion'].str.contains(';')]
            df['emotion'] = df['emotion'].apply(self.classes.index)
            
            df['wav_path'] = df['segment_id'].apply(lambda x: wav_list.get(x, None))
            df['txt_path'] = df['segment_id'].apply(lambda x: txt_list.get(x, None))
            df = df.dropna()
            df = df.to_dict('records')
            data_list.extend(df)
            
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = self.data_list[index]
        txt_path = data['txt_path']
        emotion = data['emotion']
        
        speech, sampling_rate = torchaudio.load(data['wav_path'])  # sample_rate = 16000
        speech = speech.squeeze(0)
        duration = speech.size(0) / sampling_rate
        
        try:  # KEMDy20 KEMDy19 인코딩 차이 이슈
            with open(txt_path, 'r') as f:
                text = f.readline().strip()
        except UnicodeDecodeError:
            with open(txt_path, 'r', encoding='cp949') as f:
                text = f.readline().strip()
        
        speech_maxlen = self.max_seq_len * sampling_rate
        speech = self.processor(speech, return_tensors='pt',
                                sampling_rate=sampling_rate,
                                max_length=speech_maxlen,
                                truncation=True).input_values.squeeze(0)
        text = self.tokenizer(text, return_tensors='pt',
                              max_length=self.max_text_len,
                              truncation=True).input_ids.squeeze(0)
        
        return speech, text, emotion
