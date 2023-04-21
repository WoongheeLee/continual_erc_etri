import re
import sys

import argparse
from itertools import chain
from pathlib import Path
from collections import namedtuple

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score
from transformers import (
    BertModel, BertTokenizer, BertConfig,
    Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config)


sys.path.append('utils')
sys.path.append('models')
from my_dataloader import (
    read_annotation_csv, pad_with_attention_mask, collate_fn,
    Dataset, FoldDataset)

from mywav2vec2model import MyWav2Vec2Model
from adapter_bert import BertAdapter

base_wav2vec2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
base_wav2vec2.init_weights() # without pre-training

class MultiModalClassifier(nn.Module):
    
    def __init__(self, num_labels=7, dropout_prob=0.1, num_tasks=2):
        super(MultiModalClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('klue/bert-base')
        self.bert.init_weights() # without pretraining
        
        for param in self.bert.parameters():
            param.requires_grad = False
        self.text_encoder = nn.ModuleList([BertAdapter(self.bert, adapter_size=64) for _ in range(num_tasks)])

        self.speech_encoder = MyWav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        self.speech_encoder.init_weights() # without pretraining
        

        self.speech_projector = nn.ModuleList([nn.Linear(768, 256) for _ in range(num_tasks)])
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.ModuleList([nn.Linear(768 + 256, num_labels) for _ in range(num_tasks)])


    def forward(self, speech_inputs, text_inputs, task_id):
        
        text_outputs = self.text_encoder[task_id](**text_inputs)
        text_hidden_states = text_outputs.pooler_output  # [CLS] hidden state
        # text_hidden_states.shape = (batch_size, 768)

        speech_outputs = self.speech_encoder(
            task_id=task_id,
            input_values=speech_inputs['input_values'],
            attention_mask=speech_inputs['attention_mask']
        )

        speech_hidden_states = speech_outputs.last_hidden_state
        # .extract_features : 512 dim
        # .last_hidden_state : 768 dim
        speech_hidden_states = self.speech_projector[task_id](speech_hidden_states)

        # mean_pooling (attention mask에서 1인 스텝들만 고려)
        padding_mask = base_wav2vec2._get_feature_vector_attention_mask(
            speech_hidden_states.shape[1], speech_inputs['attention_mask'], add_adapter=True)
        speech_hidden_states[~padding_mask] = 0.0
        speech_hidden_states = speech_hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        # speech_hidden_states.shape = (batch_size, 256)

        hidden_states = torch.cat([text_hidden_states, speech_hidden_states], dim=-1)
        hidden_states = self.dropout(hidden_states)
        output = self.classifier[task_id](hidden_states)
        # output.shape = (batch_size, num_labels)

        ClassifierOutput = namedtuple('ClassifierOutput', ['logits'])

        return ClassifierOutput(output)


def train_step(data_loader, model, optimizer, loss_fn, device, task_id):

    y_true, y_pred = [], []
    train_loss = 0.0

    model.train()

    for batch in tqdm(data_loader, ncols=80, desc=f'[Train step]', leave=False):
        optimizer.zero_grad()

        speech_inputs, text_inputs, labels = batch
        speech_inputs = {k: v.to(device) for k, v in speech_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        labels = labels.to(device)

        outputs = model(speech_inputs, text_inputs, task_id)
        logits = outputs.logits

        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * len(labels)
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(1).tolist())
    
    train_loss /= len(data_loader.dataset)

    outputs = {
        'report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='micro', zero_division=0) * 100.0,
        'accuracy': np.equal(y_true, y_pred).mean() * 100.0,
        'loss': train_loss
    }

    return outputs


def valid_step(data_loader, model, loss_fn, device, task_id):

    y_true, y_pred = [], []
    valid_loss = 0.0

    model.eval()

    for batch in tqdm(data_loader, ncols=80, desc=f'[Valid step]', leave=False):

        speech_inputs, text_inputs, labels = batch
        speech_inputs = {k: v.to(device) for k, v in speech_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(speech_inputs, text_inputs, task_id)
            logits = outputs.logits

        loss = loss_fn(outputs.logits, labels)
        
        valid_loss += loss.item() * len(labels)
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(1).tolist())

    valid_loss /= len(data_loader.dataset)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    outputs = {
        'report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='micro', zero_division=0) * 100.0,
        'accuracy': np.equal(y_true, y_pred).mean() * 100.0,
        'loss': valid_loss
    }

    return outputs


def parse_args(args=None):
    """
    # 주피터 노트북에서 사용 예시:
    from speech_classification import parse_args
    args = parse_args(['--data_root=../data/KEMDy19/'])
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='text_classification', help='experiment name')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--num_fold', type=int, default=0, help='k-fold 중에서 사용할 fold 숫자')
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold의 fold 개수 (k)')

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--max_text_len', type=int, default=256, help='max sequence length of speech')
    parser.add_argument('--max_seq_len', type=int, default=5, help='max sequence length of speech')

    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args(args)

    return args


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


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cpu' if args.cpu else 'cuda'

    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

    dataset19 = Dataset(data_root='../data/KEMDy19', tokenizer=tokenizer, processor=processor,
                        max_text_len=args.max_text_len, max_seq_len=args.max_seq_len)
    dataset20 = Dataset(data_root='../data/KEMDy20', tokenizer=tokenizer, processor=processor,
                        max_text_len=args.max_text_len, max_seq_len=args.max_seq_len)
    
    train19 = FoldDataset(dataset19, k=args.k_fold, fold=args.num_fold, split='train', seed=args.seed)
    valid19 = FoldDataset(dataset19, k=args.k_fold, fold=args.num_fold, split='valid', seed=args.seed)
    test19 = FoldDataset(dataset19, k=args.k_fold, fold=args.num_fold, split='test', seed=args.seed)

    train20 = FoldDataset(dataset20, k=args.k_fold, fold=args.num_fold, split='train', seed=args.seed)
    valid20 = FoldDataset(dataset20, k=args.k_fold, fold=args.num_fold, split='valid', seed=args.seed)
    test20 = FoldDataset(dataset20, k=args.k_fold, fold=args.num_fold, split='test', seed=args.seed)

    train_sampler19 = make_weighted_random_sampler(train19)
    train_sampler20 = make_weighted_random_sampler(train20)


    train19_dl = DataLoader(
        train19, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_sampler19, shuffle=False)
    valid19_dl = DataLoader(valid19, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    # test19_dl = DataLoader(test19, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    train20_dl = DataLoader(
        train20, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_sampler20, shuffle=False)
    valid20_dl = DataLoader(valid20, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    # test20_dl = DataLoader(test20, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    output_dir = Path(f'./outputs/our_adapter/{args.exp_name}/fold_{args.num_fold}')
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MultiModalClassifier()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best19_f1_score = 0.0
    best20_f1_score = 0.0
    history = {'train19': [], 'valid19': [], 'train20': [], 'valid20': []}

    for epoch in tqdm(range(args.num_epochs), ncols=80, desc='[Epochs]'):
        train19_outputs = train_step(train19_dl, model, optimizer, loss_fn, device, 0)
        valid19_outputs = valid_step(valid19_dl, model, loss_fn, device, 0)
        train20_outputs = train_step(train20_dl, model, optimizer, loss_fn, device, 1)
        valid20_outputs = valid_step(valid20_dl, model, loss_fn, device, 1)

        history['train19'].append(train19_outputs)
        history['valid19'].append(valid19_outputs)
        history['train20'].append(train20_outputs)
        history['valid20'].append(valid20_outputs)

        if valid19_outputs['f1_score'] > best19_f1_score:
            best19_f1_score = valid19_outputs['f1_score']
            torch.save(model.state_dict(), output_dir.joinpath('best19_model.pt'))

        if valid20_outputs['f1_score'] > best20_f1_score:
            best20_f1_score = valid20_outputs['f1_score']
            torch.save(model.state_dict(), output_dir.joinpath('best20_model.pt'))
        
        torch.save(history, output_dir.joinpath('history.pt'))
        
        print('\n[Epoch {} task_0] train_f1_score: {:.3f}% | valid_f1_score: {:.3f}%'.format(
            epoch + 1, train19_outputs['f1_score'], valid19_outputs['f1_score']))
        print('\n[Epoch {} task_1] train_f1_score: {:.3f}% | valid_f1_score: {:.3f}%'.format(
            epoch + 1, train20_outputs['f1_score'], valid20_outputs['f1_score']))

if __name__ == '__main__':
    print(sys.argv[:])
    args = parse_args()
    main(args)
