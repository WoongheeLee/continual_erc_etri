import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

from utils.data_utils import get_data_loader
from models.multimodal_classifiers import MultiModalClassifier


def train_step(data_loader, model, optimizer, loss_fn, device):

    y_true, y_pred = [], []
    train_loss = 0.0

    model.train()

    for batch in tqdm(data_loader, ncols=80, desc=f'[Train step]', leave=False):
        optimizer.zero_grad()

        speech_inputs, text_inputs, labels = batch
        speech_inputs = {k: v.to(device) for k, v in speech_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        labels = labels.to(device)

        outputs = model(speech_inputs, text_inputs)
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


def valid_step(data_loader, model, loss_fn, device):

    y_true, y_pred = [], []
    valid_loss = 0.0

    model.eval()

    for batch in tqdm(data_loader, ncols=80, desc=f'[Valid step]', leave=False):

        speech_inputs, text_inputs, labels = batch
        speech_inputs = {k: v.to(device) for k, v in speech_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(speech_inputs, text_inputs)
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
    parser.add_argument('--task', type=str, choices=['KEMDy19', 'KEMDy20'], required=True, help='train task')
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


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cpu' if args.cpu else 'cuda'

    train19_dl, valid19_dl, test19_dl = get_data_loader(data_root='./data/KEMDy19',
                                                        max_text_len=args.max_text_len,
                                                        max_seq_len=args.max_seq_len,
                                                        k_fold=args.k_fold,
                                                        fold=args.num_fold,
                                                        batch_size=args.batch_size,
                                                        seed=args.seed)
    
    train20_dl, valid20_dl, test20_dl = get_data_loader(data_root='./data/KEMDy20',
                                                        max_text_len=args.max_text_len,
                                                        max_seq_len=args.max_seq_len,
                                                        k_fold=args.k_fold,
                                                        fold=args.num_fold,
                                                        batch_size=args.batch_size,
                                                        seed=args.seed)
    
    train_dl = train19_dl if args.task == 'KEMDy19' else train20_dl
    valid_dl = valid19_dl if args.task == 'KEMDy19' else valid20_dl
    test_dl = test19_dl if args.task == 'KEMDy19' else test20_dl

    output_dir = Path(f'./outputs/task_a/fold_{args.num_fold}/{args.task}')
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MultiModalClassifier().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1_score = 0.0
    history = {'train': [], 'valid': []}

    for epoch in tqdm(range(args.num_epochs), ncols=80, desc='[Epochs]'):
        train_outputs = train_step(train_dl, model, optimizer, loss_fn, device)
        valid_outputs = valid_step(valid_dl, model, loss_fn, device)

        history['train'].append(train_outputs)
        history['valid'].append(valid_outputs)

        if valid_outputs['f1_score'] > best_f1_score:
            best_f1_score = valid_outputs['f1_score']
            torch.save(model.state_dict(), output_dir.joinpath('best_model.pt'))
        
        torch.save(model.state_dict(), output_dir.joinpath('last_epoch.pt'))
        torch.save(history, output_dir.joinpath('history.pt'))
        
        print('\n[Epoch {}] train_f1_score: {:.3f}% | valid_f1_score: {:.3f}%'.format(
            epoch + 1, train_outputs['f1_score'], valid_outputs['f1_score']))


if __name__ == '__main__':
    print(sys.argv[:])
    args = parse_args()
    main(args)
