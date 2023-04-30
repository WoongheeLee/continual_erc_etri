import random

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score


def init_seed(seed):
    # torch reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_step(data_loader, model, device, task_id=None):

    y_true, y_pred = [], []

    model.eval()

    for batch in tqdm(data_loader, ncols=80, desc=f'[Test step]', leave=False):

        speech_inputs, text_inputs, labels = batch
        speech_inputs = {k: v.to(device) for k, v in speech_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            if task_id == None:
                outputs = model(speech_inputs, text_inputs)
            else:  # adapter case
                outputs = model(speech_inputs, text_inputs, task_id)
            logits = outputs.logits

        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    outputs = {
        'report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'accuracy': np.equal(y_true, y_pred).mean() * 100.0,
        'y_true': y_true,
        'y_pred': y_pred
    }

    return outputs
