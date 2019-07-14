import editdistance as ed
import numpy as np
import torch

from preprocess import ALL_CHARS, EOS_TKN, SOS_TKN, TOKENS

def calc_acc(predict, label):
    '''
    Input arguments:
    * predict: A [batch_size, seq_len, char_dim] tensor, representing
    the prediction made for the label
    * label:  A [batch_size, seq_len] of mapped characters to indexes

    Returns the character-level accuracy of the prediction for
    the whole batch.
    '''
    predict = np.argmax(predict.cpu().detach(),axis=-1)
    label = label.cpu()
    accs = []
    for p,l in zip(predict,label):
        correct = 0.0
        total_char = 0
        for pp,ll in zip(p,l):
            if ll == 0: break
            correct += int(pp==ll)
            total_char += 1
        accs.append(correct/total_char)

    return sum(accs)/len(accs)

def calc_err(predict, label, mapper):
    '''
    Input arguments:
    * predict: A [batch_size, seq_len, char_dim] tensor, representing
    the prediction made for the label
    * label:  A [batch_size, seq_len] of mapped characters to indexes

    Returns the error rate in terms of edit distance for word-by-word
    comparisons between predictions and labels for each sample in the
    batch
    '''
    label = label.cpu()
    predict = np.argmax(predict.cpu().detach(), axis=-1)
    predict = [mapper.translate(p) for p in predict]
    label = [mapper.translate(l) for l in label]

    ds = [float(ed.eval(p.split(' '), l.split(' '))) / len(l.split(' '))
        for p,l in zip(predict,label)]

    return sum(ds)/len(ds)

def draw_att(att_maps, hyps):
    '''
    Input arguments:
    * att_maps (Tensor) of [batch_size, decode_steps, encode_steps] tensor
    containing attention scores for the entire batch
    * hyps (list): A list of predictions
    '''
    attmaps = []
    for i in range(att_maps.shape[0]):
        att_i = att_maps[i, :, :]
        att_len = len(trim_eos(hyps[i]))
        attmaps.append(torch.stack([att_i,att_i,att_i],dim=0)[:, :att_len, :])
    return attmaps

def trim_eos(sequence):
    new_pred = []
    for char in sequence:
        new_pred.append(int(char))
        # HACK: 1 maps to '>', generally speakingn
        if char == 1:
            break
    return new_pred