import torch
import numpy as np
import editdistance as ed

from preprocess import ALL_CHARS, TOKENS, SOS_TKN

class Mapper():
    '''Mapper for index2token'''
    def __init__(self, tokens=TOKENS+ALL_CHARS):
        self.mapping = {tokens[i]: i for i in range(len(tokens))}
        self.r_mapping = {v:k for k,v in self.mapping.items()}

    def get_dim(self):
        return len(self.mapping)

    def translate(self,seq,return_string=False):
        new_seq = []
        for c in trim_eos(seq):
            new_seq.append(self.r_mapping[c])

        if return_string:
            new_seq = ''.join(new_seq).replace('<sos>','').replace('<eos>','')
        return new_seq

def trim_eos(seqence):
    new_pred = []
    for char in seqence:
        new_pred.append(int(char))
        if char == 1:
            break
    return new_pred

class Hypothesis:
    '''Hypothesis for beam search decoding.
       Stores the history of label sequence & score 
       Stores the previous decoder state, ctc state, ctc score, lm state and attention map (if necessary)'''
    
    def __init__(self, decoder_state, emb, output_seq=[], output_scores=[], 
                 lm_state=None, att_map=None):
        assert len(output_seq) == len(output_scores)
        # attention decoder
        self.decoder_state = decoder_state
        self.att_map = att_map
        
        # RNN language model
        self.lm_state = lm_state
        
        # Previous outputs
        self.output_seq = output_seq
        self.output_scores = output_scores
                
        # Embedding layer for last_char
        self.emb = emb
        

    def avgScore(self):
        '''Return the averaged log probability of hypothesis'''
        assert len(self.output_scores) != 0
        return sum(self.output_scores) / len(self.output_scores)

    def addTopk(self, topi, topv, decoder_state, att_map=None, lm_state=None):
        '''Expand current hypothesis with a given beam size'''
        new_hypothesis = []
        term_score = None
        beam_size = len(topi[0])
        
        for i in range(beam_size):
            # Detect <eos>
            if topi[0][i].item() == 1:
                term_score = topv[0][i].cpu()
                continue
            
            idxes = self.output_seq[:]     # pass by value
            scores = self.output_scores[:] # pass by value
            idxes.append(topi[0][i].cpu())
            scores.append(topv[0][i].cpu()) 
            new_hypothesis.append(Hypothesis(decoder_state, self.emb,
                                      output_seq=idxes, output_scores=scores, lm_state=lm_state,
                                      att_map=att_map))
        if term_score is not None:
            self.output_seq.append(torch.tensor(1))
            self.output_scores.append(term_score)
            return self, new_hypothesis
        
        return None, new_hypothesis

    @property
    def outIndex(self):
        return [i.item() for i in self.output_seq]

    @property
    def last_char_idx(self):
        idx = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        return torch.LongTensor([[idx]])
    @property
    def last_char(self):
        idx = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        return self.emb(torch.LongTensor([idx]).to(next(self.emb.parameters()).device))


def cal_acc(pred,label):
    pred = np.argmax(pred.cpu().detach(),axis=-1)
    label = label.cpu()
    accs = []
    for p,l in zip(pred,label):
        correct = 0.0
        total_char = 0
        for pp,ll in zip(p,l):
            if ll == 0: break
            correct += int(pp==ll)
            total_char += 1
        accs.append(correct/total_char)
    return sum(accs)/len(accs)

def calc_er(pred,label,mapper,get_sentence=False, argmax=True):
    if argmax:
        pred = np.argmax(pred.cpu().detach(),axis=-1)
    label = label.cpu()
    pred = [mapper.translate(p,return_string=True) for p in pred]
    label = [mapper.translate(l,return_string=True) for l in label]

    if get_sentence:
        return pred,label
    
    eds = [float(ed.eval(p.split(' '),l.split(' ')))/len(l.split(' ')) for p,l in zip(pred,label)]
    
    return sum(eds)/len(eds)

def draw_att(att_list,hyp_txt):
    attmaps = []
    for att,hyp in zip(att_list[0],np.argmax(hyp_txt.cpu().detach(),axis=-1)):
        att_len = len(trim_eos(hyp))
        att = att.detach().cpu()
        attmaps.append(torch.stack([att,att,att],dim=0)[:,:att_len,:]) # +1 for att. @ <eos>
    return attmaps
