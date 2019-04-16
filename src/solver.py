import os
import torch
import copy
import math
import numpy as np
import itertools
from tensorboardX import SummaryWriter
from joblib import Parallel, delayed
from tqdm import tqdm
import torch.nn.functional as F
from asr import Seq2Seq
from dataset import load_dataset
from postprocess import calc_acc,calc_er,draw_att 

from lm import LM

VAL_STEP = 30        # Additional Inference Timesteps to run during validation (to calculate CER)
TRAIN_WER_STEP = 250 # steps for debugging info.
GRAD_CLIP = 5

class Solver():
    ''' Super class Solver for all kinds of tasks'''
    def __init__(self,config, paras):
        self.config = config
        self.paras = paras
        if torch.cuda.is_available(): 
            self.device = torch.device('cuda') 
            self.paras.gpu = True
        else:
            self.device = torch.device('cpu')
            self.paras.gpu = False

        if not os.path.exists(paras.ckpdir):
            os.makedirs(paras.ckpdir)

        self.ckpdir = os.path.join(paras.ckpdir,self.paras.name)
        if not os.path.exists(self.ckpdir):
            os.makedirs(self.ckpdir)

    def verbose(self,msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            print('[INFO]',msg)
   
    def progress(self,msg):
        ''' Verbose function for updating progress on stdout'''
        if self.paras.verbose:
            print(msg+'                              ',end='\r')


class Trainer(Solver):
    ''' Handler for complete training progress'''
    def __init__(self,config,paras):
        super(Trainer, self).__init__(config,paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir,self.paras.name)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['solver']['eval_step']
        self.best_val_ed = 10000.0

        # Training details
        self.step = 0
        self.max_step = config['solver']['total_steps']
        
    def load_data(self):
        ''' Load date for training/validation'''
        (self.mapper, _ ,self.train_set) = load_dataset(self.config['solver']['train_index_path'], 
            use_gpu=self.paras.gpu)
        (_, _, self.eval_set) = load_dataset(self.config['solver']['eval_index_path'], use_gpu=self.paras.gpu)
        
    def set_model(self):
        ''' Setup ASR'''
        self.verbose('Init ASR model. Note: validation is done through greedy decoding w/ attention decoder.')
        
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(self.device)#, reduction='none')
        
        if self.paras.load:
            self.verbose('Loading pretrained model')
            self.asr_model = torch.load(self.paras.load)
            self.best_val_ed = self.asr_model.get_best_val()
            self.verbose('The pretrained model is currently at step: {} and best loss is {:.4f}'
                .format(self.asr_model.get_global_step(), self.best_val_ed))
        
        else:
            # Build attention end-to-end ASR
            self.asr_model = Seq2Seq(self.mapper.get_dim(),self.config['asr_model']).to(self.device)

        self.starting_step = self.asr_model.get_global_step()

        # setup optimizer            
        self.asr_opt = getattr(torch.optim,self.config['asr_model']['optimizer']['type'])
        self.asr_opt = self.asr_opt(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'],eps=1e-8)

    def exec(self):
        ''' Training End-to-end ASR system'''
        self.verbose('Training set total '+str(len(self.train_set))+' batches.')

        while self.step< self.max_step:
            for x, y in self.train_set:
                self.progress('Training step: {}, global step: {}'.format(self.step, self.step+self.starting_step))

                x = x.squeeze(0).to(device = self.device,dtype=torch.float32)
                y = y.squeeze(0).to(device = self.device,dtype=torch.long)
                state_len = np.sum(np.sum(x.cpu().data.numpy(),axis=-1)!=0,axis=-1)
                state_len = [int(sl) for sl in state_len]
                ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                # ASR forwarding 
                self.asr_opt.zero_grad()
                state_len, att_pred, _ =  self.asr_model(x, ans_len, teacher=y,state_len=state_len)

                # Calculate loss function
                loss_log = {}
                label = y[:,1:ans_len+1].contiguous()
                att_loss = 0
                
                b,t,c = att_pred.shape
                att_loss = self.seq_loss(att_pred.view(b*t,c),label.view(-1))
                att_loss = torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                    .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                att_loss = torch.mean(att_loss) # Mean by batch
                loss_log['train_att'] = att_loss
                
                asr_loss = att_loss
                loss_log['train_full'] = asr_loss
                
                # Backprop
                asr_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.asr_model.parameters(), GRAD_CLIP)
                if math.isnan(grad_norm):
                    self.verbose('Error : grad norm is NaN @ step '+str(self.step))
                else:
                    self.asr_opt.step()
                
                # Logger
                self.write_log('loss',loss_log)
                self.write_log('acc',{'train':calc_acc(att_pred,label)})
                if self.step % TRAIN_WER_STEP ==0:
                    self.write_log('error rate',
                                   {'train':calc_er(att_pred,label,mapper=self.mapper)})

                # Validation
                if self.step % self.valid_step == 0 and self.step != 0:
                    self.asr_opt.zero_grad()
                    self.valid()

                self.step+=1
                if self.step > self.max_step: 
                    self.verbose('Stopping after reaching maximum training steps')
                    break
    

    def write_log(self,val_name,val_dict):
        '''Write log to TensorBoard'''
        if 'att' in val_name:
            self.log.add_image(val_name,val_dict, self.starting_step + self.step)
        elif 'txt' in val_name or 'hyp' in val_name:
            self.log.add_text(val_name, val_dict, self.starting_step + self.step)
        else:
            self.log.add_scalars(val_name,val_dict,self.starting_step + self.step)


    def valid(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding with Attention decoder only)'''
        self.asr_model.eval()
        
        # Init stats
        val_loss, val_att, val_acc, val_cer = 0.0, 0.0, 0.0, 0.0
        val_len = 0    
        all_pred,all_true = [],[]
        
        # Perform validation
        for cur_b,(x,y) in enumerate(self.eval_set):
            self.progress(' '.join(['Validation step -',str(self.step),'(',str(cur_b),'/',str(len(self.eval_set)),')']))
            
            x = x.squeeze(0).to(device = self.device,dtype=torch.float32)
            y = y.squeeze(0).to(device = self.device,dtype=torch.long)
            state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0, dim=-1)
            state_len = [int(sl) for sl in state_len]
            ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))
            
            # Forward
            state_len, att_pred, att_maps = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)

            # Compute attention loss & get decoding results
            label = y[:,1:ans_len+1].contiguous()
           
            seq_loss = self.seq_loss(att_pred[:,:ans_len,:].contiguous().view(-1,att_pred.shape[-1]),label.view(-1))
            seq_loss = torch.sum(seq_loss.view(x.shape[0],-1),dim=-1)/torch.sum(y!=0,dim=-1)\
                .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
            seq_loss = torch.mean(seq_loss) # Mean by batch
            
            val_att += seq_loss.detach()*int(x.shape[0])
            
            t1, t2 = calc_er(att_pred,label,mapper=self.mapper,get_sentence=True)
            all_pred.append(t1)
            all_true.append(t2)
            val_acc += calc_acc(att_pred,label)*int(x.shape[0])
            val_cer += calc_er(att_pred,label,mapper=self.mapper)*int(x.shape[0])
            
            val_len += int(x.shape[0])

        # Logger
        val_loss = val_att
        loss_log = {}
        for k,v in zip(['eval_full','eval_att'],[val_loss, val_att]):
            if v > 0.0: loss_log[k] = v/val_len
        self.write_log('loss',loss_log)
 
        # Plot attention map to log
        val_hyp,val_txt = calc_er(att_pred,label,mapper=self.mapper,get_sentence=True)
        val_attmap = draw_att(att_maps,att_pred)

        # Record loss
        self.write_log('error rate',{'eval':val_cer/val_len})
        self.write_log('acc',{'eval':val_acc/val_len})
        for idx,attmap in enumerate(val_attmap):
            self.write_log('att_'+str(idx),attmap)
            self.write_log('hyp_'+str(idx),val_hyp[idx])
            self.write_log('txt_'+str(idx),val_txt[idx])
 
        # Save model by val er.
        if val_cer/val_len  < self.best_val_ed:
            self.best_val_ed = val_cer/val_len
            self.verbose('Best validation error : {:.4f} @ step {} (global: {})'
                .format(self.best_val_ed,self.step, self.starting_step + self.step))
            self.asr_model.set_global_step(self.starting_step + self.step)
            self.asr_model.set_best_val(self.best_val_ed)
            torch.save(self.asr_model, os.path.join(self.ckpdir,'asr'))
            # Save hyps.
            with open(os.path.join(self.ckpdir,'best_hyp.txt'),'w') as f:
                for t1,t2 in zip(all_pred, all_true):
                    f.write(t1[0]+','+t2[0]+'\n')

        self.asr_model.train()

class RNNLM_Trainer(Solver):
    ''' Trainer for RNN-LM only'''
    def __init__(self, config, paras):
        super(RNNLM_Trainer, self).__init__(config,paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir,self.paras.name)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['solver']['eval_step']
        self.best_eval_ppx = 1000

        # training details
        self.step = 0
        self.max_step = config['solver']['total_steps']

    def load_data(self):
        ''' Load training / evaluation sets '''
        # For sanity we don't sort on the fly and instead create the indexes
        # needed and store as files. This helps alot with debugging
        (self.mapper, _, self.train_set) = load_dataset(self.config['solver']['train_index_path'], 
            batch_size=self.config['solver']['batch_size'], 
            use_gpu=self.paras.gpu, text_only=True)
        (_, _, self.eval_set) = load_dataset(self.config['solver']['eval_index_path'], 
            batch_size=self.config['solver']['eval_batch_size'],
            use_gpu=self.paras.gpu, text_only=True)

    def set_model(self):
        ''' Setup RNNLM'''
        self.verbose('Init RNNLM model.')

        if self.paras.load:
            self.verbose('Loading pretrained model')
            self.rnnlm = torch.load(self.paras.load)
            self.best_eval_ppx = self.rnnlm.get_best_ppx()
            self.verbose('The pretrained model is currently at step: {} and best ppx is {:.4f}'
                .format(self.rnnlm.get_global_step(), self.best_eval_ppx))
        
        else:
            self.rnnlm = LM(out_dim=self.mapper.get_dim(),**self.config['rnn_lm']['model_para'])

        self.rnnlm = self.rnnlm.to(self.device)
    
        self.starting_step = self.rnnlm.get_global_step()

        # optimizer
        self.rnnlm_opt = getattr(torch.optim,self.config['rnn_lm']['optimizer']['type'])
        self.rnnlm_opt = self.rnnlm_opt(self.rnnlm.parameters(), lr=self.config['rnn_lm']['optimizer']['learning_rate'],eps=1e-8)

    def exec(self):
        ''' Training RNN-LM'''
        self.verbose('RNN-LM Training set total '+str(len(self.train_set))+' batches.')

        while self.step < self.max_step:
            for y in self.train_set:
                self.progress('Training step: {}, Global step: {}'.format(self.step, self.starting_step + self.step))
                # Load data
                y = y.squeeze(0).to(device = self.device,dtype=torch.long)
                ans_len = torch.sum(y!=0,dim=-1)
                #print(ans_len)
                self.rnnlm_opt.zero_grad()
                _, prob = self.rnnlm(y[:,:-1],ans_len)
                loss = F.cross_entropy(prob.view(-1,prob.shape[-1]), y[:,1:].contiguous().view(-1), ignore_index=0)
                loss.backward()
                self.rnnlm_opt.step()

                # logger
                ppx = torch.exp(loss.cpu()).item()
                self.log.add_scalars('perplexity',{'train':ppx}, self.starting_step + self.step)

                # Next step
                self.step += 1
                if self.step % self.valid_step ==0:
                    self.valid()

                if self.step > self.max_step:
                    break

    def valid(self):
        self.rnnlm.eval()

        print_loss = 0.0
        eval_size = 0 

        for cur_b,y in enumerate(self.eval_set):
            self.progress(' '.join(['Valid step -',str(self.step),'(',str(cur_b),'/',str(len(self.eval_set)),')']))
            y = y.squeeze(0).to(device = self.device,dtype=torch.long)
            ans_len = torch.sum(y!=0,dim=-1)
            _, prob = self.rnnlm(y[:,:-1],ans_len)
            loss = F.cross_entropy(prob.view(-1,prob.shape[-1]), y[:,1:].contiguous().view(-1), ignore_index=0)
            print_loss += loss.clone().detach() * y.shape[0]
            eval_size += y.shape[0]

        print_loss /= eval_size
        eval_ppx = torch.exp(print_loss).cpu().item()
        self.log.add_scalars('perplexity',{'eval':eval_ppx}, self.starting_step + self.step)
        
        # Store model with the best perplexity
        if eval_ppx < self.best_eval_ppx:
            self.best_eval_ppx  = eval_ppx
            self.verbose('Best validation ppx : {:.4f} @ step {} (global: {})'
                .format(self.best_eval_ppx,self.step, self.starting_step + self.step))
            self.rnnlm.set_global_step(self.starting_step + self.step)
            self.rnnlm.set_best_ppx(self.best_eval_ppx)
            torch.save(self.rnnlm, os.path.join(self.ckpdir,'rnnlm'))

        self.rnnlm.train()