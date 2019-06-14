import copy
import itertools
import math
import os
import json
import time
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
from librosa.display import specshow

import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# Import modules
from asr import ASR
from text_autoencoder import TextAutoEncoder
from speech_autoencoder import SpeechAutoEncoder
from discriminator import Discriminator
from charlm import CharLM

# Data, processing and tracking imports
from TrackerHandler import TrackerHandler
from LogHandler import LogHandler
from ASRDataset import load_asr_dataset, prepare_x, prepare_y
from LMDataset import LMDataset, load_lm_dataset
from postprocess import calc_acc, calc_err, draw_att
from preprocess import SOS_TKN, EOS_TKN


# Additional Inference Timesteps to run during validation 
# (to calculate CER)

class Solver:
    ''' Super class Solver for all kinds of tasks'''
    def __init__(self, config, paras, module_id):
        self.config = config
        self.paras = paras
        self.module_id = module_id

        if torch.cuda.is_available():
            self.device = torch.device('cuda') 
            self.verbose("A cuda device is available and will be used.")
            self.paras.gpu = True
        else:
            self.device = torch.device('cpu')
            self.verbose("A cuda device was not available.")
            self.paras.gpu = False

        # Create directories and files, if needed.
        if not os.path.exists(paras.ckpdir):
            os.makedirs(paras.ckpdir)
        self.ckpdir = os.path.join(self.paras.ckpdir,self.paras.name)
        if not os.path.exists(self.ckpdir):
            os.makedirs(self.ckpdir)

        # setup tracker and logger        
        self.tr = TrackerHandler(os.path.join(self.ckpdir, 'tracker.json'),
            self.module_id)
        self.lg = LogHandler(
            os.path.join(self.paras.logdir, self.paras.name, self.module_id),
            self.module_id)

        # default values for some class variables
        self.ckppath = os.path.join(self.ckpdir, self.module_id+'.cpt')
        self.best_ckppath = os.path.join(self.ckpdir, self.module_id+'_best.cpt')
        
        self.valid_step = self.set_if_exists('valid_step', 500)
        self.logging_step = self.set_if_exists('logging_step', 250)
        self.save_step = self.set_if_exists('save_step', 1000)
        self.n_epochs = self.set_if_exists('n_epochs', 5)
        self.train_batch_size = self.set_if_exists('train_batch_size', 32)
        self.valid_batch_size = self.set_if_exists('valid_batch_size', 32)
        self.test_batch_size = self.set_if_exists('test_batch_size', 1)
        self.sanity_steps = self.set_if_exists('sanity_steps', 1000)

        self.verbose_summary()

    def verbose_summary(self):
        self.verbose("-------SUMMARY-------")
        self.verbose("Current step : {}".format(self.tr.step))
        self.verbose("Best metric value : {}".format(self.tr.get_best()))
        self.verbose("Number of epochs: {}".format(self.n_epochs))
        self.verbose("Steps: [Logging {}], [Saving {}], [Validation {}]".format(
            self.logging_step, self.save_step, self.valid_step))
        self.verbose("Batch sizes: [Train {}], [Validation{}], [Testing {}]".format(
            self.train_batch_size, self.valid_batch_size, self.test_batch_size))
        self.verbose("---------------------")

    def set_if_exists(self, key, default):
        if key in self.config[self.module_id]:
            return self.config[self.module_id][key]
        else: 
            return default

    def verbose(self, msg, progress=False):
        ''' Verbose function for print information to stdout'''
        end = '\r' if progress else '\n'
        if progress:
            msg += '                              '
        else:
            msg = '[INFO ({} / {})] '.format(self.module_id, self.paras.name) + str(msg)

        if self.paras.verbose:
            print(msg, end=end)
    
    def grad_clip(self, params, optim, grad_clip=5):
        grad_norm = nn.utils.clip_grad_norm_(params, grad_clip)
        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step {}'.format(self.tr.step))
        else:
            optim.step()

    def setup_module(self, module, ckp_path, *para, **model_para):
        '''
        Given a model module type and parameters, the model will be created
        and if a pre-trained model exists, it will be loaded.

        Input arguments:
        * module (nn.Module): A module type
        * ckp_path (str): A path to where the checkpoint for the module would
        * sanity (bool): If true, we return an untrained model. 
        be stored
        * para: Any number of positional arguments passed to the module
        * model_para: Any number of keyword arguments passed to the module 
        '''
        model = module(*para, **model_para)
        if os.path.isfile(ckp_path) and not self.paras.sanity:
            self.verbose('Loading a pretrained model from {}'.format(ckp_path))
            model.load_state_dict(torch.load(ckp_path))
        else:
            if self.paras.sanity:
                self.verbose("Returning a new model ({}) for sanity checking"
                    .format(ckp_path))
            else:
                self.verbose('No model found at {}. A new model will be created'
                    .format(ckp_path))
        return model.to(self.device)

    def genpath(self, p, module_id):
        '''
        Input arguments:
        * p (None, str or tuple):
            * if None, return (self.ckpdir/<module_id>.cpt, self.ckpdir/<module_id>.cpt) 
            * if str,  return (p, p)         
            * if tuple, return the tuple
        '''
        if p is None:
            p_in = os.path.join(self.ckpdir, '{}.cpt'.format(module_id))
            p_out = p_in
            return (p_in, p_out)
        if isinstance(p, str):
            p_in = p
            p_out = p
            return (p_in, p_out)
        assert len(p) == 2
        return p    
    
    def close(self):
        # simply a parnet method to avoid some bugs
        return None
    
class CHARLMTrainer(Solver):
    def __init__(self, config, paras):
        super(CHARLMTrainer, self).__init__(config, paras, 'char_lm')

    def load_data(self):
        self.chunk_size = self.config['char_lm']['chunk_size']
        self.tf_rate = self.config['char_lm']['model_para']['tf_rate']

        self.ds, self.train_set = load_lm_dataset(self.config['char_lm']['train_index'], 
            self.chunk_size, self.train_batch_size, shuffle=True)

    def set_model(self):
        self.loss_metric = nn.CrossEntropyLoss(reduction='none').to(self.device)
        
        self.lm = self.setup_module(CharLM, self.ckppath, self.ds.get_num_chars(),            
            self.config['char_lm']['model_para']['hidden_size'])
        
        # setup optimizer
        self.optim = getattr(torch.optim,
            self.config['char_lm']['optimizer']['type'])
        self.optim = self.optim(self.lm.parameters(), 
            lr=self.config['char_lm']['optimizer']['learning_rate'], eps=1e-8)
    
    def sanity_test(self):
        self.chunk_size = 10
        self.train_batch_size = 1
        self.load_data()

        _ , ((s_x, s_y), (x, y)) = next(enumerate(self.train_set))
        
        for it in range(self.sanity_steps):
            self.lm.zero_grad()
            loss = 0
            last_char = torch.zeros((self.train_batch_size)).to(self.device) # <SOS> for whole batch
            # get inital hidden states
            (h_1, h_2) = self.lm.init_hidden(self.train_batch_size, self.device)
            # step over the whole sequence
            for i in range(self.chunk_size):
                # out.shape = [batch_size, char_dim]
                out, (h_1, h_2) = self.lm(last_char, h_1, h_2)
                label = y[:, i] # [batch size]
                loss += self.loss_metric(out, label.long())
                
                if random.random() <= self.tf_rate:
                    last_char = label.to(self.device)
                else:
                    # sample from previous prediction
                    last_char = Categorical(F.softmax(out, dim=-1)).sample() # [bs]
                    last_char = last_char.to(self.device)
            # take the mean over samples in batch
            loss = torch.mean(loss)
            loss.backward()
            self.grad_clip(self.lm.parameters(), self.optim)

            self.verbose("Loss at step ({}/{}) is : {:.4f}".format(
                it+1, self.sanity_steps,loss.item()), progress=True)
            self.lg.scalar('sanity', loss, it)

    def exec(self):
        self.verbose('Training set total {} batches.'.format(len(self.train_set)))
        epoch = 0
        while epoch < self.n_epochs:
            self.verbose("Starting epoch {} out of {}".format(epoch+1, self.n_epochs))
            for b_ind, ((s_x, s_y), (x, y)) in enumerate(self.train_set):
                # x, y shapes: [batch_size, chunk_size]
                self.verbose('Batch: {}/{}, global step: {}'.format(
                    b_ind, len(self.train_set), self.tr.step), progress=True)
                self.lm.zero_grad()
                loss = 0
                last_char = torch.zeros((self.train_batch_size)).to(self.device) # <SOS> for whole batch
                # get inital hidden states
                (h_1, h_2) = self.lm.init_hidden(self.train_batch_size, self.device)
                # step over the whole sequence
                for i in range(self.chunk_size):
                    # out.shape = [batch_size, char_dim]
                    out, (h_1, h_2) = self.lm(last_char, h_1, h_2)
                    label = y[:, i] # [batch size]
                    loss += self.loss_metric(out, label.long())
                    
                    if random.random() <= self.tf_rate:
                        last_char = label.to(self.device)
                    else:
                        # sample from previous prediction
                        last_char = Categorical(F.softmax(out, dim=-1)).sample() # [bs]
                        last_char = last_char.to(self.device)
                
                # take the mean over samples in batch
                loss = torch.mean(loss)
                loss.backward()
                self.grad_clip(self.lm.parameters(), self.optim)

                loss_by_char = loss.item() / self.chunk_size

                if self.tr.step % self.logging_step == 0:
                    self.lg.scalar('train_loss', loss_by_char, 
                        self.tr.step)

                if self.tr.step % self.valid_step == 0:
                    generated = self.generate()
                    self.lg.text('text_generate', generated, 
                            self.tr.step)

                    # save the "best" model if training loss better
                    if loss_by_char < self.tr.get_best():
                        # log the best value and save as best model
                        self.tr.set_best(loss_by_char)
                        torch.save(self.lm.state_dict(), self.best_ckppath)

                if self.tr.step % self.save_step == 0:
                    # save the model as a precaution
                    self.verbose("Model saved at step {}".format(self.tr.step))
                    torch.save(self.lm.state_dict(), self.ckppath)
                
                self.tr.do_step()

            self.verbose('Epoch {} finished'.format(epoch))
            epoch += 1

    def predict(self, x, y, tf_rate):
        '''
        Input arguments:
        * x (str): The input string
        * y (str): The output string, wherre y[i] =  x[i-1]
        * tf_force (bool): If True, we use teacher forcing with the tf_rate
        in config, otherwise we always sample from the previous prediction
        '''
        batch_size = 1
        chunk_size = len(x)
        y_text = y
        x = self.ds.s2oh(x).unsqueeze(dim=0)
        y = self.ds.s2l(y).unsqueeze(dim=0)
        last_char = torch.zeros((batch_size)).to(self.device) # <SOS> for whole batch
        # get inital hidden states
        (h_1, h_2) = self.lm.init_hidden(batch_size, self.device)
        # step over the whole sequence
        predict = []
        for i in range(chunk_size):
            # out.shape = [batch_size, char_dim]
            out, (h_1, h_2) = self.lm(last_char, h_1, h_2)
            predict.append(torch.argmax(F.softmax(out, dim=-1)))
            label = y[:, i] # [batch size]
            
            if random.random() <= tf_rate:
                last_char = label.to(self.device)
            else:
                # sample from previous prediction
                last_char = Categorical(F.softmax(out, dim=-1)).sample() # [bs]
                last_char = last_char.to(self.device)            

        predict_str = ''.join(self.ds.idx2char[p.item()] for p in predict)
        
        c = 0
        for i in range(len(predict_str)):
            if predict_str[i] == y_text[i]: c+=1
        c = 100*c/len(predict_str)
        print(predict_str+" {}".format(c))
            
    def generate(self, length=100, temp=0.8, start=SOS_TKN):
        '''
        Input arguments:
        * length (int): The number of characters to produce
        * temp (float): Low value -> more correct , high value -> more varying
        * start (str): The first characters fed as input to 
        the network. If not set, the SOS token is used.

        Use of temp: For the model probabilites, p, we define a mapping f, which
        gives new probabilities:
        
        f(p)_i = p_i ^ (1 / t) / sum[j] p_j ^ (1 / t)

        where t is the chosen temperature. If:
        * t = 1 : then we use the original probability distribution
        * t < 1 : Probability shifted towards higher values -> more selective
        * t > 1 : Probabilities become more alike -> more varying
        '''
        h_1, h_2 = self.lm.init_hidden(1, self.device)
        x = self.ds.s2l(start).to(self.device) # [seq]
        out_string = start

        for i in range(x.shape[0] - 1):
            out, (h_1, h_2) = self.lm(x[i], h_1, h_2)
        
        # The last character of 'start' is the first input
        # for the rest of predicting, shape: [1, 1, features]
        x = x[-1].view(-1)
        for i in range(length):
            out, (h_1, h_2) = self.lm(x, h_1, h_2)
            # first, create a probability distribution over characters
            # using softmax
            dist = torch.softmax(out, dim=-1)
            # then we apply the mapping using temperature, taking the pow
            # is ok here, since all values in dist are 0+.
            dist = dist**(1/temp)
            dist = dist / torch.sum(dist, dim=-1)
            # finally, we sample from this multinomial distribution
            predict = torch.multinomial(dist, 1)[0]
            predict_str = self.ds.idx2char[predict.item()]
            # add to the string
            out_string += predict_str
            # set the next input
            x = self.ds.s2l(predict_str).to(self.device)

        return out_string

    def close(self):
        '''
        Save the most recent model
        '''
        self.verbose("Finished training! The most recent model will"+\
            "be saved at step {}".format(self.tr.step))
        torch.save(self.lm.state_dict(), self.ckppath)

class ASRTrainer(Solver):
    ''' Handler for complete training progress'''
    def __init__(self, config, paras):
        super(ASRTrainer, self).__init__(config, paras , 'asr')
        
    def load_data(self):
        '''
        Load date for training/validation
        Data must be sorted by length of x
        '''
        (self.mapper, _ ,self.train_set) = load_asr_dataset(
            self.config['asr']['train_index'],
            batch_size=self.train_batch_size, use_gpu=self.paras.gpu)
        
        (_, _, self.valid_set) = load_asr_dataset(
            self.config['asr']['valid_index'], 
            batch_size=self.valid_batch_size, use_gpu=self.paras.gpu)

        self.wer_step = self.config['asr']['wer_step']
        
    def set_model(self):

        ''' Setup ASR'''        
        self.loss_metric = nn.CrossEntropyLoss(ignore_index=0, 
            reduction='none').to(self.device)
        
        self.asr_model = self.setup_module(ASR, self.ckppath, self.mapper.get_dim(),
            **self.config['asr']['model_para'])

        # setup optimizer
        self.optim = getattr(torch.optim, self.config['asr']['optimizer']['type'])
        self.optim = self.optim(self.asr_model.parameters(), 
            lr=self.config['asr']['optimizer']['learning_rate'], eps=1e-8)

    def get_asr_model(self):
        return self.asr_model

    def sanity_test(self):
        self.train_batch_size = 1
        self.load_data()

        _, (x, y) = next(enumerate(self.train_set))
        (x, x_lens) = prepare_x(x, device=self.device)
        (y, y_lens) = prepare_y(y, device=self.device)

        for it in range(self.sanity_steps):
            state_len = x_lens
            ans_len = max(y_lens) - 1

            # ASR forwarding 
            self.optim.zero_grad()
            _, prediction, _ = self.asr_model(x, ans_len, teacher=y, 
                state_len=state_len)

            # Calculate loss function
            label = y[:,1:ans_len+1].contiguous()
    
            b,t,c = prediction.shape

            # this suspicious view has been shown to work
            loss = self.loss_metric(prediction.view(b*t,c),label.view(-1))
            # Sum each uttr and devide by length
            loss = torch.sum(loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                .to(device = self.device, dtype=torch.float32)
            # Mean by batch
            loss = torch.mean(loss)     
                            
            # Backprop
            loss.backward()
            self.grad_clip(self.asr_model.parameters(), self.optim)

            self.verbose("Loss at step ({}/{}) is : {:.4f}".format(
                it+1, self.sanity_steps, loss.item()), progress=True)
            self.lg.scalar('sanity', loss, it)

    def exec(self):
        ''' Training End-to-end ASR system'''

        self.verbose('Training set total {} batches'.format(len(self.train_set)))

        epoch = 0
        while epoch < self.n_epochs:
            self.verbose("Starting epoch {} out of {}".format(epoch+1, self.n_epochs))
            for b_ind, (x, y) in enumerate(self.train_set):
                self.verbose('Batch: {}/{}, global step: {}'.format(
                    b_ind, len(self.train_set), self.tr.step), progress=True)

                (x, x_lens) = prepare_x(x, device=self.device)
                (y, y_lens) = prepare_y(y, device=self.device)
                state_len = x_lens
                ans_len = max(y_lens) - 1

                # ASR forwarding 
                self.optim.zero_grad()
                _, prediction, _ = self.asr_model(x, ans_len, teacher=y, 
                    state_len=state_len)

                # Calculate loss function
                label = y[:,1:ans_len+1].contiguous()
        
                b,t,c = prediction.shape

                # this suspicious view has been shown to work
                loss = self.loss_metric(prediction.view(b*t,c),label.view(-1))
                # Sum each uttr and devide by length
                loss = torch.sum(loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                    .to(device = self.device, dtype=torch.float32)
                # Mean by batch
                loss = torch.mean(loss)     
                                
                # Backprop
                loss.backward()
                self.grad_clip(self.asr_model.parameters(), self.optim)
                
                if self.tr.step % self.logging_step == 0:
                    self.lg.scalar('train_loss', loss, self.tr.step)
                    self.lg.scalar('train_acc', calc_acc(prediction,label), self.tr.step)

                if self.tr.step % self.wer_step == 0:
                    self.lg.scalar('train_error', 
                        calc_err(prediction, label, mapper=self.mapper), self.tr.step)

                if self.tr.step % self.save_step == 0:
                    # save the model as a precaution
                    self.verbose("Model saved at step {}".format(self.tr.step))
                    torch.save(self.asr_model.state_dict(), self.ckppath)

                if self.tr.step % self.valid_step == 0:
                    self.optim.zero_grad()
                    self.valid()

                self.tr.do_step()
            epoch += 1

    def valid(self):
        '''Perform validation step'''
        self.asr_model.eval()
        
        # Init stats
        total_loss, total_acc, total_err = 0.0, 0.0, 0.0
        num_batches = 0    
        predictions, labels = [], []
        
        # Perform validation
        for b_idx,(x,y) in enumerate(self.valid_set):
            self.verbose('Validation step - ( {} / {} )'.format(
                b_idx, len(self.valid_set)), progress=True)
            
            (x, x_lens) = prepare_x(x, device=self.device)
            (y, y_lens) = prepare_y(y, device=self.device)
            state_len = x_lens
            ans_len = max(y_lens) - 1

            # Forward, without a teacher
            state_len, prediction, att_map = self.asr_model(
                x, ans_len+30, state_len=state_len)

            # Compute attention loss & get decoding results
            label = y[:,1:ans_len+1].contiguous()
           
            loss = self.loss_metric(prediction[:,:ans_len,:]
                .contiguous().view(-1,prediction.shape[-1]), label.view(-1))

            # Sum each uttr and devide by length
            loss = torch.sum(loss.view(x.shape[0],-1),dim=-1)\
                /torch.sum(y!=0,dim=-1).to(device=self.device, dtype=torch.float32)
            # Mean by batch
            loss = torch.mean(loss)
            
            total_loss += loss.detach()
            
            mapped_prediction = [self.mapper.translate(p) for p in 
                np.argmax(prediction.cpu().detach(), axis=-1)]
            mapped_label = [self.mapper.translate(l) for l in label.cpu()]

            predictions.append(mapped_prediction)
            labels.append(mapped_label)
            
            total_acc += calc_acc(prediction,label)
            total_err += calc_err(prediction,label,mapper=self.mapper)
            
            num_batches += 1 

        # Logger
        avg_loss = total_loss / num_batches
        avg_err = total_err / num_batches
        avg_acc = total_acc / num_batches

        self.lg.scalar('eval_loss', avg_loss, self.tr.step)
        self.lg.scalar('eval_error', avg_err, self.tr.step)
        self.lg.scalar('eval_acc', avg_acc, self.tr.step)

        # Plot attention map to log for the last batch in the validation
        # dataset.
        val_hyp = [self.mapper.translate(p) for p in 
            np.argmax(prediction.cpu().detach(), axis=-1)]
        val_txt = [self.mapper.translate(l) for l in label.cpu()]
        val_attmaps = draw_att(att_map, np.argmax(prediction.cpu().detach(), axis=-1))
        
        for idx, attmap in enumerate(val_attmaps):
            #plt.imshow(attmap)
            #plt.show()
            self.lg.image('eval_att_'+str(idx), attmap, self.tr.step)
            self.lg.text('eval_hyp_'+str(idx),"{} |predict vs. real| {}".format(val_hyp[idx], val_txt[idx]), self.tr.step)
 
        # Save model by val er.
        if avg_loss  < self.tr.get_best():
            self.tr.set_best(avg_loss.item())
            self.verbose('Best validation loss for ASR : {:.4f} @ global step {}'
                .format(self.tr.get_best(), self.tr.step))
            self.verbose('Saving best model.')
            torch.save(self.asr_model.state_dict(), self.best_ckppath)
            
            # Save hyps.
            with open(os.path.join(self.ckpdir, 'best_hyp.txt'), 'w') as f:
                for t1,t2 in zip(predictions, labels):
                    f.write(t1[0]+','+t2[0]+'\n')
        else:
            self.verbose("Validation metric worse : ({:.4f} vs. {:.4f})".format(
                avg_loss, self.tr.get_best()))

        self.asr_model.train()

    def close(self):
        '''
        Save the most recent model
        '''
        self.verbose("Finished training! The most recent model will"+\
            "be saved at step {}".format(self.tr.step))
        torch.save(self.asr_model.state_dict(), self.ckppath)
    
class ASRTester(Solver):
    ''' Handler for complete inference progress'''
    def __init__(self, config, paras):
        super(ASRTester, self).__init__(config, paras, 'asr')
        
        self.decode_file = "_".join(
            ['decode','beam',str(self.config['asr']['decode_beam_size']),
            'len',str(self.config['asr']['max_decode_step_ratio'])])

    def load_data(self):
        (self.mapper, self.ds ,self.test_set) = load_asr_dataset(
            self.config['asr']['test_index'],
            batch_size=self.test_batch_size, use_gpu=self.paras.gpu)
        
    def set_model(self):
        ''' Load saved ASR'''
        self.asr_model = self.setup_module(ASR, self.ckppath, self.mapper.get_dim(),
            **self.config['asr']['model_para'])
        self.asr_model.eval()

        self.lm = CharLM(self.ds.get_char_dim(), 
            self.config['char_lm']['hidden_size']).to(self.device)        
        self.lm.eval()

        self.lm_weight = self.config['asr']['decode_lm_weight']
        self.decode_beam_size = self.config['asr']['decode_beam_size']
        self.njobs = self.config['asr']['decode_jobs']
        self.decode_step_ratio = self.config['asr']['max_decode_step_ratio']

        self.decode_file += '_lm{:}'.format(self.config['asr']['decode_lm_weight'])
    
    def exec(self, lm_weight=None):
        if lm_weight is None:
            lm_weight = self.lm_weight
        '''Perform inference step with beam search decoding.'''
        self.verbose('Start decoding with beam search, beam size: {}'
            .format(self.decode_beam_size))
        self.verbose('Number of utts to decode : {}, decoding with {} threads.'
            .format(len(self.test_set),self.njobs))
        results = []
        for b_ind, (x, y) in enumerate(self.test_set):
            x, x_len = prepare_x(x, self.device)
            y, _ = prepare_y(y, self.device)
            # TODO: we are using simple decoding, not beam decoding
            results.append(self.asr_model.decode(x, x_len, self.lm, self.mapper, lm_weight))
        return results

    def beam_decode(self, infer_lengths=False):
        '''
        Batch size is set at one by default
        
        Input arguments:
        * infer_lengths:bool (default False). If set to True, model keeps predicting
        until <EOS>
        '''
        b_idx, (x, y) = next(enumerate(self.test_set))


class TAETrainer(Solver):
    '''
    Train the Text AutoEncoder
    '''
    def __init__(self, config, paras):
        super(TAETrainer, self).__init__(config, paras, 'tae')

    def load_data(self):
        '''
        These are loaded with text only and noise, meaning dataloader
        will return (clean_y, noise_y) for both the training and validation
        sets

        Also, data must be sorted by length of y
        '''
        (self.mapper, self.dataset, self.train_set) = load_asr_dataset(
            self.config['tae']['train_index'], batch_size=self.train_batch_size, 
            use_gpu=self.paras.gpu, text_only=True, drop_rate=self.config['tae']['drop_rate'])

        (_, _, self.valid_set) = load_asr_dataset(
            self.config['tae']['valid_index'], batch_size=self.valid_batch_size, 
            use_gpu=self.paras.gpu, text_only=True, drop_rate=self.config['tae']['drop_rate'])

    def set_model(self, asrpath=None):

        (self.asrpath_in, self.asrpath_out) = self.genpath(asrpath, 'asr')

        self.asr_model = self.setup_module(ASR, self.asrpath_in, 
            self.mapper.get_dim(), **self.config['asr']['model_para'])
        self.text_autoenc = self.setup_module(TextAutoEncoder, self.ckppath,
            self.mapper.get_dim(), **self.config['tae']['model_para'])        

        '''
        The optimizer will optimize:
        * The whole textautoencoder
        * The ASR character embedding
        * The whole ASR attention module
        * The whole ASR speller module
        * The ASR char_trans linear layer
        '''

        self.optim = getattr(torch.optim, self.config['tae']['optimizer']['type'])
        self.optim = self.optim( 
                list(self.text_autoenc.parameters()) + \
                list(self.asr_model.embed.parameters()) + \
                list(self.asr_model.attention.parameters()) + \
                list(self.asr_model.decoder.parameters()) + \
                list(self.asr_model.char_trans.parameters()),
                lr=self.config['tae']['optimizer']['learning_rate'], eps=1e-8)

        self.loss_metric = torch.nn.CrossEntropyLoss(ignore_index=0, 
            reduction='none').to(self.device)

    def encode(self, x, drop_rate):
        from postprocess import simple_wer
        original_text = x
        text = ''
        for c in x:
            if random.random() > drop_rate:
                text += c
        print(text)
        text = SOS_TKN+text+EOS_TKN
        text = torch.from_numpy(self.dataset.encode(text)).unsqueeze(dim=0).to(self.device)
        text_clean = torch.from_numpy(self.dataset.encode(SOS_TKN+x+EOS_TKN)).unsqueeze(dim=0).to(self.device)
        _, enc_out = self.text_autoenc(self.asr_model, text_clean, text, 
            text_clean.shape[1], noise_lens=[text.shape[1]])
        predict = self.mapper.translate(np.argmax(enc_out.squeeze(dim=0).cpu().detach(), axis=-1))
        print(predict)
        print(simple_wer(predict, original_text))
    
    def sanity_test(self):
        self.train_batch_size = 1
        self.load_data()

        _, (y, y_noise) = next(enumerate(self.train_set))
        y, y_lens = prepare_y(y, device=self.device)
        y_max_len = max(y_lens)
        y_noise, y_noise_lens = prepare_y(y_noise, device=self.device)
        y_noise_max_lens = max(y_noise_lens)
        decode_step = y_max_len 
       
        for it in range(self.sanity_steps):
            self.optim.zero_grad()

            # decode steps == longest target
            noise_lens, enc_out = self.text_autoenc(self.asr_model, y, y_noise, 
                decode_step, noise_lens=y_noise_lens)
            
            b,t,c = enc_out.shape
            loss = self.loss_metric(enc_out.view(b*t,c), y.view(-1))
            # Sum each uttr and devide by length
            loss = torch.sum(loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                .to(device=self.device, dtype=torch.float32)
            
            # Mean by batch
            loss = torch.mean(loss)
            loss.backward()
            self.grad_clip(self.text_autoenc.parameters(), self.optim)

            self.verbose("Loss at step ({}/{}) is : {:.4f}".format(
                it+1, self.sanity_steps ,loss.item()), progress=True)
            self.lg.scalar('sanity', loss, it)

    def exec(self):
        self.verbose('Training set total {} batches'.format(len(self.train_set)))
        epoch = 0
        while epoch < self.n_epochs:
            self.verbose("Starting epoch {} out of {}".format(epoch+1, self.n_epochs))
            for b_ind, (y, y_noise) in enumerate(self.train_set):
                self.verbose('Batch: {}/{}, global step: {}'.format(
                    b_ind, len(self.train_set), self.tr.step), progress=True)

                y, y_lens = prepare_y(y, device=self.device)
                y_max_len = max(y_lens)

                y_noise, y_noise_lens = prepare_y(y_noise, device=self.device)
                y_noise_max_lens = max(y_noise_lens)
                
                self.optim.zero_grad()

                # decode steps == longest target
                decode_step = y_max_len 
                noise_lens, enc_out = self.text_autoenc(self.asr_model, y, y_noise, 
                    decode_step, noise_lens=y_noise_lens)
                
                b,t,c = enc_out.shape
                loss = self.loss_metric(enc_out.view(b*t,c), y.view(-1))
                # Sum each uttr and devide by length
                loss = torch.sum(loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                    .to(device=self.device, dtype=torch.float32)
                
                # Mean by batch
                loss = torch.mean(loss)
                loss.backward()
                self.grad_clip(self.text_autoenc.parameters(), self.optim)

                if self.tr.step % self.logging_step == 0:
                    self.lg.scalar('train_loss', loss, self.tr.step)
                
                if self.tr.step % self.valid_step == 0:
                    self.optim.zero_grad()
                    self.valid()
                
                if self.tr.step % self.save_step == 0:
                    # save the model as a precaution
                    self.verbose("Model saved at step {}".format(self.tr.step))
                    torch.save(self.text_autoenc.state_dict(), self.ckppath)
                    torch.save(self.asr_model.state_dict(), self.asrpath_out)

                self.tr.do_step()
            epoch += 1
            
    def valid(self):
        self.text_autoenc.eval()
        self.asr_model.eval()

        avg_loss = 0.0
        n_batches = 0
        for b_idx, (y, y_noise) in enumerate(self.valid_set):
            self.verbose('Validation step -( {} / {} )'.format(
                b_idx, len(self.valid_set)), progress=True)
            
            y, y_lens = prepare_y(y, device=self.device)
            y_max_len = max(y_lens)

            y_noise, y_noise_lens = prepare_y(y_noise, device=self.device)
            y_noise_max_lens = max(y_noise_lens)
            
            # decode steps == longest target
            decode_step = y_max_len 
            noise_lens, enc_out = self.text_autoenc(self.asr_model, y, y_noise, 
                decode_step, noise_lens=y_noise_lens)
            
            b,t,c = enc_out.shape
            loss = self.loss_metric(enc_out.view(b*t,c), y.view(-1))
            # Sum each uttr and devide by length
            loss = torch.sum(loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                .to(device=self.device, dtype=torch.float32)
            
            # Mean by batch
            loss = torch.mean(loss)
            
            avg_loss += loss.detach()
            n_batches += 1
        
        # compare the strings of the last batch
        labels = [self.mapper.translate(l) for l in y.cpu()]
        predicts = [self.mapper.translate(p) for p in 
            np.argmax(enc_out.cpu().detach(), axis=-1)]
        for i in range(self.valid_batch_size):
            self.lg.text('eval_text'+str(i), '{} |vs.| {}'.format(labels[i], predicts[i]), self.tr.step)

        avg_loss /=  n_batches
        avg_loss = avg_loss.item()      
        
        self.lg.scalar('eval_loss', avg_loss, self.tr.step)
        if avg_loss < self.tr.get_best():
            # then we save the model
            self.tr.set_best(avg_loss)
            self.verbose('Best validation loss : {:.4f} @ global step {}'.format(
                self.tr.get_best(), self.tr.step))
            torch.save(self.text_autoenc.state_dict(), self.best_ckppath)
            self.verbose("Both the text autoencoder and ASR have been saved")
        
        else:
            self.verbose("Validation metric worse : ({:.4f} vs. {:.4f})".format(
                avg_loss, self.tr.get_best()))

        self.text_autoenc.train()
        self.asr_model.train()

    def close(self):
        '''
        Save the most recent model
        '''
        self.verbose("Finished training! The most recent model will"+\
            "be saved at step {} as well as the ASR model".format(self.tr.step))
        torch.save(self.text_autoenc.state_dict(), self.ckppath)
        torch.save(self.asr_model.state_dict(), self.asrpath_out)

class SAETrainer(Solver):
    '''
    Train the Speech AutoEncoder.
    '''
    def __init__(self, config, paras):
        super(SAETrainer, self).__init__(config, paras, 'sae')
    
    def load_data(self):
        # data must be sorted by length of x.

        (self.mapper, _, self.train_set) = load_asr_dataset(
            self.config['sae']['train_index'], 
            batch_size=self.train_batch_size, use_gpu=self.paras.gpu)

        (_, _, self.valid_set) = load_asr_dataset(
            self.config['sae']['valid_index'], 
            batch_size=self.valid_batch_size, use_gpu=self.paras.gpu)

    def set_model(self, asrpath=None):
        (self.asrpath_in, self.asrpath_out) = self.genpath(asrpath, 'asr')

        self.asr_model = self.setup_module(ASR, self.asrpath_in, 
                self.mapper.get_dim(), **self.config['asr']['model_para'])

        self.speech_autoenc = self.setup_module(SpeechAutoEncoder, self.ckppath, 
            self.asr_model.encoder.out_dim, self.config['asr']['model_para']['feature_dim'],
            **self.config['sae']['model_para'])

        '''
        The optimizer will optimize:
        * The whole SpeechAutoencoder
        * The listener of the ASR model
        '''
        self.optim = getattr(torch.optim, self.config['sae']['optimizer']['type'])
        self.optim = self.optim(
            list(self.speech_autoenc.parameters()) + \
            list(self.asr_model.encoder.parameters()), 
            lr=self.config['sae']['optimizer']['learning_rate'], eps=1e-8)

        self.loss_metric = nn.SmoothL1Loss().to(self.device)

    def sanity_test(self):
        '''
        Each sample in the x batch is padded up to the maximum in the whole dataset.
        So x is something like [1, 32, 3000, 40]
        '''
        self.train_batch_size = 1
        self.load_data()

        _, (x, y) = next(enumerate(self.train_set)) # x shape: [32, 3000, 40]
        x, x_lens = prepare_x(x, device=self.device) # x_lens e.g. [500, 300, 600, etc.]
        #x = torch.zeros((1, max(x_lens), 40),dtype=torch.float).to(self.device)
        batch_t = max(x_lens)
        listener_out, _ = self.asr_model.encoder(x, x_lens)

        for it in range(self.sanity_steps):
            self.optim.zero_grad()

            # shape is [batch_size, 8*(~seq/8), feature_dim]
            autoenc_out = self.speech_autoenc(x, listener_out.detach())
            # pad the encoder output UP to the maximum batch time frames and 
            # pad x DOWN to the same number of frames
            enc_final = torch.zeros([autoenc_out.shape[0], batch_t, 
                autoenc_out.shape[2]]).to(self.device)
            # shapes : [bs, seq, features]
            enc_final[:, :autoenc_out.shape[1], :] = autoenc_out
            loss = self.loss_metric(enc_final, x[:, :batch_t, :])
            #loss = torch.mean(loss)
            loss.backward()
            self.grad_clip(self.speech_autoenc.parameters(), self.optim)

            self.verbose("Loss at step ({}/{}) is : {:.4f}".format(
                it+1, self.sanity_steps,loss.item()), progress=True)
            self.lg.scalar('sanity', loss, it)

            if it % 10:
                label_img = x[0,:batch_t,:].cpu()
                predict_img = autoenc_out[0, :batch_t, :].detach().cpu()
                # permute so we get correct specshow results
                label_img = label_img.permute(1, 0)
                predict_img = predict_img.permute(1, 0)

                fig = plt.figure()
                plt.subplot(2,1,1)
                specshow(label_img.numpy())
                plt.subplot(2,1,2)
                specshow(predict_img.numpy())
                
                self.lg.figure('sanity_compare', fig, it)

        # measure loss for an 8-frame window
        for it in range(self.sanity_steps):
            self.optim.zero_grad()

            autoenc_out = self.speech_autoenc(x, listener_out.detach(), just_first=True) # [bs, 8, f]
            loss = self.loss_metric(autoenc_out, x[:, :8, :])
            loss.backward()
            self.grad_clip(self.speech_autoenc.parameters(), self.optim)

            self.verbose("8-frame lossLoss at step ({}/{}) is : {:.4f}".format(
                it+1, self.sanity_steps,loss.item()), progress=True)
            self.lg.scalar('sanity_8frames', loss, it)

            predict_img = autoenc_out.squeeze(dim=0).detach().cpu()
            # permute so we get correct specshow results
            predict_img = predict_img.permute(1, 0)
            fig = plt.figure()
            plt.subplot(1,1,1)
            specshow(predict_img.numpy())
            plt.savefig('tests/img_{}'.format(it))

    def exec(self):
        self.verbose('Training set total {} batches.'.format(len(self.train_set)))
        epoch = 0
        while epoch < self.n_epochs:
            self.verbose("Starting epoch {} out of {}".format(epoch+1, self.n_epochs))
            for b_ind, (x, y) in enumerate(self.train_set):
                self.verbose('Batch: {}/{}, global step: {}'.format(
                    b_ind, len(self.train_set), self.tr.step), progress=True)

                self.optim.zero_grad()

                x, x_lens = prepare_x(x, device=self.device)
                listener_out, _  = self.asr_model.encoder(x, x_lens)
                # shape is [batch_size, 8*(~seq/8), feature_dim]
                autoenc_out = self.speech_autoenc(x, listener_out)
                # pad the encoder output UP to the maximum batch time frames and 
                # pad x DOWN to the same number of frames
                batch_t = max(x_lens)
                x = x[:, :batch_t, :]
                enc_final = torch.zeros([autoenc_out.shape[0], batch_t, 
                    autoenc_out.shape[2]]).to(self.device)
                enc_final[:, :autoenc_out.shape[1], :] = autoenc_out
                loss = self.loss_metric(enc_final, x)
                # ATTN: not doing this anymore, just use the default
                # Divide each by length of x and sum, then take the average over the
                # batch
                loss.backward()
                self.grad_clip(self.speech_autoenc.parameters(), self.optim)

                if self.tr.step % self.logging_step == 0:
                    self.lg.scalar('train_loss', loss, self.tr.step)

                if self.tr.step % self.valid_step == 0:
                    self.optim.zero_grad()
                    self.valid()

                if self.tr.step % self.save_step == 0:
                    # save the model as a precaution
                    self.verbose("Model saved at step {}".format(self.tr.step))
                    torch.save(self.speech_autoenc.state_dict(), self.ckppath)
                    torch.save(self.asr_model.state_dict(), self.asrpath_out)

                self.tr.do_step()
            epoch += 1
                
    def valid(self):
        self.speech_autoenc.eval()
        self.asr_model.eval()

        avg_loss = 0.0
        n_batches = 0
        for b_idx, (x, y) in enumerate(self.valid_set):
            self.verbose('Validation step - {} ( {} / {} )'.format(
                self.tr.step, b_idx, len(self.valid_set)), progress=True)
            
            x, x_lens = prepare_x(x, device=self.device)
            listener_out, _ = self.asr_model.encoder(x, x_lens)
            enc_out = self.speech_autoenc(x, listener_out)
            # pad the encoder output UP to the maximum batch time frames and 
            # pad x DOWN to the same number of frames
            batch_t = max(x_lens)
            x = x[:, :batch_t, :]
            enc_final = torch.zeros([enc_out.shape[0], batch_t, enc_out.shape[2]]).to(self.device)
            enc_final[:, :enc_out.shape[1], :] = enc_out

            loss = self.loss_metric(enc_final, x)

            avg_loss += loss.detach()
            n_batches += 1

        # draw comparisons for each sample in the last batch
        for i in range(self.valid_batch_size):
            x_len = x_lens[i]
            label_img = x[i,:x_len,:].cpu()
            predict_img = enc_final[i, :x_len, :].detach().cpu()
            # permute so we get correct specshow results
            label_img = label_img.permute(1, 0)
            predict_img = predict_img.permute(1, 0)

            fig = plt.figure()
            plt.subplot(2,1,1)
            specshow(label_img.numpy())
            plt.subplot(2,1,2)
            specshow(predict_img.numpy())
            
            self.lg.figure('encode_compare_'+str(i), fig, self.tr.step)

        avg_loss /=  n_batches
        avg_loss = avg_loss.item()

        self.lg.scalar('eval_loss', avg_loss, self.tr.step)
      
        if avg_loss < self.tr.get_best():
            # then we save the model
            self.tr.set_best(avg_loss)
            self.verbose('Best validation loss : {:.4f} @ global step {}'
                .format(self.tr.get_best(), self.tr.step))
            torch.save(self.speech_autoenc.state_dict(), self.best_ckppath)
        else:
            self.verbose("Validation metric worse : ({:.4f} vs. {:.4f})".format(
                avg_loss, self.tr.get_best()))

        self.speech_autoenc.train()
        self.asr_model.train()
    
    def close(self):
        '''
        Save the most recent model
        '''
        self.verbose("Finished training! The most recent model will"+\
            "be saved at step {} as well as the ASR model".format(self.tr.step))
        torch.save(self.speech_autoenc.state_dict(), self.ckppath)
        torch.save(self.asr_model.state_dict(), self.asrpath_out)

class ADVTrainer(Solver):
    '''
    Do adversarial training on both the Discriminator and
    the Generator (Listener) using the Text encoder as a 
    data source
    '''
    def __init__(self, config, paras):
        super(ADVTrainer, self).__init__(config, paras, 'adv')

    def load_data(self):
        # data is sorted by length of x
        (self.mapper, self.dataset, self.train_set) = load_asr_dataset(
            self.config['adv']['train_index'], batch_size=self.train_batch_size, 
            use_gpu=self.paras.gpu)

        (_, _, self.valid_set) = load_asr_dataset(
            self.config['adv']['eval_index'], batch_size=self.valid_batch_size,
            use_gpu=self.paras.gpu)

        self.chunk_size = self.config['adv']['chunk_size']
        self.lm_ds, self.lm_train_set = load_lm_dataset(self.config['adv']['lm_train_index'], 
            self.chunk_size, self.train_batch_size, shuffle=True, label_format=True)
        
    def set_model(self, asrpath=None, taepath=None):
        (self.asrpath_in, self.asrpath_out) = self.genpath(asrpath, 'asr')
        (taepath_in, _) = self.genpath(taepath, 'tae')

        self.asr_model = self.setup_module(ASR, self.asrpath_in, 
            self.mapper.get_dim(), **self.config['asr']['model_para'])
       
        self.text_autoenc = self.setup_module(TextAutoEncoder, taepath_in, 
            self.mapper.get_dim(), **self.config['tae']['model_para'])        

        self.discriminator = self.setup_module(Discriminator, self.ckppath,
            self.asr_model.encoder.get_outdim(), 
            **self.config['adv']['model_para'])
        
        self.data_distribution = self.text_autoenc.encoder
        
        self.G_optim = getattr(torch.optim, 
            self.config['adv']['G_optimizer']['type'])
        self.G_optim = self.G_optim(self.asr_model.encoder.parameters(), 
            lr=self.config['adv']['G_optimizer']['learning_rate'], eps=1e-8)

        self.D_optim = getattr(torch.optim, 
            self.config['adv']['D_optimizer']['type'])
        self.D_optim = self.D_optim(self.discriminator.parameters(), 
            lr=self.config['adv']['D_optimizer']['learning_rate'], eps=1e-8)

        self.loss_metric = torch.nn.BCELoss().to(self.device)
    
    def sanity_test(self):
        '''
        Adversarial training:
        * (G)enerator: Listener (from LAS)
        * (D)iscriminator: This discriminator module
        * Data distribution: The text encoder
        '''
        self.train_batch_size = 1
        self.load_data()

        _ , (x, y) = next(enumerate(self.train_set))
        x, x_lens = prepare_x(x, device=self.device)    
        y, y_lens = prepare_y(y, device=self.device)    
        batch_size = x.shape[0]

        _ , (_, (lm_x, _)) = next(enumerate(self.lm_train_set))
        lm_x = lm_x.long().to(self.device)

        train_discriminator = False
        train_generator = True

        for it in range(self.sanity_steps):

            '''
            DISCRIMINATOR TRAINING
            maximize log(D(x)) + log(1 - D(G(z)))

            The discriminator should be trained to differentiate between _vectors_
            generated by the generator and the distribution (i.e. each _sample_ 
            is a 512D vector)
            '''
            self.discriminator.zero_grad()

            '''Discriminator real data training'''
            real_data = self.data_distribution(lm_x) # [bs, text_seq, 512]
            #print("Real data shape: ", real_data.shape)
            real_data = real_data.view(batch_size*real_data.shape[1], -1) # [bs*seq, 512]
            real_emb = real_data
            D_out = self.discriminator(real_data) # [bs*text_seq, 1]
            real_labels = torch.ones(D_out.shape[0]).to(self.device)\
                - self.config['adv']['label_smoothing'] # [bs, text_seq]
            D_realloss = self.loss_metric(D_out.squeeze(dim=1), real_labels) # e.g[200,1]->[200]
            D_realloss.backward()

            '''Discriminator fake data training'''
            fake_data, _ = self.asr_model.encoder(x, x_lens)
            #print("Fake data shape :", fake_data.shape)
            fake_data = fake_data.view(batch_size*fake_data.shape[1], -1) # [bs*seq, 512]
            fake_emb = fake_data
            # Note: fake_data.detach() is used here to avoid backpropping
            # through the generator. (see grad_pointers.gp_6 for details)
            D_out = self.discriminator(fake_data.detach())
            #print(D_out)
            fake_labels = torch.zeros(D_out.shape[0]).to(self.device)
            D_fakeloss = self.loss_metric(D_out, fake_labels)
            D_fakeloss.backward()
            
            # update the parameters and collect total loss
            D_totalloss = D_realloss + D_fakeloss
        
            if train_discriminator:
                self.grad_clip(self.discriminator.parameters(), self.D_optim)

            '''
            GENERATOR TRAINING 
            maximize log(D(G(z)))
            '''
            self.asr_model.encoder.zero_grad()

            # fake labels for the listener are the true labels for the
            # discriminator 
                        
            '''
            we cant call .detach() here, to avoid gradient calculations
            on the discriminator, since then we would lose the history needed
            to update the generator.

            x -> G -> fake_data -> D -> D_out -> Calculate loss
            
            vs.

            x -> G -> fake_data -> fake_data.detach()
            [    LOST HISTORY    ]      | -> D -> D_out -> Calculate loss
            '''
            D_out = self.discriminator(fake_data) # [bs*text_seq, 1]
            fake_labels = torch.ones(D_out.shape[0]).to(self.device)

            G_loss = self.loss_metric(D_out.squeeze(dim=1), fake_labels)
            G_loss.backward()
            if train_generator:
                self.grad_clip(self.asr_model.encoder.parameters(), self.G_optim)
            
            self.lg.scalar('adversiarial_sanity', {'discriminator_fake':D_fakeloss,
                'discriminator_real': D_realloss, 'generator':G_loss}, it)

            self.verbose("({}/{}) D_fake: {:.4f}, D_real: {:.4f}, G : {:.4f}"
                .format(it+1, self.sanity_steps, D_fakeloss, D_realloss, G_loss), progress=True)

            if it % 50 == 0:
                # get the first real and first fake embedding
                embs = torch.cat((real_emb, fake_emb))
                meta = []
                meta = meta + ['real' for _ in range(real_emb.shape[0])]
                meta = meta + ['fake' for _ in range(fake_emb.shape[0])]
                self.lg.embedding('sanity_emb', embs, meta, it)

    def exec(self):
        '''
        Adversarial training:
        * (G)enerator: Listener (from LAS)
        * (D)iscriminator: This discriminator module
        * Data distribution: The text encoder
        '''
        self.verbose('Training set total {} batches'.format(len(self.train_set)))
        epoch = 0
        while epoch < self.n_epochs:
            self.verbose("Starting epoch {} out of {}".format(epoch+1, self.n_epochs))
            for b_idx, (x, y) in enumerate(self.train_set):
                self.verbose('Global step - {} ( {} / {} )'.format(
                    self.tr.step, b_idx, len(self.train_set)), progress=True)     

                x, x_lens = prepare_x(x, device=self.device)    
                y, y_lens = prepare_y(y, device=self.device)
                batch_size = x.shape[0]
                '''
                DISCRIMINATOR TRAINING
                maximize log(D(x)) + log(1 - D(G(z)))
                '''
                self.discriminator.zero_grad()

                '''Discriminator real data training'''
                real_data = self.data_distribution(y) # [bs, seq, 512]
                D_out = self.discriminator(real_data)
                real_labels = torch.ones(batch_size, real_data.shape[1]).to(self.device) \
                    - self.config['adv']['label_smoothing']

                # TODO why use squeeze here ? 
                D_realloss = self.loss_metric(D_out.squeeze(dim=2), real_labels)
                D_realloss.backward()

                '''Discriminator fake data training'''
                fake_data, _ = self.asr_model.encoder(x, x_lens)
                # Note: fake_data.detach() is used here to avoid backpropping
                # through the generator. (see grad_pointers.gp_6 for details)
                D_out = self.discriminator(fake_data.detach())
                fake_labels = torch.zeros(batch_size, fake_data.shape[1]).to(self.device)
                D_fakeloss = self.loss_metric(D_out.squeeze(dim=2), fake_labels)
                D_fakeloss.backward()
            
                # update the parameters and collect total loss
                D_totalloss = D_realloss + D_fakeloss

                self.grad_clip(self.discriminator.parameters(), self.D_optim)

                '''
                GENERATOR TRAINING 
                maximize log(D(G(z)))
                '''
                self.asr_model.encoder.zero_grad()

                # fake labels for the listener are the true labels for the
                # discriminator 
                fake_labels = torch.ones(batch_size, fake_data.shape[1]).to(self.device)
                
                '''
                we cant call .detach() here, to avoid gradient calculations
                on the discriminator, since then we would lose the history needed
                to update the generator.

                x -> G -> fake_data -> D -> D_out -> Calculate loss
                
                vs.

                x -> G -> fake_data -> fake_data.detach()
                [    LOST HISTORY    ]      | -> D -> D_out -> Calculate loss
                '''
                D_out = self.discriminator(fake_data)

                G_loss = self.loss_metric(D_out.squeeze(dim=2), fake_labels)
                G_loss.backward()
                self.grad_clip(self.asr_model.encoder.parameters(), self.G_optim)

                if self.tr.step % self.logging_step == 0:
                    self.lg.scalar('discrim_real_loss_train', D_realloss, self.tr.step)
                    self.lg.scalar('discrim_fake_loss_train', D_fakeloss, self.tr.step)
                    self.lg.scalar('discrim_loss_train', D_totalloss, self.tr.step)
                    self.lg.scalar('gen_loss_train', G_loss, self.tr.step)

                if self.tr.step % self.valid_step == 0:
                    self.G_optim.zero_grad()
                    self.D_optim.zero_grad()
                    self.valid()

                if self.tr.step % self.save_step == 0:
                    # save the model as a precaution
                    self.verbose("Model saved at step {}".format(self.tr.step))
                    torch.save(self.discriminator.state_dict(), self.ckppath)
                    torch.save(self.asr_model.state_dict(), self.asrpath_out)

                self.tr.do_step()
            epoch += 1

    def valid(self):
        self.asr_model.eval()
        self.discriminator.eval()

        avg_real_loss = 0.0
        avg_fake_loss = 0.0
        n_batches = 0
        for b_idx, (x, y) in enumerate(self.valid_set):
            self.verbose('Validation step - {} ( {} / {} )'.format(
                self.tr.step, b_idx, len(self.valid_set)), progress=True)
            
            x, x_lens = prepare_x(x, device=self.device)    
            y, y_lens = prepare_y(y, device=self.device)

            batch_size = x.shape[0]

            '''Discriminator real data training'''
            real_data = self.data_distribution(y) # [bs, seq, 512]
            D_out = self.discriminator(real_data)
            real_labels = torch.ones(batch_size, real_data.shape[1]).to(self.device)
            
            D_realloss = self.loss_metric(D_out.squeeze(dim=2), real_labels)

            '''Discriminator fake data training'''
            fake_data, _ = self.asr_model.encoder(x, x_lens)
            fake_emb = fake_data.view(batch_size*fake_data.shape[1], -1)

            # Note: fake_data.detach() is used here to avoid backpropping
            # through the generator. (see grad_pointers.gp_6 for details)
            D_out = self.discriminator(fake_data.detach())
            fake_labels = torch.zeros(batch_size, fake_data.shape[1]).to(self.device)
            D_fakeloss = self.loss_metric(D_out.squeeze(dim=2), fake_labels)
            
            # update the parameters and collect total loss
            D_totalloss = D_realloss + D_fakeloss

            avg_real_loss += D_realloss.detach()
            avg_fake_loss += D_fakeloss.detach()

            n_batches += 1
        avg_real_loss /= n_batches
        avg_fake_loss /= n_batches

        # embedding logging for last 
        real_emb = real_data[0,:,:]
        fake_emb = fake_data[0,:,:]
        embs = torch.cat((real_emb, fake_emb))
        meta = []
        meta = meta + ['real' for _ in range(real_emb.shape[0])]
        meta = meta + ['fake' for _ in range(fake_emb.shape[0])]
        self.lg.embedding('validation_emb', embs, meta, self.tr.step)

        avg_loss = avg_real_loss + avg_fake_loss
        self.lg.scalar('discrim_real_loss_eval', avg_real_loss, self.tr.step)
        self.lg.scalar('discrim_fake_loss_eval', avg_fake_loss, self.tr.step)
        self.lg.scalar('discrim_loss_eval', avg_loss, self.tr.step)
         
        if avg_loss < self.tr.get_best():
            # then we save the model
            self.tr.set_best(avg_loss.item())
            self.verbose('Best validation loss : {:.4f} @ global step {}'
                .format(self.tr.get_best(), self.tr.step))
            torch.save(self.discriminator.state_dict(), self.best_ckppath)
            self.verbose("Both the discriminator and ASR have been saved")

        self.asr_model.train()
        self.discriminator.train()

    def close(self):
        '''
        Save the most recent model
        '''
        self.verbose("Finished training! The most recent model will"+\
            "be saved at step {} as well as the ASR model".format(self.tr.step))
        torch.save(self.discriminator.state_dict(), self.ckppath)
        torch.save(self.asr_model.state_dict(), self.asrpath_out)


def asr_seed_train(config, paras, super_its=1):
    '''
    Input arguments:
    * config, paras (see e.g. solver)
    * super_its (int): The number of times we iterate the
    execution sequence listed below

    This trainer will:
    1) Train the text autoencoder. This saves:
        * Most recent TAE
        * Best TAE
        * Most recent ASR
    2) Adversarial training
        * Most recent Discriminator
        * Best discriminator
        * Most recent ASR
    3) Train the speech autoencoder
        * Most recent SAE
        * Best SAE
        * Most recent ASR
    '''
    ckpdir = os.path.join(paras.ckpdir, paras.name)
    for i in range(super_its):
        print('Starting Super Iteration {}'.format(i+1))
        # TRAIN TAE
        print('Starting TAE training')
        tae_solver = TAETrainer(config, paras)
        tae_solver.load_data()
        tae_solver.set_model(asrpath=(os.path.join(ckpdir, 'asr_1.cpt'), 
            os.path.join(ckpdir, 'asr_1.cpt')))
        tae_solver.exec()
        tae_solver.close()
        tae_path = tae_solver.ckppath

        del tae_solver

        # TRAIN ADV
        print('Starting ADV training')
        adv_solver = ADVTrainer(config, paras)
        adv_solver.load_data()
        adv_solver.set_model(taepath=tae_path, asrpath=(os.path.join(ckpdir, 'asr_1.cpt'),
            os.path.join(ckpdir, 'asr_2.cpt')))
        adv_solver.exec()
        adv_solver.close()
        del adv_solver
        
        # TRAIN SAE
        print('Starting SAE training')
        sae_solver = SAETrainer(config, paras)
        sae_solver.load_data()
        sae_solver.set_model(asrpath=(os.path.join(ckpdir, 'asr_2.cpt'),
            os.path.join(ckpdir, 'asr_3.cpt')))
        sae_solver.exec()
        sae_solver.close()
        del sae_solver

def sae_extra_train(config, paras):
    '''
    Input arguments:
    * config, paras (see e.g. solver)
    * super_its (int): The number of times we iterate the
    execution sequence listed below

    This trainer will:
    1) Train the text autoencoder. This saves:
        * Most recent TAE
        * Best TAE
        * Most recent ASR
    2) Adversarial training
        * Most recent Discriminator
        * Best discriminator
        * Most recent ASR
    3) Train the speech autoencoder
        * Most recent SAE
        * Best SAE
        * Most recent ASR
    '''
    ckpdir = os.path.join(paras.ckpdir, paras.name)

    # TRAIN SAE
    print('Starting SAE training')
    adv_solver = ADVTrainer(config, paras)
    adv_solver.load_data()
    taepath = os.path.join(adv_solver.ckpdir, 'tae.cpt')
    adv_solver.set_model(taepath=taepath, asrpath=(os.path.join(ckpdir, 'asr_2.cpt'),
        os.path.join(ckpdir, 'asr_3.cpt')))
    adv_solver.exec()
    adv_solver.close()
