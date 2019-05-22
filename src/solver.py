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
from tqdm import tqdm

# Import modules
from asr import ASR
from text_autoencoder import TextAutoEncoder
from speech_autoencoder import SpeechAutoEncoder
from discriminator import Discriminator
from charlm import CharLM

# Data, processing and tracking imports
from TrackerHandler import TrackerHandler
from LogHandler import LogHandler
from dataset import load_dataset, prepare_x, prepare_y
from text_dataset import LMDataset, dataload
from postprocess import calc_acc, calc_err, draw_att
from preprocess import SOS_TKN


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
        self.ckpdir = os.path.join(paras.ckpdir,self.paras.name)
        if not os.path.exists(self.ckpdir):
            os.makedirs(self.ckpdir)

        # setup tracker and logger        
        self.tr = TrackerHandler(os.path.join(self.ckpdir, 'tracker.json'),
            self.module_id)
        self.lg = LogHandler(
            os.path.join(self.paras.logdir, self.paras.name, self.module_id),
            self.module_id)

        self.ckppath = os.path.join(self.ckpdir, self.module_id+'.cpt')
        self.best_ckppath = os.path.join(self.ckpdir, self.module_id+'_best.cpt')

        # default values for some class variables
        self.valid_step = self.set_if_exists('valid_step', 500)
        self.logging_step = self.set_if_exists('logging_step', 250)
        self.save_step = self.set_if_exists('save_step', 1000)
        self.n_epochs = self.set_if_exists('n_epochs', 5)
        self.train_batch_size = self.set_if_exists('train_batch_size', 32)
        self.valid_batch_size = self.set_if_exists('valid_batch_size', 32)
        self.test_batch_size = self.set_if_exists('test_batch_size', 1)

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
            msg = '[INFO ({})] '.format(self.module_id) + str(msg)

        if self.paras.verbose:
            print(msg, end=end)
    
    def grad_clip(self, params, optim, grad_clip=5):
        grad_norm = nn.utils.clip_grad_norm_(params, grad_clip)
        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.tr.step))
        else:
            optim.step()

    def setup_module(self, module, ckp_path, *para, **model_para):
        '''
        Given a model module type and parameters, the model will be created
        and if a pre-trained model exists, it will be loaded.

        Input arguments:
        * module (nn.Module): A module type
        * ckp_path (str): A path to where the checkpoint for the module would
        be stored
        * para: Any number of positional arguments passed to the module
        * model_para: Any number of keyword arguments passed to the module 
        '''
        model = module(*para, **model_para)

        if os.path.isfile(ckp_path):
            self.verbose('Loading a pretrained model from {}'.format(ckp_path))
            model.load_state_dict(torch.load(ckp_path))
        else:
            self.verbose('No model found at {}. A new model will be created'.format(ckp_path))
        return model.to(self.device)
    
class CHARLMTrainer(Solver):
    def __init__(self, config, paras):
        super(CHARLMTrainer, self).__init__(config, paras, 'char_lm')

    def load_data(self):
        self.chunk_size = self.config['char_lm']['chunk_size']
        self.tf_rate = self.config['char_lm']['tf_rate']

        # TODO: Use GPU is not used for this dataloader, check if we should
        self.ds, self.train_set = dataload(self.config['char_lm']['train_index'], 
            self.chunk_size, self.train_batch_size, shuffle=True)

    def set_model(self):
        self.loss_metric = nn.CrossEntropyLoss(reduction='none').to(self.device)

        self.lm = self.setup_module(CharLM, self.ckppath, self.ds.get_num_chars(),            
            self.config['char_lm']['hidden_size'])
        
        # setup optimizer
        self.optim = getattr(torch.optim,
            self.config['char_lm']['optimizer']['type'])
        self.optim = self.optim(self.lm.parameters(), 
            lr=self.config['char_lm']['optimizer']['learning_rate'], eps=1e-8)
    
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
        x = x[-1]#.view(-1)
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
        (self.mapper, _ ,self.train_set) = load_dataset(
            self.config['asr']['train_index'],
            batch_size=self.train_batch_size, use_gpu=self.paras.gpu)
        
        (_, _, self.valid_set) = load_dataset(
            self.config['asr']['valid_index'], 
            batch_size=self.valid_batch_size, use_gpu=self.paras.gpu)

        self.wer_step = self.config['asr']['wer_step']
        
    def set_model(self, asr=None):
        ''' Setup ASR'''        
        self.seq_loss = nn.CrossEntropyLoss(ignore_index=0, 
            reduction='none').to(self.device)
        
        # Build attention end-to-end ASR
        if asr is None:
            self.asr_model = self.setup_module(ASR, self.ckppath, self.mapper.get_dim(),
                **self.config['asr']['model_para'])
        else:
            # a pretrained model has been passed to the solver.
            self.asr_model = asr
        
        # setup optimizer
        self.optim = getattr(torch.optim,
            self.config['asr']['optimizer']['type'])
        self.optim = self.optim(self.asr_model.parameters(), 
            lr=self.config['asr']['optimizer']['learning_rate'], eps=1e-8)

    def get_asr_model(self):
        return self.asr_model

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
                loss = self.seq_loss(prediction.view(b*t,c),label.view(-1))
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

                if self.tr.step % self.valid_step == 0 and self.tr.step != 0:
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

            # Forward
            state_len, prediction, att_map = self.asr_model(
                x, ans_len+30, state_len=state_len)

            # Compute attention loss & get decoding results
            label = y[:,1:ans_len+1].contiguous()
           
            loss = self.seq_loss(prediction[:,:ans_len,:]
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
            self.lg.text('eval_hyp_'+str(idx),val_hyp[idx], self.tr.step)
            self.lg.text('eval_txt_'+str(idx),val_txt[idx], self.tr.step)
 
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
        (self.mapper, _ ,self.test_set) = load_dataset(
            self.config['asr']['test_index'],
            batch_size=self.batch_size, use_gpu=self.paras.gpu)
        
    def set_model(self):
        ''' Load saved ASR'''
        self.asr_model = self.setup_module(ASR, self.ckppath, self.mapper.get_dim(),
            **self.config['asr']['model_para'])
        # move origin model to cpu, clone it to GPU for each thread
        self.asr_model.eval()
        #self.asr_model = self.asr_model.to('cpu')

        self.rnnlm = self.setup_module(LM, os.path.join(self.ckpdir, 'rnn_lm.cpt'), self.mapper.get_dim(), 
            **self.config['rnn_lm']['model_para'])
        
        self.rnnlm.eval()

        self.lm_weight = self.config['asr']['decode_lm_weight']
        self.decode_beam_size = self.config['asr']['decode_beam_size']
        self.njobs = self.config['asr']['decode_jobs']
        self.decode_step_ratio = self.config['asr']['max_decode_step_ratio']

        self.decode_file += '_lm{:}'.format(self.config['asr']['decode_lm_weight'])
    
    def exec(self):
        '''Perform inference step with beam search decoding.'''
        self.verbose('Start decoding with beam search, beam size: {}'
            .format(self.decode_beam_size))
        self.verbose('Number of utts to decode : {}, decoding with {} threads.'
            .format(len(self.test_set),self.njobs))
        
        for b_ind, (x, y) in enumerate(self.test_set):
            x, x_len = prepare_x(x, self.device)
            y, _ = prepare_y(y, self.device)
            # TODO: we are using simple decoding, not beam decoding
            self.asr_model.decode(x, x_len, self.rnnlm, self.mapper)
                   
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

        (self.mapper, self.dataset, self.train_set) = load_dataset(
            self.config['tae']['train_index'], batch_size=self.train_batch_size, 
            use_gpu=self.paras.gpu, text_only=True, drop_rate=self.config['tae']['drop_rate'])

        (_, _, self.valid_set) = load_dataset(
            self.config['tae']['valid_index'], batch_size=self.valid_batch_size, 
            use_gpu=self.paras.gpu, text_only=True, drop_rate=self.config['tae']['drop_rate'])

    def set_model(self, asr_model=None):
        if asr_model is not None:
            self.asr_model = asr_model
        else:
            self.asr_model = self.setup_module(ASR, os.path.join(self.ckpdir, 'asr.cpt'), 
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

    def get_tae_model(self):
        return self.text_autoenc

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
                
                if self.tr.step % self.valid_step == 0 and self.tr.step != 0:
                    self.optim.zero_grad()
                    self.valid()
                
                self.tr.step += 1
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
            self.lg.scalar('eval_loss', loss, self.tr.step)
        
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
        if avg_loss < self.tr.get_best():
            # then we save the model
            self.tr.set_best(avg_loss)
            self.verbose('Best validation loss : {:.4f} @ global step {}'.format(
                self.tr.get_best(), self.tr.step))
            torch.save(self.text_autoenc.state_dict(), self.best_ckppath)
            # TODO: swap with real ASR
            torch.save(self.asr_model.state_dict(), os.path.join(self.ckpdir, 'modified_asr.cpt'))
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
        # TODO: swap with real ASR
        torch.save(self.asr_model.state_dict(), os.path.join(self.ckpdir, 'modified_asr.cpt'))

class SAETrainer(Solver):
    '''
    Train the Speech AutoEncoder.
    '''
    def __init__(self, config, paras):
        super(SAETrainer, self).__init__(config, paras, 'sae')
    
    def load_data(self):
        # data must be sorted by length of x.

        (self.mapper, _, self.train_set) = load_dataset(
            self.config['sae']['train_index'], 
            batch_size=self.train_batch_size, use_gpu=self.paras.gpu)

        (_, _, self.valid_set) = load_dataset(
            self.config['sae']['valid_index'], 
            batch_size=self.valid_batch_size, use_gpu=self.paras.gpu)
    
    def set_model(self, asr_model=None):
        if asr_model is not None:
            self.asr_model = asr_model
        else:
            self.asr_model = self.setup_module(ASR, os.path.join(self.ckpdir, 'asr.cpt'), 
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

        self.loss_metric = nn.SmoothL1Loss(reduction='none').to(self.device)

    def exec(self):
        self.verbose('Training set total {} batches.'.format(len(self.train_set)))
        epoch = 0
        while epoch < self.n_epochs:
            self.verbose("Starting epoch {} out of {}".format(epoch+1, self.n_epochs))
            for b_ind, (x, y) in enumerate(self.train_set):
                self.verbose('Batch: {}/{}, global step: {}'.format(
                    b_ind, len(self.train_set), self.tr.step), progress=True)

                x, x_lens = prepare_x(x, device=self.device)
                self.optim.zero_grad()

                # shape is [batch_size, 8*(~seq/8), feature_dim]
                autoenc_out = self.speech_autoenc(self.asr_model, x, x_lens)
                # pad the encoder output UP to the maximum batch time frames and 
                # pad x DOWN to the same number of frames
                batch_t = max(x_lens)
                x = x[:, :batch_t, :]
                enc_final = torch.zeros([autoenc_out.shape[0], batch_t, 
                    autoenc_out.shape[2]]).to(self.device)
                enc_final[:, :autoenc_out.shape[1], :] = autoenc_out
                loss = self.loss_metric(enc_final, x)

                # Divide each by length of x and sum, then take the average over the
                # batch
                loss = torch.sum(loss.view(loss.shape[0], -1))/torch.Tensor(x_lens)\
                    .to(self.device)
                loss = torch.mean(loss)
                loss.backward()
                self.grad_clip(self.speech_autoenc.parameters(), self.optim)

                self.lg.scalar('train_loss', loss, self.tr.step)

                if self.tr.step % self.valid_step == 0 and self.tr.step != 0:
                    self.optim.zero_grad()
                    self.valid()

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

            enc_out = self.speech_autoenc(self.asr_model, x, x_lens)
            # pad the encoder output UP to the maximum batch time frames and 
            # pad x DOWN to the same number of frames
            batch_t = max(x_lens)
            x = x[:, :batch_t, :]
            enc_final = torch.zeros([enc_out.shape[0], batch_t, enc_out.shape[2]]).to(self.device)
            enc_final[:, :enc_out.shape[1], :] = enc_out

            loss = self.loss_metric(enc_final, x)

            # Divide each by length of x and sum, then take the average over the
            # batch
            loss = torch.sum(loss.view(loss.shape[0], -1)) / torch.Tensor(x_lens)\
                .to(self.device)
            loss = torch.mean(loss)

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
            # TODO swap with real ASR
            torch.save(self.asr_model.state_dict(), os.path.join(self.ckpdir, 'modified_asr.cpt'))
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
        # TODO: swap with real ASR
        torch.save(self.asr_model.state_dict(), os.path.join(self.ckpdir, 'modified_asr.cpt'))

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
        self.max_step = self.config['adv']['total_steps']

        (self.mapper, self.dataset, self.train_set) = load_dataset(
            self.config['adv']['train_index'], 
            batch_size=self.config['adv']['train_batch_size'], 
            use_gpu=self.paras.gpu)

        (_, _, self.valid_set) = load_dataset(
            self.config['adv']['eval_index'], 
            batch_size=self.config['adv']['valid_batch_size'],
            use_gpu=self.paras.gpu)
        
    def set_model(self, asr_model=None, tae_model=None):
        if asr_model is not None:
            self.asr_model = asr_model
        else:
            self.asr_model = self.setup_module(ASR, os.path.join(self.ckpdir, 'asr.cpt'), 
                self.mapper.get_dim(), **self.config['asr']['model_para'])
        if tae_model is not None:
            self.text_autoenc = tae_model
        else:
            self.text_autoenc = self.setup_module(TextAutoEncoder, 
                os.path.join(self.ckpdir, 'tae.cpt'), self.mapper.get_dim(), 
                **self.config['tae']['model_para'])        

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

    def exec(self):
        '''
        Adversarial training:
        * (G)enerator: Listener (from LAS)
        * (D)iscriminator: This discriminator module
        * Data distribution: The text encoder
        '''
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
            #self.discriminator.zero_grad()

            '''Discriminator real data training'''
            real_data = self.data_distribution(y) # [bs, seq, 512]
            D_out = self.discriminator(real_data)
            real_labels = torch.ones(batch_size, real_data.shape[1]).to(self.device) \
                - self.config['adv']['label_smoothing']

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

            self.lg.scalar('discrim_real_loss_train', D_realloss, self.tr.step)
            self.lg.scalar('discrim_fake_loss_train', D_fakeloss, self.tr.step)
            self.lg.scalar('discrim_loss_train', D_totalloss, self.tr.step)


            '''
            GENERATOR TRAINING 
            maximize log(D(G(z)))
            '''
            #self.asr_model.encoder.zero_grad()

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

            self.lg.scalar('gen_loss_train', G_loss)


            if self.tr.step % self.valid_step == 0 and self.tr.step != 0:
                self.G_optim.zero_grad()
                self.D_optim.zero_grad()
                self.valid()
            
            self.tr.step += 1
            
            if self.tr.step > self.max_step:
                self.verbose('Stopping after reaching maximum training steps')
                break
        
    def valid(self):
        self.discriminator.eval()

        avg_loss = 0.0
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
            # Note: fake_data.detach() is used here to avoid backpropping
            # through the generator. (see grad_pointers.gp_6 for details)
            D_out = self.discriminator(fake_data.detach())
            fake_labels = torch.zeros(batch_size, fake_data.shape[1]).to(self.device)
            D_fakeloss = self.loss_metric(D_out.squeeze(dim=2), fake_labels)
            
            # update the parameters and collect total loss
            D_totalloss = D_realloss + D_fakeloss

            self.lg.scalar('discrim_real_loss_eval', D_realloss, self.tr.step)
            self.lg.scalar('discrim_fake_loss_eval', D_fakeloss, self.tr.step)
            self.lg.scalar('discrim_loss_eval', D_totalloss, self.tr.step)
            avg_loss += D_totalloss.detach()
            n_batches += 1
        
        avg_loss /=  n_batches
        avg_loss = avg_loss.item()      
        if avg_loss < self.tr.get_best():
            # then we save the model
            self.tr.set_best(avg_loss)

            self.verbose('Best validation loss : {:.4f} @ global step {}'
                .format(self.tr.get_best(), self.tr.step))
            torch.save(self.discriminator.state_dict(), self.best_ckppath)

        self.discriminator.train()

    def close(self):
        '''
        Save the most recent model
        '''
        self.verbose("Finished training! The most recent model will"+\
            "be saved at step {} as well as the ASR model".format(self.tr.step))
        torch.save(self.discriminator.state_dict(), self.ckppath)
        # TODO: swap with real ASR
        #torch.save(self.asr_model.state_dict(), os.path.join(self.ckpdir, 'modified_asr.cpt'))

'''
SuperSolver combines every training solver in src.solver.py and runs each
in turns. The asr core is only saved when it's evaluation loss is decreased
'''
class SuperSolver:
    def __init__(self, config, paras):
        
        self.config = config
        self.paras = paras

        self.asr_solver = ASRTrainer(self.config, self.paras)
        self.asr_solver.load_data()
        self.asr_solver.set_model()
        self.asr_model = self.asr_solver.get_asr_model()

        self.tae_solver = TAETrainer(self.config, self.paras)
        self.tae_solver.load_data()
        self.tae_solver.set_model(asr_model=self.asr_model)
        self.tae_model = self.tae_solver.get_tae_model()
        
        self.sae_solver = SAETrainer(self.config, self.paras)
        self.sae_solver.load_data()
        self.sae_solver.set_model(asr_model=self.asr_model)

        self.adv_solver = AdvTrainer(self.config, self.paras)
        self.adv_solver.load_data()
        self.adv_solver.set_model(asr_model=self.asr_model, tae_model=self.tae_model)

        self.superloops = 5

    def exec(self):
        # each solver takes care of logging, evaluating, checkpointing
        # and so on
        for i in range(self.superloops):
            print('Starting superloop #{}'.format(i+1))
            self.asr_solver.exec()
            self.tae_solver.exec()
            self.sae_solver.exec()
            self.adv_solver.exec()
