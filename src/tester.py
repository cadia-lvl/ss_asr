class Tester(Solver):
    ''' Handler for complete inference progress'''
    def __init__(self,config,paras):
        super(Tester, self).__init__(config,paras)
        self.verbose('During beam decoding, batch size is set to 1, please speed up with --njobs.')
        self.njobs = self.paras.njobs
        self.decode_step_ratio = config['solver']['max_decode_step_ratio']
        
        self.decode_file = "_".join(['decode','beam',str(self.config['solver']['decode_beam_size']),
                                     'len',str(self.config['solver']['max_decode_step_ratio'])])

    def load_data(self):
        self.verbose('Loading testing data '+str(self.config['solver']['test_set'])\
                     +' from '+self.config['solver']['data_path'])
        setattr(self,'test_set',LoadDataset('test',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))

    def set_model(self):
        ''' Load saved ASR'''
        self.verbose('Load ASR model from '+os.path.join(self.ckpdir))
        self.asr_model = torch.load(os.path.join(self.ckpdir,'asr'))
        
        # Enable joint RNNLM decoding
        self.decode_lm = self.config['solver']['decode_lm_weight'] >0
        setattr(self.asr_model,'decode_lm_weight',self.config['solver']['decode_lm_weight'])
        if self.decode_lm:
            assert os.path.exists(self.config['solver']['decode_lm_path']), 'Please specify RNNLM path.'
            self.asr_model.load_lm(**self.config['solver'])
            self.verbose('Joint RNNLM decoding is enabled with weight = '+str(self.config['solver']['decode_lm_weight']))
            self.verbose('Loading RNNLM from '+self.config['solver']['decode_lm_path'])
            self.decode_file += '_lm{:}'.format(self.config['solver']['decode_lm_weight'])
        
        # Check models dev performance before inference
        self.asr_model.eval()
        self.asr_model.clear_att()
        self.asr_model = self.asr_model.to(self.device)
        self.verbose('Checking models performance on dev set '+str(self.config['solver']['dev_set'])+'...')
        self.valid()
        self.asr_model = self.asr_model.to('cpu') # move origin model to cpu, clone it to GPU for each thread

    def exec(self):
        '''Perform inference step with beam search decoding.'''
        test_cer = 0.0
        self.decode_beam_size = self.config['solver']['decode_beam_size']
        self.verbose('Start decoding with beam search, beam size = '+str(self.config['solver']['decode_beam_size']))
        self.verbose('Number of utts to decode : {}, decoding with {} threads.'.format(len(self.test_set),self.njobs))
        _ = Parallel(n_jobs=self.njobs)(delayed(self.beam_decode)(x[0],y[0].tolist()[0]) for x,y in tqdm(self.test_set))
        
        self.verbose('Decode done, best results at {}.'.format(str(os.path.join(self.ckpdir,self.decode_file+'.txt'))))
        
        self.verbose('Top {} results at {}.'.format(self.config['solver']['decode_beam_size'],
                                                    str(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'))))
        
    def write_hyp(self,hyps,y):
        '''Record decoding results'''
        gt = self.mapper.translate(y,return_string=True)
        # Best
        with open(os.path.join(self.ckpdir,self.decode_file+'.txt'),'a') as f:
            best_hyp = self.mapper.translate(hyps[0].outIndex,return_string=True)
            f.write(gt+'\t'+best_hyp+'\n')
        # N best
        with open(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'),'a') as f:
            for hyp in hyps:
                best_hyp = self.mapper.translate(hyp.outIndex,return_string=True)
                f.write(gt+'\t'+best_hyp+'\n')
        

    def beam_decode(self,x,y):
        '''Perform beam decoding with end-to-end ASR'''
        # Prepare data
        x = x.to(device = self.device,dtype=torch.float32)
        state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
        state_len = [int(sl) for sl in state_len]

        # Forward
        with torch.no_grad():
            max_decode_step =  int(np.ceil(state_len[0]*self.decode_step_ratio))
            model = copy.deepcopy(self.asr_model).to(self.device)
            hyps = model.beam_decode(x, max_decode_step, state_len, self.decode_beam_size)
        del model
        
        self.write_hyp(hyps,y)
        del hyps
        
        return 1

    
    def valid(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding on Attention decoder only)'''
        val_cer = 0.0
        val_len = 0    
        all_pred,all_true = [],[]
        with torch.no_grad():
            for cur_b,(x,y) in enumerate(self.eval_set):
                self.progress(' '.join(['Valid step - (',str(cur_b),'/',str(len(self.eval_set)),')']))

                # Prepare data
                if len(x.shape)==4: x = x.squeeze(0)
                if len(y.shape)==3: y = y.squeeze(0)
                x = x.to(device = self.device,dtype=torch.float32)
                y = y.to(device = self.device,dtype=torch.long)
                state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
                state_len = [int(sl) for sl in state_len]
                ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                # Forward
                state_len, att_pred, att_maps = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)

                # Result
                label = y[:,1:ans_len+1].contiguous()
                t1,t2 = calc_er(att_pred,label,mapper=self.mapper,get_sentence=True)
                all_pred += t1
                all_true += t2
                val_cer += calc_er(att_pred,label,mapper=self.mapper)*int(x.shape[0])
                val_len += int(x.shape[0])
        
        
        # Dump model score to ensure model is corrected
        self.verbose('Validation Error Rate of Current model : {:.4f}      '.format(val_cer/val_len)) 
        self.verbose('See {} for validation results.'.format(os.path.join(self.ckpdir,'dev_att_decode.txt'))) 
        with open(os.path.join(self.ckpdir,'dev_att_decode.txt'),'w') as f:
            for hyp,gt in zip(all_pred,all_true):
                f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')