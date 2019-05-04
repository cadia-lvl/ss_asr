import torch
import yaml

from dataset import load_dataset, prepare_x, prepare_y
from asr import ASR

import matplotlib.pyplot as plt


VAL_STEP = 30 

def asr_test(asr_cpt: str, index_path: str, conf_path: str):
    '''
    1. Reload an ASR
    2. Create a dataset
    3. Evaluate
    '''
    if torch.cuda.is_available(): 
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')

    config = yaml.load(open(conf_path ,'r'), Loader=yaml.FullLoader)
    (mapper, _ , train_set) = load_dataset(
            index_path,
            batch_size=10,
            use_gpu=torch.cuda.is_available())


    asr_model = ASR(mapper.get_dim(), **config['asr_model']['model_para'])
    asr_model.load_state_dict(torch.load(asr_cpt))
    asr_model.eval()

    b_idx, (x, y) = next(enumerate(train_set))

    (x, x_lens) = prepare_x(x, device=device)
    (y, y_lens) = prepare_y(y, device=device)

    # TODO: Same issue here as in self.exec()
    state_len = x_lens
    ans_len = max(y_lens) - 1

    # Forward
    state_len, prediction, att_map = asr_model(
        x, ans_len+VAL_STEP, state_len=state_len)

    print(att_map.shape)

    first = att_map[0, :, :].view(att_map.shape[1], att_map.shape[2]).numpy()

    plt.imshow(first)
    plt.show()


    '''
    # Compute attention loss & get decoding results
    label = y[:,1:ans_len+1].contiguous()
    
    seq_loss = self.seq_loss(prediction[:,:ans_len,:]
        .contiguous().view(-1,prediction.shape[-1]), label.view(-1))

    # Sum each uttr and devide by length
    seq_loss = torch.sum(seq_loss.view(x.shape[0],-1),dim=-1)\
        /torch.sum(y!=0,dim=-1).to(device=self.device, dtype=torch.float32)
    # Mean by batch
    seq_loss = torch.mean(seq_loss)
    
    loss += seq_loss.detach()*int(x.shape[0])
    
    mapped_prediction = [self.mapper.translate(p) for p in 
        np.argmax(prediction.cpu().detach(), axis=-1)]
    mapped_label = [self.mapper.translate(l) for l in label.cpu()]

    predictions.append(mapped_prediction)
    labels.append(mapped_label)
    
    acc += calc_acc(prediction,label)*int(x.shape[0])
    err += calc_err(prediction,label,mapper=self.mapper)*int(x.shape[0])
    
    val_len += int(x.shape[0])

    # Logger
    self.log_scalar('eval_loss', loss/val_len)
                    
    # Plot attention map to log for the last batch in the validation
    # dataset.
    val_hyp = [self.mapper.translate(p) for p in 
        np.argmax(prediction.cpu().detach(), axis=-1)]
    val_txt = [self.mapper.translate(l) for l in label.cpu()]
    val_attmaps = draw_att(att_map)
    
    # Record loss
    self.log_scalar('eval_error', err/val_len)
    self.log_scalar('eval_acc',  acc/val_len)
    
    for idx, attmap in enumerate(val_attmaps):
        #plt.imshow(attmap)
        #plt.show()
        self.log_image('eval_att_'+str(idx), attmap)
        self.log_text('eval_hyp_'+str(idx),val_hyp[idx])
        self.log_text('eval_txt_'+str(idx),val_txt[idx])

    # Save model by val er.
    if err/val_len  < self.best_val_loss:
        self.best_val_loss = err/val_len
        self.verbose('Best validation loss for ASR : {:.4f} @ global step {}'
            .format(self.best_val_loss, self.step))

        self.set_globals(self.step, self.best_val_loss)
        
        torch.save(self.asr_model.state_dict(), self.ckppath)
        
        # Save hyps.
        with open(os.path.join(self.ckpdir, 'best_hyp.txt'), 'w') as f:
            for t1,t2 in zip(predictions, labels):
                f.write(t1[0]+','+t2[0]+'\n')
    '''


if __name__ == '__main__':
    asr_test('./result/newertest/asr.cpt', './data/processed/index.tsv', './conf/test.yaml')