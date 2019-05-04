import torch
import yaml
import numpy as np

from dataset import load_dataset, prepare_x, prepare_y
from asr import ASR
from postprocess import trim_eos


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
    asr_model = asr_model.to(device)
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
    
    print(state_len)
    for i in range(x.shape[0]):
        label = y[i, :].view(-1)
        print("Current label: ", label)
        print("Label mapped: ", mapper.translate(label))
    

        pred = prediction[i, :, :].view(prediction.shape[1], prediction.shape[2])
        pred = np.argmax(pred.cpu().detach(), axis=-1)
        att_len = len(trim_eos(pred))
        print("Prediction shape: ", pred.shape)
        print("The actual prediction: ", pred)
        print("The mapped version: ",mapper.translate(pred))

        attmap = att_map[i, :att_len, :].numpy()
        print("Attmap shape: ", attmap.shape)

        plt.imshow(attmap)
        plt.savefig('attmap_{}.png'.format(i))

        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
if __name__ == '__main__':
    #asr_test('./result/newertest/asr.cpt', './data/processed/index.tsv', './conf/test.yaml')
    asr_test('./result/malromur2017_default/asr.cpt', './processed_data/malromur2017/production_indexes/train_index_byxlen.tsv', 
	'./conf/malromur2017_default.yaml')
