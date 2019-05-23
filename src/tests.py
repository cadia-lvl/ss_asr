import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow
from ASRdataset import load_asr_dataset, prepare_x, prepare_y

from preprocess import load_wav, log_fbank


'''
    Dataset tests
'''

def plt_n_save(x, path):
    '''
    '''
    plt.figure()
    plt.subplot(2,1,1)
    librosa.display.specshow(x)
    plt.subplot(2,1,2)
    librosa.display.specshow(x)
    plt.savefig(path)

def specshow_test(index_path='./processed_data/malromur2017/2_5hour.tsv'):
    '''
    Sanity test for a batch size of 1 , add the dataset to the return of load_asr_dataset
    to test
    '''
    N_DIMS = 40
    WIN_SIZE = 25
    STRIDE = 10

    #audio_path = '/data/malromur2017/correct/is_is-landsbankinn4_06-2011-11-23T18:46:18.854654.wav'
    audio_path = '/data/malromur2017/correct/is_is-ec04-2011-09-21T16:55:02.951204.wav'
    y, sr = librosa.load(audio_path)
    ws=int(sr*0.001*WIN_SIZE)
    st= int(sr*0.001*STRIDE)
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_DIMS, n_fft=ws, hop_length=st)
    m = np.log(m + np.finfo(float).eps).astype('float32')
    print(m.shape)
    plt_n_save(m, 'spectro.png')

    sr, y = load_wav(audio_path)    
    m = log_fbank(y, sr)
    print(m.shape)
    plt_n_save(m, 'prepro.png')


    x = np.load('./processed_data/malromur2017/fbanks/is_is-marel4_02-2011-11-18T09:52:14.802145.npy')
    x = torch.from_numpy(x)
    x = x.view(1, 1, x.shape[0], x.shape[1])
    x, x_len = prepare_x(x)
    batch_t = max(x_len)
    x = x[:, :batch_t, :]
    x = x.view(x.shape[1], x.shape[2])
    print(x.shape)
    plt_n_save(x.cpu().numpy(), 'numpy.png')

    mapper , dataset, dataloader = load_asr_dataset(index_path, batch_size=1, text_only=False)
    batch_idx, (x, y) = next(enumerate(dataloader))
    x = torch.from_numpy(dataset.get_fbank(0))
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    x, x_lens = prepare_x(x)
    batch_t = max(x_lens)
    x = x[:, :batch_t, :]
    x = x.permute(0, 2, 1)
    #x = x[0,:,:].view(x.shape[2], x.shape[1]).cpu()
    print(x.shape)
    x = x.squeeze(0)
    plt_n_save(x.numpy(), 'data.png')


def simple_dataset_test(index_path='data/processed/index.tsv'):
    '''
    Sanity test for a batch size of 1 , add the dataset to the return of load_asr_dataset
    to test
    '''
    _, dataset, dataloader = load_asr_dataset(index_path, batch_size=1, text_only=True)

    for batch_idx, data in enumerate(dataloader):
        for i in range(data.shape[0]):
            print(batch_idx, dataset.decode(np.reshape(data[i,:,:], [data.shape[2]])))

def test_drop_func(index_path='data/processed/index.tsv'):
    batch_size = 2
    _, dataset, noised_dataloader = load_asr_dataset(index_path, 
        text_only=True, batch_size=batch_size, drop_rate=0.2)
    _, dataset, normal_dataloader = load_asr_dataset(index_path, 
        text_only=True, batch_size=batch_size, drop_rate=0.0)

    idx, normal_data  =  next(enumerate(normal_dataloader))
    normal_y, y_lens = prepare_y(normal_data)
    print(normal_y.shape)

    idx, noised_data  =  next(enumerate(noised_dataloader))
    noised_y, noised_y_lens = prepare_y(noised_data)
    print(noised_y.shape)

    for i in range(batch_size - 1):
        print("Normal padded:")
        print(dataset.decode(normal_data[i, :].view(-1)))
        print("Noisy padded:")
        print(dataset.decode(noised_data[i, :].view(-1)))



def simple_shape_check(index_path='data/processed/index.tsv'):
    '''
    Just for checking shapes of things
    '''
    dataloader = load_asr_dataset(index_path, batch_size=32)
    _ , data = next(enumerate(dataloader))
    
    print(data[0].shape)
    print(data[1].shape)

'''

'''
def test_simple_lm(index_path, lm_path, conf_path):
    from lm import LM
    import torch.nn.functional as F
    '''
    Performs a simple sanity LM test on a single sample, given some
    dataset.
    '''
    (mapper, dataset, dataloader) = load_asr_dataset(index_path, text_only=True)
   
    batch_idx, data = next(enumerate(dataloader))
    data = data.long()

    config = yaml.load(open(conf_path,'r'), Loader=yaml.FullLoader)

    lm = LM(mapper.get_dim(), **config['rnn_lm']['model_para'])
    lm.load_state_dict(torch.load(lm_path))
    
    lm_hidden = None
    corrects = 0
    
    for i in range(1, data.shape[2]):
        current_char = data[:, :, i]
        if i + 1 < data.shape[2]:
            next_char = data[:, :, i+1]

        lm_hidden, lm_out = lm(current_char, [1], lm_hidden)
        #print(F.softmax(lm_out.view(lm_out.shape[2])))
        #print(F.softmax(lm_out))
        prediction = dataset.idx2char(torch.argmax(lm_out).item())
        
        current_char = dataset.idx2char(data[0, 0, i].item())
        if i+1  < data.shape[2]: 
            next_char = dataset.idx2char(data[0, 0, i+1].item())
            x = ' '            
            if next_char == prediction:
                x = 'X'

            print('current: {0}, next: {1}, prediction: {2}, [{3}]'
                .format(current_char, next_char, prediction, x))

            if next_char == prediction: corrects += 1
    
    print('The model had accuracy of {}%'.format(100*corrects/data.shape[2]))


def check_yaml(path='./conf/asr_confs/default.yaml'):
    config = yaml.load(open(path,'r'), Loader=yaml.FullLoader)

    print(config)


if __name__  == '__main__':
    specshow_test()