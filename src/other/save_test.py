import torch
import torch.nn as nn
import json
import os.path

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.mp_a = mp_a()
        self.mp_b = mp_b()

class mp_a(nn.Module):
    def __init__(self):
        super(mp_a, self).__init__()
        self.linear = nn.Linear(20, 20)

class mp_b(nn.Module):
    def __init__(self):
        super(mp_b, self).__init__()
        self.linear = nn.Linear(20, 10)


def sanity():
    '''
    Assuring myself that we can truly swap in/out modules
    '''
    torch.manual_seed(0)
    m = model()
    for param_tensor in m.state_dict():
        print(param_tensor, "\t", m.state_dict()[param_tensor].sum())
    
    print('----------------------')
    mp = mp_a()
    for param_tensor in mp.state_dict():
        print(param_tensor, "\t", mp.state_dict()[param_tensor].sum())
    print('----------------------')    
    m.mp_a = mp
    for param_tensor in m.state_dict():
        print(param_tensor, "\t", m.state_dict()[param_tensor].sum())


def save_model():
    m = model()
    for param_tensor in m.state_dict():
        print(param_tensor, "\t", m.state_dict()[param_tensor].sum())
    torch.save(m.state_dict(), './result/save_test/model.cpt')

def load_model():
    m = model()
    m.load_state_dict(torch.load('./result/save_test/model.cpt'))
    for param_tensor in m.state_dict():
        print(param_tensor, "\t", m.state_dict()[param_tensor].sum())

def json_test(model_name, step, loss):
    # first check if json file exists:
    json_path = './result/save_test/tracker.json' 
    if not os.path.isfile(json_path):
        data = {}
    else:
        data = json.load(open(json_path, 'r'))
    for key, item in data.items():
        for k, it in item.items():
            print(type(it))
    data[model_name] = {}
    data[model_name]['step'] = step
    data[model_name]['loss'] = loss

    json.dump(data, open(json_path, 'w'))

if __name__ == '__main__':
    json_test('qweqwe', 10, 3.2)    
