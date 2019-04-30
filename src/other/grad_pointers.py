import torch
import torch.nn as nn


def gp_1():
    '''
        requires_grad and .detach()
    '''
    # create a tensor, requireding grad
    a = torch.randn(3, 3, requires_grad=True)
    print("a : \n", a)

    # create a new result without detaching
    b = a + a
    print("b = a + a :\n", b)

    # create a new result and detach a
    b = a.detach() + a.detach()
    print("b = a.detach() + a.detach() : \n", b)
    
    # Note that b requires grad if a is not detached
    # otherwise it does not.

    print("a after creating detached copies : \n", a)
    # Also note that a.detach() does not remove a's requirement
    # for grad.

    a.detach_()
    print("Detached a, in-place : \n", a)

def gp_2():
    '''
    Testing backward
    '''
    a = torch.randn(3, 3, requires_grad=True)
    print("A: \n", a)

    b = torch.randn(3, 3, requires_grad=True)

    c = torch.mean(a + b)
    
    print("The gradient input can only be a scalar value, here is c: ", c)
    
    # before calling c.backward(), a.grad is non existant
    print("The gradient d(c)/dx before calling c.backward(): \n", a.grad)
    c.backward()

    # after calling c.backward(), a.grad now exists
    print("The gradient d(c)/dx after calling c.backward(): \n", a.grad)


def gp_3():
    '''
    Computation graph:

                |> loss_1
    a -> b -> c |
                |> loss_2
    '''
    a = torch.ones(1, 4, requires_grad=True)
    b = a**2
    c = b*2
    loss_1 = c.mean()
    loss_2 = c.sum()

    # if we call loss_1.backward(), the part of the computation graph
    # to calculate d will be destroyed.
    loss_1.backward(retain_graph=True)

    print("Gradient of a after backpropping loss_1: \n" , a.grad)

    # and as a consequence, we cannot call loss_2.backward() since a part
    # of it's history has been destroyed
    loss_2.backward()

    print("Gradient of a after backpropping loss_2: \n", a.grad)


def gp_4():
    '''
    To achieve the same results as in the example above without 
    retaining the graph, we can just add together the losses 
    and then do the backprop w.r.t. the total loss
    '''
    a = torch.ones(1, 4, requires_grad=True)
    b = a**2
    c = b*2
    loss_1 = c.mean()
    loss_2 = c.sum()

    total_loss = loss_1 + loss_2
    total_loss.backward()

    print(a.grad)


def gp_5():
    '''
    Training parameters

    A simple linear regression model.
    Want to learn 2 parameters 
    
    y = ax + b
    '''
    import matplotlib.pyplot as plt

    a = 3 
    b = 5   


    # create 1K datapoints
    from torch.utils.data import DataLoader, Dataset
    
    class ds(Dataset):
        def __init__(self):
            super(ds, self).__init__()

            # generate 1000 random values in (0, 5) 
            self.x = torch.randn(5000)*5
            self.y = a*self.x + b

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return {'x': self.x[idx], 'y': self.y[idx]}

    dl = DataLoader(ds(), batch_size=10, shuffle=True, num_workers=2)

    model = nn.Linear(1, 1)

    loss_criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    '''
    Computational graph: 

    '''
    for epoch in range(50):
        for b_idx, data in enumerate(dl):
            model.zero_grad()

            x = data['x'].view(10, 1)
            y = data['y'].view(10, 1)

            y_predict = model(x)

            loss = loss_criterion(y_predict, y)

            loss.backward()
            optimizer.step()


def gp_6():
    import time 
    # We have a simple network B and a more complex
    # network A
    A = nn.Sequential(nn.Linear(30, 30), nn.Linear(30, 30), nn.Linear(30, 20))
    B = nn.Linear(20, 20)

    # The input to B is the output of A, but we 
    # only need to train B.
    optimizer = torch.optim.SGD(B.parameters(), lr=0.01, momentum=0.9)
    loss_criterion = nn.L1Loss()

    '''
    We could just run autograd on the complete computational 
    graph, i.e. [input -> A -> B -> output]. The optimizer will
    only update the parameters of B
    '''
    start = time.time()
    for i in range(10_000):
        B.zero_grad()

        sample = torch.randn(32, 30) # does not require grad (leave node)
        target = torch.ones(32, 20)
        
        A_out = A(sample) # does require grad
        B_out = B(A_out)
        loss =  loss_criterion(B_out, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print('Loss: %.4f' % loss.item())
    print('Total time: {}'.format(time.time() - start))

    '''
    But we still have to do gradient calculations on the A network, which
    would be pointless since we are not updating the parameters of the A 
    network. To avoid that, we can "detach" the output of A to save time.
    '''
    A = nn.Sequential(nn.Linear(30, 30), nn.Linear(30, 30), nn.Linear(30, 20))
    B = nn.Linear(20, 20)
    optimizer = torch.optim.SGD(B.parameters(), lr=0.01, momentum=0.9)
    start = time.time()
    for i in range(10_000):
        B.zero_grad()

        sample = torch.randn(32, 30) # does not require grad (leave node)
        target = torch.ones(32, 20)
        
        A_out = A(sample).detach() # does NOT require grad
        B_out = B(A_out)
        loss =  loss_criterion(B_out, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print('Loss: %.4f' % loss.item())
    print('Total time: {}'.format(time.time() - start))


def gp_7():
    '''
    Same as gp_6, but now we need to train A, not B
    '''

    import time 
    # We have a simple network A and a more complex
    # network B
    A = nn.Linear(30, 20)
    B = nn.Sequential(nn.Linear(20, 50), nn.Linear(50, 30), nn.Linear(30, 20))

    # The input to B is the output of A, but we 
    # only need to train A.
    optimizer = torch.optim.SGD(A.parameters(), lr=0.01, momentum=0.9)
    loss_criterion = nn.L1Loss()

    '''
    We could just run autograd on the complete computational 
    graph, i.e. [input -> A -> B -> output]. The optimizer will
    only update the parameters of B
    '''
    start = time.time()
    for i in range(10_000):
        A.zero_grad()

        sample = torch.randn(32, 30) # does not require grad (leave node)
        target = torch.ones(32, 20)
        
        A_out = A(sample) # does require grad
        B_out = B(A_out)
        loss =  loss_criterion(B_out, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print('Loss: %.4f' % loss.item())
    print(time.time() - start)
    
    '''
    But we still have to do gradient calculations on the A network, which
    would be pointless since we are not updating the parameters of the A 
    network. To avoid that, we can "detach" the output of A to save time.
    '''

    start = time.time()
    for i in range(10_000):
        B.zero_grad()

        sample = torch.randn(32, 30) # does not require grad (leave node)
        target = torch.ones(32, 20)
        
        A_out = A(sample) # does NOT require grad
        B_out = B(A_out).detach()
        loss =  loss_criterion(B_out, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print('Loss: %.4f' % loss.item())
    print(time.time() - start)


if __name__ == '__main__':
    gp_6()