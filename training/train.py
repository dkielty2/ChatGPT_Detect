
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])
base_dir = '../'

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print(mp.cpu_count(),' CPUs available')

from pytorch_model import Classifier

# define model
model = Classifier(use_LSTM=True,N_metrics=5)
model = model.to(device)# put it on gpu

# define loss function
loss = nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#, momentum=0.9)

# from extrernal notebook, found which essays had atleast 1000 words
#with open('human_wc.npy','rb') as f:
#    wc = np.load(f)


def load_data(tensor_path='data/train_human_tensor.pt', metrics_path = 'data/train_human_metrics.csv', N_metrics = 5):
    # load tensor forms of the text
    list_of_tensors = torch.load(tensor_path)
    
    # extract various text metrics
    df = pd.read_csv(metrics_path)
    metrics_cols = ['met%i'%(i+1) for i in range(N_metrics)]
    arr =  df[metrics_cols].values
    
    # normalize the burstiness metric
    arr[:,0] = (np.log10(arr[:,0]) - 2.3) / 2.3
    
    return list_of_tensors, arr


def get_data():
    torch.cuda.empty_cache()
    # load human 
    tensors1,met1 = load_data(tensor_path=base_dir+'data/train_human_tensor.pt', metrics_path = base_dir+'data/train_human_metrics.csv', N_metrics = 5)
    # first col is GPT, second is human
    y1 = torch.cat((torch.zeros((1,len(tensors1)) ), torch.ones((1,len(tensors1)) ))).T
    
    # I want to remove a particular problematic essay that has a large wordcount
    #mask = (wc<=500)
    #met1 = met1[mask]
    #y1 = y1[mask]
    #tensors1 = [t.to(device) for t,m in zip(tensors1,mask) if m]
    tensors1 = [t.to(device) for t in tensors1]
    
    # load GPT
    tensors2,met2 = load_data(tensor_path=base_dir+'data/train_GPT_tensor.pt', metrics_path = base_dir+'data/train_GPT_metrics.csv', N_metrics = 5)
    # first col is GPT, second is human
    y2 = torch.cat((torch.ones((1,len(tensors2)) ), torch.zeros((1,len(tensors2)) ))).T
    
    # there' s particular essay that's giving me an error so I want to remove it
    #mask = ~np.isin(np.arange(len(y2)), [20993-len(y1)-1])
    #print(np.where(~mask))
    #met2 = met2[mask]
    #y2 = y2[mask]
    #tensors2 = [t.to(device) for t,m in zip(tensors2,mask) if m]
    tensors2 = [t.to(device) for t in tensors2]
    
    # text
    tensors = tensors1+tensors2
    
    # metrics
    met = np.concatenate((met1,met2), axis=0)
    met_tensor =  torch.from_numpy(met).float().to(device)
    
    # labels
    y = torch.cat((y1,y2), axis=0).to(device)
    
    return (tensors, met_tensor), y

def run_model(tensor_list, met_tensor,inds):
    """
    for given data and indices, evaluate the model
    """
    # gotta do some padding to put everything in batches    
    x_text = pad_sequence([tensor_list[ind] for ind in inds])
    x_text.requires_grad = True
    #x_text = x_text.to(device)
    # extract the normalized metrics
    x_met = met_tensor[inds]# torch.from_numpy(met_arr[inds]).float()
    x_met.requires_grad = True
    #x_met = x_met.to(device)

    # calculate the model predictions
    y_probs = model(x_text,x_met)
    return y_probs

def train_epoch(tensor_list, met_tensor, y, train_inds, batch_size = 50):
    """
    function to do 1 epoch of training over the dataset
    input:
        tensor list: list of the embedded text tensors
        met arr: tensor of the metrics for the corresponding text
        y: true classifications
    output:
        batch loss: loss for each batch of training
    
    """
    batch_loss =[]
    batch_num = 0
    while batch_num*batch_size < len(train_inds):
        #while batch_num*batch_size < 100:

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # define batch indicies
        i0 = batch_num*batch_size
        if i0+batch_size < len(train_inds):
            i1 = i0+batch_size
        else:
            i1 =len(train_inds)
        
        batch_inds = train_inds[i0:i1]
        #print(batch_num, (i0,i1), batch_inds)
        
        y_true = y[batch_inds]
        #print(y_true)
        
        # evaluate the inputs
        y_probs = run_model(tensor_list, met_tensor, batch_inds)

        # calculate the loss
        L = loss(y_probs, y_true)
        batch_loss.append(L.item())
        # training
        L.backward() # calculates the backwards gradients
        optimizer.step()# does a step
        
        # try to free up some memory please
        torch.cuda.empty_cache()
        if batch_num%100 == 0:
            print('\t did batch %i'%batch_num)

        batch_num+=1
    return np.array(batch_loss)

def test_pred_loss(tensor_list, met_tensor,y, test_inds):
    """
    The prediction arr is too damn big, 
    so we'll just loop thru it
    """
    #y_pred = run_model(tensor_list, metrics,test_inds)
    y_preds = []
    for ind in test_inds:
        x_text = tensor_list[ind].reshape(-1,1,300)
        x_met = met_tensor[ind].reshape(1,5)
        y_prob = model(x_text,x_met)
        y_preds.append(y_prob)
        torch.cuda.empty_cache()
    L = loss(torch.cat(y_preds, axis=0),y[test_inds])
    return L.item()

if __name__=='__main__':
    seed = 6287 # for train test splitting
    num_epochs = 15
    batch_size = 5
    # there are some essays with a lot of words, which takes up a lot of memory
    
    # load data
    (tensor_list, metrics),y = get_data()
    print('Loaded Data, %i essays'%len(tensor_list))
    
    # do train test split
    N_essays = len(tensor_list)
    inds = np.arange(N_essays)
    train_inds, test_inds = train_test_split(inds, test_size=0.2, random_state=seed, shuffle=True)
    
    
    epoch_loss = []
    for i in range(num_epochs):
        print('Starting epoch %i'%i)
        
        # do a whole epoch of training
        batch_loss = train_epoch(tensor_list, metrics, y, train_inds, batch_size = batch_size)
        # save the trained model weights at each epoch
        print('Saving model weights')
        model_weights_path = base_dir+'training/model_weights_epoch%i_seed%s_batch%i.pt'%(i, seed, batch_size)
        torch.save(model.state_dict(), model_weights_path)
        # get prediction from test set (test set is too damn large)
        print('Getting test set loss')
        test_loss = test_pred_loss(tensor_list, metrics, y, test_inds)
        
        # save epoch loss
        epoch_loss.append([np.mean(batch_loss), test_loss])
        #epoch_loss.append(batch_loss)

        print('Done training epoch %i'%i)
    
    # save the loss arrays for plotting
    with open(base_dir+'training/train_loss_%iepochs_seed%s_batch%i.npy'%(num_epochs,seed,batch_size), 'wb') as f:
        np.save(f, np.array(epoch_loss))
        
    print('Done.')