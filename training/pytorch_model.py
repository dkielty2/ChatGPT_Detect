
import torch
from torch import nn

class Classifier(nn.Module):
    
    
    def __init__(self, N_text_layers=1, N_text_in = 300, N_text_out=128, use_LSTM=False, N_metrics=5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.useLSTM=use_LSTM
        # RNN or LSTM part
        if use_LSTM:
            self.text_read = nn.LSTM(input_size= N_text_in, hidden_size=N_text_out, num_layers=N_text_layers)
        else:
            self.text_read = nn.RNN(input_size= N_text_in, hidden_size=N_text_out, num_layers=N_text_layers)
        # linear NN part
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N_text_out + N_metrics, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2), # last layer is outputting probabilities
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, metrics):
        #x = self.flatten(x)
        
        # do the RNN or LSTM part
        # TODO: look into adding attention mechanism
        text_out = self.text_read(x)
        
        # concatenate
        if self.useLSTM:
            x2 = torch.cat((text_out[1][0][0], metrics), axis=1)
        else:
            x2 = torch.cat((text_out[1][0], metrics),axis=1)
        # text_out[1] is the final hidden state for each element in the batch.
        
        # put both the metric array 
        logits = self.linear_relu_stack(x2)
        
        # softmax to return probabilities
        return self.softmax(logits)