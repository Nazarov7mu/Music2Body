# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class AudioToKeypointRNN(nn.Module):

    def __init__(self, options):
        super(AudioToKeypointRNN, self).__init__()

        # Instantiating the model
        self.init = None

        hidden_dim = options['hidden_dim']
        if options['trainable_init']:
            device = options['device']
            batch_sz = options['batch_size']
            # Create the trainable initial state
            h_init = \
                init.constant_(torch.empty(1, batch_sz, hidden_dim, device=device), 0.0)
            c_init = \
                init.constant_(torch.empty(1, batch_sz, hidden_dim, device=device), 0.0)
            h_init = Variable(h_init, requires_grad=True)
            c_init = Variable(c_init, requires_grad=True)
            self.init = (h_init, c_init)

        # Declare the model
        self.fc_before1 = nn.Linear(options['input_dim'], 64)
        #self.fc_before2 = nn.Linear(32, 64)
        
        self.lstm = nn.LSTM(64, hidden_dim, 1)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, 1)
        
        self.dropout = nn.Dropout(options['dropout'])
        self.fc_1 = nn.Linear(hidden_dim, options['output_dim'])
        #self.fc = nn.Linear(128, options['output_dim'])
        
        
        self.LSTMs = [self.lstm, self.lstm1]
        
        
        self.initialize()

    def initialize(self):
        # Initialize LSTM Weights and Biases
        for lstm in self.LSTMs:            
            for layer in lstm._all_weights:
                for param_name in layer:
                    if 'weight' in param_name:
                        weight = getattr(lstm, param_name)
                        init.xavier_normal_(weight.data)
                    else:
                        bias = getattr(lstm, param_name)
                        init.uniform_(bias.data, 0.25, 0.5)

        # Initialize FC
        #init.xavier_normal_(self.fc.weight.data)
        #init.constant_(self.fc.bias.data, 0)

    def forward(self, inputs):
        # perform the Forward pass of the model
        print(inputs.size())
        outfc1 = self.fc_before1(inputs)
        #outfc2 = self.fc_before2(outfc1)
        print(outfc1.size())
     
        output, (h_n, c_n) = self.lstm(outfc1, self.init)
        print(output.size())
        output_after_lstm1, _ = self.lstm1(output, self.init)
        print(output_after_lstm1.size())
        #output_after_lstm2, _ = self.lstm2(output_after_lstm1, self.init)
        
        output_after_lstm1 = output_after_lstm1.view(-1, output_after_lstm1.size()[-1])  # flatten before FC
        print(output_after_lstm1.size())
        dped_output = self.dropout(output_after_lstm1)
        predictions = self.fc_1(dped_output)
        print(predictions.size())
       
        #predictions = self.fc(fc1)
        return predictions
