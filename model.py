from transformers import BertModel
import torch.nn as nn
import torch

class BertClassifier_LSTM(nn.Module):
    '''
    - pooled_output : (batch_size, bert_output_dim = 768)
    - sequence_output : (batch_size, seq_len = 512, bert_output_dim = 768)
    - lstm_out : (batch_size, 2 * hidden_dim)  
    - fc_out : (batch_size, output_dim = 2)
    
    Note: Apply softmax outside the classifier object.
    - output_probabilities = fc_out.argmax(dim = 1) 
    '''

    def __init__(self, BERT_MODEL_NAME, n_layers, bidirectional, hidden_dim, output_dim = 2, dropout=0.5):

        super(BertClassifier_LSTM, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.embedding_dim = self.bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.LSTM(self.embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_id, mask):
        with torch.no_grad():
            sequence_output, pooled_output = self.bert(input_ids = input_id, attention_mask = mask, return_dict=False)
        # sequence_output shape : (batch_size, num_tokens = 512, output_dim = 768)
        
        # print(f"\nSequence output shape : ({sequence_output.shape})")
        
        _, (hidden, cell) = self.rnn(sequence_output)
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])   # Shape: [batch size, hid dim]
        
        # Hidden shape : (batch_size, emb_dim = 256 * 2 = 512)
        
        #print(f"Hidden shape : ({hidden.shape})")
        
        output = self.fc(hidden)     
        # Shape : [batch size, out dim]
        #print(f"Output shape : ({output.shape})\n")
        
        return output

class BertClassifier_Linear(nn.Module):

    def __init__(self, BERT_MODEL_NAME, output_dim = 2, dropout=0.5):

        super(BertClassifier_Linear, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input_id, mask):
        with torch.no_grad():
            _, pooled_output = self.bert(input_ids = input_id, attention_mask = mask, return_dict = False)

        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output
        