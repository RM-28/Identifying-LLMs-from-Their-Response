import torch
import torch.nn as nn

class RNN(nn.Module):
    #initalize the model
    def __init__(self, input_size, output_size, hidden_size = 128, dropout = 0.5, bidirectional = False):
        super().__init__()
        #initialize the model parameters
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        #embedding layer to convert input tokens to dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        #bidirectional LSTM layer with dropout
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)


    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first = True, enforce_sorted = False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #keep only final state hT
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim = 1)
        else:
            hidden = hidden[-1,:,:]

        return self.fc(hidden)

if __name__ == "__main__":
    #test a sample to see if model runs
    text = torch.randint(0, 100, (32, 10))
    text_length = torch.tensor([10] * 32)
    for i in (False, True):
        model = RNN(input_size=100, hidden_size=128, output_size=4, dropout=0.5, bidirectional=i)
        output = model(text, text_length)
        print(f"Bidirectional = {i}, Output shape = {output.shape}")
