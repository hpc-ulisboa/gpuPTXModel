import torch

class multiNN2(torch.nn.Module):
    def __init__(self, input_size, output_dim, hidden_sizes, dropout_prob):
        super(multiNN2, self).__init__()
        self.fc_in = torch.nn.Linear(input_size, hidden_sizes[0])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_f, out_f) for in_f, out_f in zip(hidden_sizes, hidden_sizes[1:])])
        self.act_relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.fc_out = torch.nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.act_relu(x)
        x = self.dropout(x)
        for hidden_layer in self.layers:
            x = hidden_layer(x)
            x = self.act_relu(x)
            x = self.dropout(x)
        out = self.fc_out(x)
        return out

class EncoderLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout_prob, batch_size, device):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.hidden  = self.initHidden(batch_size)

        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout_prob)

    def forward(self, input, batch_size):
        embeddings = self.embed(input).view(1, batch_size,  -1)
        out, self.hidden = self.lstm(embeddings, self.hidden)
        return out

    def initHidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        return (h0, c0)
