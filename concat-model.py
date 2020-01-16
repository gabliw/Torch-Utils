import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Multi-modal Concat Model for Pytorch
# 2LSTM Model

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size * 2
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out2 = nn.functional.relu(self.fc(out[:, -1, :]))
        drop_out = self.dropout(out2)
        out3 = nn.Softmax(dim=1)
        out4 = self.fc2(drop_out)

        # return out3(out4)
        return out2

# Concatenate Model
class ConcatModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, model1, model2):
        super(ConcatModel, self).__init__()
        #

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size * 2
        self.num_classes = num_classes

        # input_size, hidden_size, hidden_size2, num_layers, num_classes
        self.lstm_one = model1
        self.lstm_two = model2

        self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x1, x2):
        # Set initial hidden and cell states
        h0_1 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)

        h0_2 = torch.zeros(self.num_layers, x2.size(0), self.hidden_size).to(device)
        c0_2 = torch.zeros(self.num_layers, x2.size(0), self.hidden_size).to(device)

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out_1 = self.lstm_one(x1)

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out_2 = self.lstm_two(x2)

        prd_1 = out_1
        prd_2 = out_2
        cat_out = torch.cat((prd_1, prd_2), dim=1)
        # print(out_1.shape, out_2.shape, cat_out.shape, cat_out.shape)
        # Decode the hidden state of the last time step

        out2 = nn.functional.relu(self.fc(cat_out))
        drop_out = self.dropout(out2)
        out3 = nn.LogSoftmax(dim=1)
        out4 = self.fc2(drop_out)

        return out3(out4)

# Example Parameter
input_size = 64
hidden_size = 64
num_layer = 2
num_class = 3

# Model Init
lmodel1 = LSTM(input_size, hidden_size, num_layer, num_class)
lmodel2 = LSTM(input_size, hidden_size, num_layer, num_class)
cmodel = ConcatModel(input_size, hidden_size, num_layer, Model1=lmodel1, Model2=lmodel2)

# y_pred = cmodel(x1=dataset1, x2=dataset2)
# loss = criterion(y_pred, torch.max(labels, 1)[0])
# out_loss = loss.item()
# _, tr_pred = torch.max(y_pred.data, 1)