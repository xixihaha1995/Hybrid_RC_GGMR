import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler



def preprocess(data_dir,lookback, test_portion, batch_size, drop_rc, drop_water):
    '''
    The scaler objects will be stored in this dictionary so that our output test data from
    the model can be re-scaled during evaluation
    '''
    label_scalers = {}
    train_x = []
    train_y = []
    test_x = {}
    test_y = {}

    for file in os.listdir(data_dir):
        # Skipping the files we're not using
        if file != "case_arr_gru.csv":
            continue
        # Store csv file in a Pandas DataFrame
        df = pd.read_csv('{}/{}'.format(data_dir, file), parse_dates=[0])
        # Processing the time data into suitable input formats
        # df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%MM')
        if drop_water:
            df = df.drop("vfr_water", axis=1)

        if drop_rc:
            df = df.drop("rc_y", axis=1)
        df.Timestamp = pd.to_datetime(df.Timestamp)
        df['minute'] = df.apply(lambda x: x['Timestamp'].minute, axis=1)
        df['hour'] = df.apply(lambda x: x['Timestamp'].hour, axis=1)
        df['dayofweek'] = df.apply(lambda x: x['Timestamp'].dayofweek, axis=1)
        df = df.sort_values("Timestamp").drop("Timestamp", axis=1)

        # Scaling the input data
        sc = MinMaxScaler()
        label_sc = MinMaxScaler()
        data = sc.fit_transform(df.values)
        # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
        label_sc.fit(df.iloc[:, 0].values.reshape(-1, 1))
        label_scalers[file] = label_sc

        # Define lookback period and split inputs/labels
        inputs = np.zeros((len(data) - lookback, lookback, df.shape[1]))
        labels = np.zeros(len(data) - lookback)

        for i in range(lookback, len(data)):
            inputs[i - lookback] = data[i - lookback:i]
            labels[i - lookback] = data[i, 0]
        inputs = inputs.reshape(-1, lookback, df.shape[1])
        labels = labels.reshape(-1, 1)

        # Split data into train/test portions and combining all data from different files into a single array
        # test_portion = int(test_frac * len(inputs))
        if len(train_x) == 0:
            train_x = inputs[:-test_portion]
            train_y = labels[:-test_portion]
        else:
            train_x = np.concatenate((train_x, inputs[:-test_portion]))
            train_y = np.concatenate((train_y, labels[:-test_portion]))
        test_x[file] = (inputs[-test_portion:])
        test_y[file] = (labels[-test_portion:])

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return label_scalers, train_data, train_loader, test_x, test_y



def train(train_loader, learn_rate, batch_size, hidden_dim, EPOCHS, device, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.perf_counter()
        h = model.init_hidden(batch_size, device)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
        current_time = time.perf_counter()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def evaluate(model, test_x, test_y, label_scalers, device):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h_0 = model.init_hidden(inp.shape[0], device)
        '''
        The above h_0 is only for GRU? 
        Yes. All the layers' parameters/variables is not required. 
        Actually, GRU doesn't require h_) either. If h_0 is not required, default as zeros.
        '''
        out, h = model(inp.to(device).float(), h_0)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.perf_counter() - start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
    print("sMAPE: {}%".format(sMAPE * 100))
    return outputs, targets, sMAPE


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim) #fully connected layer
        self.relu = nn.ReLU()

    def forward(self, x, h_0):
        '''Define the layer structure of the network'''
        out, h_n = self.gru(x, h_0)
        out = self.fc(self.relu(out[:, -1]))
        return out, h_n

    def init_hidden(self, batch_size, device):
        '''
        self.parameters() include all "parameters of the network"
        in this case, it includes GRU layer, Linear layer.
        And it doesn't include ReLU variables, since ReLu don't have variables
        next(something) = something[0]
        '''
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

