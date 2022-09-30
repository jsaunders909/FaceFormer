import torch
import torch.nn as nn
import torch.nn.functional as F


class FCResBlock(nn.Module):

    def __init__(self, n_features, activation=nn.LeakyReLU):
        super(FCResBlock, self).__init__()

        self.fc1 = nn.Linear(n_features, n_features)
        self.fc2 = nn.Linear(n_features, n_features)

        self.activation = activation()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(x)
        return self.activation(x + out)


class Conv1DResblock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Conv1DResblock, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, padding),
                               padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, padding),
                               padding_mode='reflect')
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, padding),
                               padding_mode='reflect')

    def forward(self, x):  # x has shape (N, T, C)
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = out + x
        return self.conv3(out)


class TemporalConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, h_channels, conv_kernels):
        super(TemporalConvNet, self).__init__()
        self.enc = nn.Sequential(*[nn.Linear(in_channels, h_channels),
                                   nn.LeakyReLU(),
                                   FCResBlock(h_channels)])

        self.net = []
        for layer in range(len(conv_kernels)):
            self.net.append(Conv1DResblock(h_channels, h_channels, kernel_size=conv_kernels[layer]))
        self.net = nn.Sequential(*self.net)

        self.dec = nn.Sequential(*[
            FCResBlock(h_channels),
            nn.LeakyReLU(),
            nn.Linear(h_channels, out_channels)
        ])

        self.net = nn.Sequential(*self.net)

    def forward(self, x):  # Here x has shape (N, (2)T, C_in) -> (N, T, C_out)

        x = self.enc(x)                            # (N, T, H)
        x = x.permute((0, 2, 1))[:, :, None, :]    # (N, H, 1, T)
        x = self.net(x)                            # (N, H, 1, T)
        x = x[:, :, 0, :].permute((0, 2, 1))       # (N, T, H)
        x = self.dec(x)                            # (N, T, C_out)
        print(x.shape)
        return x


class RNN(nn.Module):

    def __init__(self, in_channels, out_channels, h_channels, type='GRU', n_layers=2, bidirectional=True):
        super(RNN, self).__init__()
        self.enc = nn.Sequential(*[nn.Linear(in_channels, h_channels),
                                   nn.LeakyReLU(),
                                   FCResBlock(h_channels)])

        if type == 'GRU':
            self.net = nn.GRU(h_channels, h_channels, batch_first=True, bidirectional=bidirectional)
        elif type == 'LSTM':
            self.net = nn.LSTM(h_channels, h_channels, batch_first=True, bidirectional=bidirectional)

        step_down = 2 * h_channels if bidirectional else h_channels
        self.dec = nn.Sequential(*[
            nn.Linear(step_down, h_channels),
            FCResBlock(h_channels),
            nn.LeakyReLU(),
            nn.Linear(h_channels, out_channels)
        ])

    def forward(self, x):  # Here x has shape (N, (2)T, C_in) -> (N, T, C_out)

        x = self.enc(x)                            # (N, 2T, H)
        x, *_ = self.net(x)                            # (N, H, 1, 2T)
        x = self.dec(x)                            # (N, T, C_out)
        return x


class Discriminator(nn.Module):

    def __init__(self, in_features, h_dim, model_type='TCN', use_sigmoid=True):

        super(Discriminator, self).__init__()

        if model_type == 'TCN':
            self.model = TemporalConvNet(in_features, 1, h_dim, [7, 5, 3, 3])
        elif model_type == 'GRU':
            self.model = RNN(in_features, 1, h_dim, type='GRU', bidirectional=False)
        elif model_type == 'biGRU':
            self.model = RNN(in_features, 1, h_dim, type='GRU', bidirectional=True)
        elif model_type == 'LSTM':
            self.model = RNN(in_features, 1, h_dim, type='LSTM', bidirectional=False)
        elif model_type == 'biLSTM':
            self.model = RNN(in_features, 1, h_dim, type='LSTM', bidirectional=True)

        self.use_sigmoid = use_sigmoid

    def forward(self, x):

        x = self.model(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x