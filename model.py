import torch
import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.05

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        # assume input shape 28x28x32 and we want output of 64 channels
        # so in_channels = 32 out_channels = 64
        super(DepthwiseSeparableConv, self).__init__()
        self.in_channels = in_channels
        self.depthwise = nn.Conv2d(
            1, 1, kernel_size, stride, padding, bias=False
        ) # 28x28x1 convolved with 3x3 to produce 28x28x1
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) # 28x28x32 convolved with 1x1 to produce 28x28x64
        self.relu_and_norm_and_dropout = nn.Sequential(
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(dropout_value)
        )

    def forward(self, x):
        # assume input shape 28x28x32 and we want output of 64 channels
        outputs = []
        ch_num = 0
        for ch_num in range(self.in_channels):
            x1 = x[:,ch_num,:,:]
            ch_num = ch_num + 1
            x1 = x1.view(x1.shape[0],1,x1.shape[1],x1.shape[2])
            out_depthwise = self.depthwise(x1)
            outputs.append(out_depthwise)
        stacked_output = torch.stack(outputs, dim=-1) # 28x28x32
        stacked_output = stacked_output.view(stacked_output.shape[0],stacked_output.shape[4],stacked_output.shape[2],stacked_output.shape[3])
        out = self.pointwise(stacked_output) # 28x28x64
        out = self.relu_and_norm_and_dropout(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            # output_size = 32 , receptive_field = 3

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            # output_size = 32 , receptive_field = 5

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            # output_size = 16 , receptive_field = 7

            DepthwiseSeparableConv(in_channels=64, out_channels=128)
            # output_size = 16 , receptive_field = 11
        )

        # TRANSITION BLOCK 1
        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 16 , receptive_field = 11

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            # output_size = 16 , receptive_field = 15

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            # output_size = 16 , receptive_field = 21

            DepthwiseSeparableConv(in_channels=64, out_channels=128)
            # output_size = 16 , receptive_field = 25
        )

        # TRANSITION BLOCK 2
        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 16 , receptive_field = 25

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            # output_size = 16 , receptive_field = 29

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            # output_size = 16 , receptive_field = 33

            DepthwiseSeparableConv(in_channels=64, out_channels=128)
            # output_size = 16 , receptive_field = 37
        )
            
        # TRANSITION BLOCK 3
        self.transition3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 16 , receptive_field = 37

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            # output_size = 16 , receptive_field = 41

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), dilation=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            # output_size = 14 , receptive_field = 49

            DepthwiseSeparableConv(in_channels=64, out_channels=128)
            # output_size = 14 , receptive_field = 57
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=14)
        ) # output_size = 1

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.transition1(x)
        x = self.convblock2(x)
        x = self.transition2(x)
        x = self.convblock3(x)
        x = self.transition3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  # Flatten after GAP
        x = self.fc(x)

        return x