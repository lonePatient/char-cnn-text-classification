#encoding:utf-8
import torch.nn as nn

class CharacterCNN(nn.Module):
    def __init__(self,num_classes,in_channels,max_len_seq):
        super(CharacterCNN, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels = in_channels,
                                            out_channels = 256,
                                            kernel_size=7,
                                            padding=0),
                                           nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=3))

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=256,
                                            out_channels = 256,
                                            kernel_size=7,
                                            padding=0),
                                           nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=3))

        self.conv3 = nn.Sequential(nn.Conv1d(in_channels = 256,
                                             out_channels = 256,
                                             kernel_size=3,
                                             padding=0),
                                             nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv1d(in_channels = 256,
                                             out_channels = 256,
                                             kernel_size=3,
                                             padding=0),
                                             nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv1d(in_channels = 256,
                                             out_channels = 256,
                                             kernel_size=3,
                                             padding=0),
                                             nn.ReLU())

        self.conv6 = nn.Sequential(nn.Conv1d(in_channels=256,
                                            out_channels = 256,
                                            kernel_size=3,
                                            padding=0),
                                           nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=3))
        # 计算conv层输出的大小
        conv_dim = int((max_len_seq - 96) / 27 * 256)
        self.fc1 = nn.Sequential(nn.Linear(conv_dim,1024),nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(1024,1024),nn.Dropout(0.5))
        self.fc3 = nn.Linear(1024,self.num_classes)
        self.create_weights(mean=0.0, std=0.05)

    def create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        print(input)
        # 因为一维卷积是在最后维度上扫的，
        output = input.transpose(1, 2)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

