import torch.nn as nn

class ConvNet(nn.Module):
    """
    Conv4 Backbone
    """
    def __init__(self, emb_size):
        super(ConvNet, self).__init__()
        # set size
        self.hidden = 128
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.1))
        # self.max = nn.MaxPool2d(kernel_size=2)
        # self.layer_second = nn.Sequential(nn.Linear(in_features=self.last_hidden * 2,
        #                                   out_features=self.emb_size, bias=True),
        #                                   nn.BatchNorm1d(self.emb_size))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.1))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                                  out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        self.ly4 = True
        self.ly3 = False
        self.ly2 = False


    def forward(self, input_data):
        out_1 = self.conv_1(input_data)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        output_data = self.layer4(out_3)
        out = self.layer_last(output_data.view(output_data.size(0), -1))
        # output_data0 = self.max(out_3)
        # out = []
        # out.append(self.layer_second(output_data0.view(output_data0.size(0), -1)))
        return out

    def stem_param(self):
        for name, param in self.named_parameters():
            if 'layer_last' in name:
                yield param
            if self.ly4:
                if 'layer4' in name:
                    yield param
            if self.ly3:
                if 'conv_3' in name:
                    yield param
            if self.ly2:
                if 'conv_2' in name:
                    yield param


    def stem_param_named(self):
        for name, param in self.named_parameters():
            if 'layer_last' in name:
                yield name, param
            if self.ly4:
                if 'layer4' in name:
                    yield name, param
            if self.ly3:
                if 'conv_3' in name:
                    yield name, param
            if self.ly2:
                if 'conv_2' in name:
                    yield name, param

def conv4():
    return ConvNet(emb_size=128)