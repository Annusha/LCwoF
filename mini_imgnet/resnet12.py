import torch.nn as nn
import torch
import torch.nn.functional as F
from mini_imgnet.dropblock import DropBlock
import copy



from utils.arg_parse import opt
from utils.logging_setup import logger
from mini_imgnet.resnet10 import resnet10
from mini_imgnet.conv4 import conv4


# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1, last_relu=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.last_relu = last_relu

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.last_relu:
            out = self.relu(out)

        out = self.maxpool(out)

        if self.last_relu:
            if self.drop_rate > 0:
                if self.drop_block == True:
                    feat_size = out.size()[2]
                    keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                    gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                    out = self.DropBlock(out, gamma=gamma)
                else:
                    out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, last_relu=True)
        if avg_pool:
            if opt.im_dim == 84:
                self.avgpool = nn.AvgPool2d(5, stride=1)
            if opt.im_dim == 224:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, last_relu=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, last_relu))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def stem_param(self):
        for name, param in self.named_parameters():
            if 'layer4' in name:
                yield param
            if opt.freeze_layer_from != -1:
                if opt.freeze_layer_from <= 3:
                    if 'layer3' in name:
                        yield param
                if opt.freeze_layer_from <= 2:
                    if 'layer2' in name:
                        yield param
                if opt.freeze_layer_from <= 1:
                    if 'layer1' in name:
                        yield param

    def stem_param_named(self):
        for name, param in self.named_parameters():
            if 'layer4' in name:
                yield name, param
            if opt.freeze_layer_from != -1:
                if opt.freeze_layer_from <= 3:
                    if 'layer3' in name:
                        yield name, param
                if opt.freeze_layer_from <= 2:
                    if 'layer2' in name:
                        yield name, param
                if opt.freeze_layer_from <= 1:
                    if 'layer1' in name:
                        yield name, param

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def Res12(keep_prob=1.0, avg_pool=True, **kwargs):
    """Constructs a ResNet-12 model.
    """
    if opt.resnet == 12:
        model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
        dim = 640
    elif opt.resnet == 10:
        model = resnet10()
        dim = 512
    elif opt.resnet == 4:
        model = conv4()
        dim=128
    else:
        raise KeyError
    return model, dim


class MLP(nn.Module):
    def __init__(self, checkpoint_init=None):
        super().__init__()

        self.encoder, self.i_dim = Res12()

        # self.i_dim = 640

        self.fc = nn.Linear(self.i_dim, opt.n_base_classes, bias=False)
        self.scale = None


        self.gfsv = True
        self.fc_novel = None
        self._state_dict = None
        self.checkpoint_init = copy.deepcopy(checkpoint_init)


    def fs_init(self):
        self.gfsv = False
        self.fc_novel = nn.Linear(self.i_dim, opt.n_way, bias=False)


    def update_checkpoint_init(self):

        self.checkpoint_init = copy.deepcopy(self.state_dict())

    def L2_weight_loss(self):
        loss = None
        for name, param in self.encoder.stem_param_named():
            if loss is None:
                loss = (self.checkpoint_init['encoder.' + name] - param).pow(2).sum()
            else:
                loss += (self.checkpoint_init['encoder.' + name] - param).pow(2).sum()

        return loss * opt.l2_weight


    def forward(self, input):
        x = input['data']
        output = {}
        x = x.to(device=opt.device, non_blocking=True)
        x = self.encoder(x)
        if self.gfsv:
            x = self.fc(x)
        else:
            x = self.fc_novel(x)


        output.update({'probs': x})
        return output

    def stem_param(self):
        for param in self.encoder.stem_param():
            yield param

    def fc_param(self):
        for param in self.fc.parameters():
            yield param
        for param in self.fc_novel.parameters():
            yield param

    def base_cl_param(self):
        for param in self.fc.parameters():
            yield param

    def novel_cl_param(self):
        for param in self.fc_novel.parameters():
            yield param


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss, self).__init__()
        if weight is None:
            self.weight = None
        else:
            self.weight = weight.to(opt.device)

    def forward(self, output, input, lymbda=1):
        probs = output['probs'].to(opt.device)
        target = input['label'].to(opt.device)
        if probs.size(0) > 1:
            target = target.squeeze()
        if self.weight is None:
            return lymbda * F.cross_entropy(probs, target)
        else:
            return F.cross_entropy(probs, target, weight=self.weight)


class DistillKL(nn.Module):
    """KL divergence for distillation
    from https://github.com/WangYueFt/rfs/blob/master/distill/criterion.py"""

    def __init__(self):
        super(DistillKL, self).__init__()
        self.T = opt.distill_T

    def forward(self, y_s, y_t):
        y_s = y_s['probs'].to(opt.device)
        y_t = y_t['probs'].to(opt.device)
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return opt.distill_scale * loss


def create_model(**kwargs):

    model = MLP()

    if torch.cuda.is_available():
        opt.device = 'cuda'
    else:
        opt.device = 'cpu'

    if opt.one_plus2_model.endswith('pth.tar'):
        if opt.device == 'cpu':
            checkpoint = torch.load(opt.one_plus2_model, map_location='cpu')['state_dict']
        else:
            checkpoint = torch.load(opt.one_plus2_model)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            if 'params' in checkpoint:
                checkpoint = checkpoint['params']
            # checkpoint = torch.load(opt.one_plus2_model)['params']
        if 'fc.bias' in checkpoint and not opt.bias_classifier:
            del checkpoint['fc.bias']

        a = model.load_state_dict(checkpoint, strict=True)
        logger.debug(a)



    loss = CrossEntropyLoss()

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.lr,
                                     weight_decay=opt.weight_decay)

    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay,
                                    nesterov=True
                                    )
    model.to(opt.device)
    logger.debug(str(model))
    for name, param in model.named_parameters():
        logger.debug('%s\n%s' % (str(name), str(param.norm())))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer
