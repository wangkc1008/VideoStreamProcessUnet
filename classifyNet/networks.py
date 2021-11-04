from torch import nn
import torch
from process_video_only import config
from collections import OrderedDict
from torchvision import models
config = config.HTConfig


def Network_Output(network):
    # =============================导入网络模型=================
    net = network
    net = net.cuda()

    # 建立优化
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.LR, weight_decay=config.Weight_Decay)
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.LR, weight_decay=config.Weight_Decay)
    optimizer3 = torch.optim.Adam(net.parameters(), lr=config.LR, weight_decay=config.Weight_Decay)


    # 不断调整学习率
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode="min", factor=config.LR_Gamma,
                                                           patience=config.LR_Patience, min_lr=config.LR_Min)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode="min", factor=config.LR_Gamma,
                                                           patience=config.LR_Patience, min_lr=config.LR_Min)
    scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode="min", factor=config.LR_Gamma,
                                                           patience=config.LR_Patience, min_lr=config.LR_Min)
    # 建立损失
    loss_func1 = nn.CrossEntropyLoss(weight=torch.tensor([1., 3., 2.])).cuda()   # 三分类损失函数张量weight长度为3 weight用于设置每个类别权重，适用于训练集不平衡
    loss_func2 = nn.MSELoss().cuda()

    loss_func = [loss_func1, loss_func2]
    optimizer = [optimizer1, optimizer2, optimizer3]
    scheduler = [scheduler1, scheduler2, scheduler3]
    return net, optimizer, loss_func, scheduler


def Baseline_Network(name, net_pretrained=-1, model_path=0):
    # ====================== VGG16 ======================
    if("vgg16" == name):
        # 修改VGG
        VggNet = models.vgg13(pretrained=True)
        VggNet.features[0] = nn.Conv2d(config.pic_select, 64, kernel_size=7, stride=2, padding=3, bias=False)
        VggNet.classifier[-1] = nn.Linear(4096, config.label_class, bias=True)

        # 参数初始化
        for m in VggNet.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        if(net_pretrained >= 0):
                VggNet.load_state_dict(torch.load("./models/vgg16/Beta" + str(config.beta_version) + "_fold" + str(net_pretrained) + ".pkl"))
        return VggNet



    # ====================== denseNet121 ======================
    elif("denseNet" == name):
        # 修改DenseNet
        DenseNet = models.densenet121(pretrained=True)
        DenseNet.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(config.pic_select, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        DenseNet.classifier = nn.Linear(64, config.label_class)
        if (net_pretrained >= 0):
            DenseNet.load_state_dict(torch.load("./models/denseNet/Beta" + str(config.beta_version) + "_fold" + str(net_pretrained) + ".pkl"))

        return DenseNet

    # ====================== resnet50 ======================
    elif("resnet50" == name):
        # 修改ResNet
        ResNet = models.resnet50(pretrained=True)

        ResNet.conv1 = nn.Conv2d(config.pic_select, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        features = ResNet.fc.in_features
        ResNet.fc = nn.Linear(features, config.label_class)

        if (net_pretrained >= 0):
            if(model_path==0):
                ResNet.load_state_dict(torch.load("./models/resnet50/Beta" + str(config.beta_version) + "_fold" + str(net_pretrained) + ".pkl"))
            else:
                ResNet.load_state_dict(torch.load(model_path))

        return ResNet