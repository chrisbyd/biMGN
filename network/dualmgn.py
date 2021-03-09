import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck
  # change this depend on your dataset


class DualMGN(nn.Module):
    def __init__(self,num_classes):
        super(DualMGN, self).__init__()

        feats = 256
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4
        #
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))

        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 12))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 24))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 24))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 24))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 24))
        # self.maxpool_ho_p2 = nn.MaxPool2d(kernel_size=(24,24))
        # self.maxpool_ho_p3 = nn.MaxPool2d(kernel_size=(24,24))
        self.maxpool_hp2 = nn.MaxPool2d(kernel_size= (24,12))
        self.maxpool_hp3 = nn.MaxPool2d(kernel_size= (24,8))

        self.reduction_h = nn.Sequential(nn.Conv2d(2048,feats,1,bias= False),nn.BatchNorm2d(feats),nn.ReLU())
        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)
        self._init_reduction(self.reduction_h)

        self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_1 = nn.Linear(2048, num_classes)
        self.fc_id_2048_2 = nn.Linear(2048, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)
        # the classifier for the
        # self.fc_hid_2048_0 = nn.Linear(2048, num_classes)
        # self.fc_hid_2048_1 = nn.Linear(2048, num_classes)
        # self.fc_hid_2048_2 = nn.Linear(2048, num_classes)

        self.fc_hid_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_hid_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_hid_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_hid_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_hid_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

        # # initialize weight for horizontal
        # self._init_fc(self.fc_hid_2048_0)
        # self._init_fc(self.fc_hid_2048_1)
        # self._init_fc(self.fc_hid_2048_2)

        self._init_fc(self.fc_hid_256_1_0)
        self._init_fc(self.fc_hid_256_1_1)
        self._init_fc(self.fc_hid_256_2_0)
        self._init_fc(self.fc_hid_256_2_1)
        self._init_fc(self.fc_hid_256_2_2)


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x, for_eval =False):
        x = self.backbone(x)

        p1 = self.p1(x)  #[batch_size,2048,*,*]
        p2 = self.p2(x)  #[batch_size,2048,*,*]
        p3 = self.p3(x)  #[batch_size,2048,*,*]
        print("Pa has shape",p1.shape)

        zg_p1 = self.maxpool_zg_p1(p1) #[batch_size, 2048,1, *]
        zg_p2 = self.maxpool_zg_p2(p2)  #[batch_size, 2048,1, *]
        zg_p3 = self.maxpool_zg_p3(p3)   #[batch_size, 2048,1, *]
        # hg_p2 = self.maxpool_ho_p2(p2)
        # hg_p3 = self.maxpool_ho_p3(p3)

        zp2 = self.maxpool_zp2(p2) #[batch_size, 2048,2, *]
        hp2 = self.maxpool_hp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]  #[batch_size, 2048,1, *]
        z1_p2 = zp2[:, :, 1:2, :]  #[batch_size, 2048,1, *]
        h0_p2 = hp2[:,:,:,0:1]
        h1_p2 = hp2[:,:,:,1:2]

        zp3 = self.maxpool_zp3(p3) #[batch_size, 2048,3, *]
        hp3 = self.maxpool_hp3(p3)
        z0_p3 = zp3[:, :, 0:1, :] #[batch_size, 2048,1, *]
        z1_p3 = zp3[:, :, 1:2, :] #[batch_size, 2048,1, *]
        z2_p3 = zp3[:, :, 2:3, :] #[batch_size, 2048,1, *]
        h0_p3 = hp3[:,:,:,0:1]
        h1_p3 = hp3[:,:,:,1:2]
        h2_p3 = hp3[:,:,:,2:3]


        fg_p1 = F.avg_pool2d(zg_p1,zg_p1.size()[2:]).squeeze(dim=3).squeeze(dim=2) #[batch_size, 2048]
        fg_p2 = F.avg_pool2d(zg_p2,zg_p2.size()[2:]).squeeze(dim=3).squeeze(dim=2) #[batch_size, 2048]
        fg_p3 = F.avg_pool2d(zg_p3,zg_p3.size()[2:]).squeeze(dim=3).squeeze(dim=2) #[batch_size, 2048]
        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2) #[batch_size, 256]
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2) #[batch_size, 256]
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2) #[batch_size, 256]
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2) #[batch_size, 256]
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2) #[batch_size, 256]

        # fhg_p2 = F.avg_pool2d(hg_p2,hg_p2.size()[2:]).squeeze(dim=3).squeeze(dim=2) #[batch_size,2048]
        # fhg_p3 = F.avg_pool2d(hg_p3,hg_p3.size()[2:]).squeeze(dim=3).squeeze(dim=2)
        fh0_p2 = self.reduction_h(h0_p2).squeeze(dim=3).squeeze(dim=2)
        fh1_p2 = self.reduction_h(h1_p2).squeeze(dim=3).squeeze(dim=2)
        fh0_p3 = self.reduction_h(h0_p3).squeeze(dim=3).squeeze(dim=2)
        fh1_p3 = self.reduction_h(h1_p3).squeeze(dim=3).squeeze(dim=2)
        fh2_p3 = self.reduction_h(h2_p3).squeeze(dim=3).squeeze(dim=2)


        l_p1 = self.fc_id_2048_0(fg_p1) #[2batch_size, num_classes]
        l_p2 = self.fc_id_2048_1(fg_p2) #[batch_size, num_classes]
        l_p3 = self.fc_id_2048_2(fg_p3) #[batch_size, num_classes]

        l0_p2 = self.fc_id_256_1_0(f0_p2) #[batch_size, num_classes]
        l1_p2 = self.fc_id_256_1_1(f1_p2) #[batch_size, num_classes]
        l0_p3 = self.fc_id_256_2_0(f0_p3) #[batch_size, num_classes]
        l1_p3 = self.fc_id_256_2_1(f1_p3) #[batch_size, num_classes]
        l2_p3 = self.fc_id_256_2_2(f2_p3) #[batch_size, num_classes]

        # horizontal scores
        # lh_p1 = self.fc_hid_2048_0(fg_p1)  # [batch_size, num_classes]
        # lh_p2 = self.fc_hid_2048_1(fg_p2)  # [batch_size, num_classes]
        # lh_p3 = self.fc_hid_2048_2(fg_p3)  # [batch_size, num_classes]

        lh0_p2 = self.fc_hid_256_1_0(fh0_p2)  # [batch_size, num_classes]
        lh1_p2 = self.fc_hid_256_1_1(fh1_p2)  # [batch_size, num_classes]
        lh0_p3 = self.fc_hid_256_2_0(fh0_p3)  # [batch_size, num_classes]
        lh1_p3 = self.fc_hid_256_2_1(fh1_p3)  # [batch_size, num_classes]
        lh2_p3 = self.fc_hid_256_2_2(fh2_p3)  # [batch_size, num_classes]



        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3,fh0_p2,fh1_p2,fh0_p3,fh1_p3,fh2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3,l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3, lh0_p2, lh1_p2, lh0_p3, lh1_p3, lh2_p3
