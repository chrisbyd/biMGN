import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib
import argparse
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
import torch
from torch.optim import lr_scheduler
from reid.evaluators import Evaluator, extract_features
from reid.loss import TripletLoss,WeightCE
import torch.nn as nn
#from network import MGN
from reid import models
from reid.trainers import MgnTrainer, PartTrainer, DualMgnTrainer
import os.path as osp
from reid.utils.data import transforms as T
from torchvision.transforms import Resize
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from torch.utils.data import DataLoader
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid import datasets
from reid.utils import make_dirs
from reid.utils.logging import Logger
import sys
from reid.utils.logger import setup_logger
import logging



def get_train_loader(name, data_dir, height, width, batch_size,num_instances):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.train # use all training image to train
    num_classes = dataset.num_train_pids
    transformer = T.Compose([
        Resize((height,width)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([height,width]),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability= 0.5, mean=[0.485, 0.456, 0.406])
     ])

    val_transformer = T.Compose(
        [
            Resize((height,width)),
            T.ToTensor(),
            normalizer
        ]
    )

    train_dataset = Preprocessor(train_set,transform= transformer)
    train_loader = DataLoader(
        train_dataset, batch_size= batch_size,
        sampler= RandomIdentitySampler(train_dataset,batch_size,num_instances),
        num_workers=4
    )
    test_dataset = Preprocessor(dataset.query + dataset.gallery, transform=  val_transformer)
    test_loader = DataLoader(
        test_dataset, batch_size= 128,
        num_workers=4

    )
    return train_loader , test_loader, dataset

class Main():
    def __init__(self, model, args):
        self.train_loader , self.test_loader, self.dataset_raw = get_train_loader(args.src_dataset,args.data_dir,args.height,
                                                                                  args.width,args.batch_size,args.num_instances)
        self.criterions = []
        self.criterions.append(TripletLoss(margin=args.margin))
        self.criterions.append(CrossEntropyLoss())
        self.model = model

        if args.arch == 'resnet50':
            self.trainer = PartTrainer(self.model,criterions= self.criterions)
        elif args.arch == 'dualmgn':
            self.trainer = DualMgnTrainer(self.model, criterions= self.criterions)
        else:
            self.trainer = MgnTrainer(self.model, criterions= self.criterions)
        self.evaluator = Evaluator(self.model,print_freq=args.print_freq)
        # self.model = model.to('cuda')

        # multi lr
        self.optimizer = torch.optim.SGD(model.module.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay)
        self.scheduler = WarmupMultiStepLR(self.optimizer, [50,100],0.1,0.01,10,'linear')


    def train_an_epoch(self,epoch, args,logger):

        self.model.train()
        self.trainer.train(epoch,self.train_loader,self.optimizer, args.print_freq, logger)
        self.scheduler.step()

    def evaluate(self, logger):
        top1 = self.evaluator.evaluate(self.test_loader,self.dataset_raw.query, self.dataset_raw.gallery,dataset_name= args.src_dataset, logger= logger)
        print("The top 1 accuracy is",top1)



def parse_args():
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data14
    parser.add_argument('--src_dataset', type=str, default='VeRi',                       choices= ['VeRi', 'VehicleID'])

    parser.add_argument('-b', '--batch_size', type=int, default= 400)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=384,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,default= 128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine_train', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num_instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default= 'resnet50',
                        choices= ['resnet50', 'mgn','dualmgn'])

    parser.add_argument('--mode', type= str, default= 'train')
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help="balancing parameter, default: 0.1")
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate of all parameters")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--num_split', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=30)
    parser.add_argument('--no_rerank', action='store_true', help="train without rerank")
    parser.add_argument('--dce_loss', action='store_true', help="train without rerank")
    # metric learning
    parser.add_argument('--dist_metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='./data/')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--checkpoint_path', type= str, default='./out/source_out')
    parser.add_argument('--load_dist', action='store_false', help='load pre-compute distance')
    parser.add_argument('--gpu_devices', default='0,1,2,3', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--checkpoint_interval', type= int, default= 30)
    parser.add_argument('--evaluate_interval', type= int, default= 2)
    parser.add_argument('--train_epoches', type= int, default=120)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger_name = '{}_{}_{}_{}_{}_{}_train'.format(args.arch,args.src_dataset,args.height,args.width,args.batch_size,args.lr)
    logger = setup_logger(logger_name,save_dir=args.checkpoint_path, if_train=True )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    model = models.create(args.arch, num_classes=576, num_split=args.num_split, cluster=args.dce_loss)
    model = nn.DataParallel(model).cuda()
    main = Main(model, args)

    if args.mode == 'train':

        for epoch in range(1, args.train_epoches + 1):
            print('start training epoch {}'.format(epoch))
            main.train_an_epoch(epoch= epoch, args= args, logger = logger)
            if epoch % args.evaluate_interval == 0:
                print('\nstart evaluating')
                main.evaluate(logger)

            if epoch % args.checkpoint_interval ==0:
                make_dirs(args.checkpoint_path)
                torch.save(model.module.state_dict(),
                           osp.join(args.checkpoint_path, args.arch + args.src_dataset + '_{}.pth'.format(epoch) ))

    if args.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(args.weight))
        main.evaluate()

    if args.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(args.weight))
        main.vis()
