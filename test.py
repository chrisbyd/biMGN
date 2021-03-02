from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import numpy as np
import sys
sys.path.append('.')
import pickle
import torch
from torch import nn
from torch.nn import init
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import TenCrop, Lambda, Resize
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss,FocalLoss
from reid.trainers import Trainer, FinedTrainer, FinedTrainer2
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler,RandomIdentitySelfSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from collections import defaultdict
from sklearn.cluster import DBSCAN,AffinityPropagation
from reid.rerank import *
from reid.utils.logger import setup_logger
import numpy as np
import logging
# from reid.rerank_plain import *



def get_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val= 0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.train # use all training image to train
    num_classes = dataset.num_train_pids

    transformer = T.Compose([
        Resize((height,width)),
        T.ToTensor(),
        normalizer,
    ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root= None, transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, test_loader,


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    logger_name = '{}_{}_{}_{}_{}_{}_train'.format(args.arch, args.src_dataset, args.trg_dataset, args.height,
                                                      args.width, args.batch_size)
    logger = setup_logger(logger_name, save_dir=args.test_dir, if_train=True)

    # Create data loaders

    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
            (256, 128)

    # get_target_data
    tgt_dataset, num_classes, test_loader = \
        get_data(args.trg_dataset, args.data_dir, args.height,
                  args.width, args.batch_size, args.workers)
    model = models.create(args.arch,num_classes = 576)
    logger.info("Start loading checkpoint trained on another model...")
    checkpoint = torch.load(args.resume)
    print(args.test_iteration)
    if not args.test_iteration:
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint['state_dict'],strict= False)
    model = nn.DataParallel(model).cuda()
    # Evaluator
    evaluator = Evaluator(model, print_freq=args.print_freq)
    logger.info("Test with the original model trained on the target domain:")
    cmc,mAp = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery, dataset_name=args.trg_dataset,
                                   logger=logger)
    logger.info("The cmc curve from 1-20 is")
    logger.info("1:{},2:{},4:{},5:{},6:{},8:{},10:{},12:{},14:{},16:{},18:{},20:{}".format(cmc[0],cmc[1],cmc[3],cmc[4],cmc[5],cmc[7],cmc[9],cmc[11],cmc[13],cmc[15],

                                                                                      cmc[17],cmc[19]))
    logger.info("The map is {}".format(mAp))

def parse_args():
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data14
    parser.add_argument('--src_dataset', type=str, default='VeRi',choices= ['VeRi', 'VehicleID'])
    parser.add_argument('--trg_dataset', type=str, default='VehicleID',choices= ['VeRi', 'VehicleID'])

    parser.add_argument('--test_iteration',default= False, action= 'store_true')
    parser.add_argument('-b', '--batch_size', type=int, default= 5)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--test_dir', type=str, default= './out/test_out')
    parser.add_argument('--height', type=int, default=384,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,default= 384,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    # model
    parser.add_argument('-a', '--arch', type=str, default= 'dualmgn',
                        choices= ['resnet50', 'mgn','dualmgn'])

    parser.add_argument('--gpu_devices', default='0,1,2,3', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--resume', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='./out/target_out/checkpoint/')
    parser.add_argument('--num_split', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='./data/')
    parser.add_argument('--print_freq', type=int, default=20)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    main = main (args)
