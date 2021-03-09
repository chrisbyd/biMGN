import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network import MGN,DualMGN
from loss import mgnLoss, res50Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
import torch.nn as nn
from utils.logger import setup_logger
from tqdm import tqdm
import logging
import os.path as osp
from utils.tool import results_to_excel


class Main():
    def __init__(self, model, loss, data, logger = None):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset
        self.model = model
        # self.model = model.to('cuda')
        self.loss = loss
        self.logger = logger
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):

        self.scheduler.step()

        self.model.train()
        for batch, (inputs, labels) in tqdm(enumerate(self.train_loader)):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(self,epoch = 0):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)


        r, m_ap = rank(dist)

        self.logger.info('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
        results = [item for item in r[:20]] + [m_ap]
        results_to_excel(results, opt.arch + '_reranking', opt.dataset_name, epoch)
        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)
        results = [item for item in r[:20]] + [m_ap]
        results_to_excel(results, opt.arch, opt.dataset_name)

        self.logger.info('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        # # Remove junk images
        # junk_index = np.argwhere(gallery_label == -1)
        # mask = np.in1d(index, junk_index, invert=True)
        # index = index[mask]

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show.png")
        print('result saved to show.png')


if __name__ == '__main__':
    number_of_classes = {
        'VehicleID' : 13164 ,
        'VeRi' : 576 ,
        'Market' : 751 ,
        'cuhk02' : 256 ,
        'DukeMTMC' :756
    }
    data = Data(opt=opt)
    num_classes = number_of_classes[opt.dataset_name]
    if opt.arch == 'mgn':
        model = MGN(num_classes = num_classes)
    elif opt.arch == 'dualmgn':
        model = DualMGN(num_classes = num_classes)
    elif opt.arch == 'res50':
        pass
    else:
        raise NotImplementedError('The architecture is not implemented')

    model = nn.DataParallel(model).cuda()
    logger_name = '{}_{}_{}_{}_{}_train'.format(opt.arch, opt.dataset_name, opt.height, opt.width, opt.lr)
    logger = setup_logger(logger_name, save_dir=opt.log_path, if_train=True)

    if opt.arch in ['mgn','dualmgn']:
        loss = mgnLoss(logger)
    elif opt.arch in ['res50']:
        loss = res50Loss(logger)
    else:
        raise NotImplementedError("The backbone is not supported")

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
    main = Main(model, loss, data, logger)


    if opt.mode == 'train':

        for epoch in tqdm(range(1, opt.epoch + 1)):
            print('\nepoch', epoch)
            main.train()
            if epoch % opt.test_interval == 0:
                print('\nstart evaluate')
                main.evaluate(epoch = epoch)
                model_save_path = osp.join(opt.checkpoint_path, opt.arch)
                os.makedirs(model_save_path, exist_ok= True)
                torch.save(model.state_dict(), osp.join(model_save_path,'model_{}.pth'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()
