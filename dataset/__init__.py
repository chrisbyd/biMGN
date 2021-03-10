from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomIdentitySampler
from .cuhk03 import CUHK03
from .dukemtmc import DukeMTMC
from .vehicleid import VehicleID
from .veri import VeRi
from .market1501 import Market1501
from opt import opt
import os
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True

__factory = {
    'cuhk03': CUHK03,
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'veri': VeRi,
    'vehicleid': VehicleID,
}

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img



class Data():
    __factory = {
        'cuhk03': CUHK03,
        'market': Market1501,
        'dukemtmc': DukeMTMC,
        'veri': VeRi,
        'vehicleid': VehicleID,
    }

    def __init__(self,opt):
        train_transform = transforms.Compose([
            transforms.Resize((opt.height, opt.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((opt.height, opt.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = self.__factory[opt.dataset_name](root = opt.data_path )
        self.trainset = ImageDataset(self.dataset.train, transform=train_transform)
        self.testset = ImageDataset(self.dataset.gallery, transform=test_transform)
        self.queryset = ImageDataset(self.dataset.query, transform= test_transform)
        self.num_classes = self.dataset.num_train_pids
        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomIdentitySampler(self.dataset.train, batch_size = opt.batch_size,
                                                                        num_instances=opt.num_instances),
                                                  batch_size=opt.batch_size, num_workers=16,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,
                                                  pin_memory=True)

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))



        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid,trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid