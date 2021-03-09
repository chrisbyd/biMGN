import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="Market-1501-v15.09.15",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--dataset_name',
                    default="Market-1501",
                    help='the name of the dataset')
parser.add_argument('--height',
                    default="384",
                    type= int,
                    help='the height of the input image')

parser.add_argument('--width',
                    default="128",
                    type= int,
                    help='the width of the input image')

parser.add_argument('--log_path',
                    default="./log",
                    help='the width of the input image')

parser.add_argument('--checkpoint_path',
                    default="./checkpoint",
                    help='the width of the input image')

parser.add_argument('--arch',
                    default='mgn', choices=['mgn', 'dualmgn', 'res50'],
                    help='the backbone of the model')


parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')


parser.add_argument('--weight',
                    default='weights/model.pt',

                    help='load weights ')

parser.add_argument('--epoch',
                    default=500,
                    type= int,
                    help='number of epoch to train')

parser.add_argument('--test_interval',
                    default=10,
                    type= int,
                    help='test performance every ..epoches')

parser.add_argument('--lr',
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[320, 380],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=2,
                    type = int,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=1,
                    type = int,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=8,
                    help='the batch size for test')

parser.add_argument("--gpu_devices",
                    default='0,1',
                    help='the visible gpu devices for pytorch')

opt = parser.parse_args()
