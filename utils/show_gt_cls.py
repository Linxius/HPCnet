from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F
from GTnet.model import GtNetCls
from GTnet.getGtFeature import getGtFeature, gt_feature_len


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')


opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset(
    root='../../data/shapenet_core/',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=True)

classifier = GtNetCls(k=len(test_dataset.classes), gt_feature_len = gt_feature_len)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()


for i, data in enumerate(testdataloader, 0):
    points, target = data
    points, target = Variable(points), Variable(target[:, 0])
    points = points.transpose(2, 1)
    gtFeature = getGtFeature(points)
    points = torch.zeros(gtFeature.size())
    points = gtFeature
    points, target = points.cuda(), target.cuda()
    pred, _, _ = classifier(points)
    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), correct.item() / float(32)))
