import numpy as np
import torch

gt_feature_len = 4

def getGtFeature(points, geo_template = 0, radius=1):
    #temp_len = geo_template
    # points: torch.Size([32, 3, 2500])
    temp_len = 16
    batch_size = points.size()[0]
    point_dim = points.size()[1]
    point_num = points.size()[2]

    feature = torch.zeros(batch_size, gt_feature_len, point_num,requires_grad=False)
    return torch.cat([points, feature], 1 )
