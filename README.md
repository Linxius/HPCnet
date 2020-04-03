# HPCnet

## Installation

### Install
Install this library by running the following command:

```shell
cd pointnet2
python setup.py install
cd ../

cd HPCnet
python setup.py install
cd ../
```

## Examples

### KITTI
data:
```
 ├──data
 │  ├── KITTI
 │  │   ├── ImageSets
 │  │   ├── object
 │  │   │   ├──training
 │  │   │      ├──calib & velodyne & label_2 & image_2
```

train and test:
```
python tools/kitti_train_test.py
```

net arch in `HPCnet/hpcnet_kitti.py`

### ModelNet40
put data in `data/modelnet40_normal_resampled`
