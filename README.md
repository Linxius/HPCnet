# Download data and running

```
git clone https://github.com/Linxius/GTnet.git
cd GTnet
pip3 install -e .
```

Download and build visualization tool
```
cd script
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```

Training
```
python3 utils/train_gt_cls.py  --dataset ~/data/shapenet_core --nepoch=1 --dataset_type shapenet --num_points 20

```

Training pointnet
```
python3 utils/train_classification.py --dataset ~/data/shapenet_core --nepoch=1 --dataset_type shapenet --num_points 20
```

Use `--feature_transform` to use feature transform.
