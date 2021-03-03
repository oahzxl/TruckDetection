# 卡车检测项目

## 需求

* 从图片里的黄牌车辆中找出载货车辆
* 准确率>90%
* 完成时间4-5月份
* 计算量未确定

# 车篷检测文档

环境初始化 服务器43005

```bash
# 安装detectron2
pip install -e Truck-Detection
# 创建数据集软连接
cd Truck-Detection
ln -s /data01/zxl/Truck-Detection/data data
```

## 检测

训练

```bash
cd ./detection/tools
python train_net.py --num-gpus 1 --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml
```

验证

```bash
cd ./detection/tools
python ./plain_train_net.py --eval-only --num-gpus 1 --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml MODEL.WEIGHTS ../outputs/faster_rcnn_R_101_FPN_3x/model.pth
```

输出卡车检测的可视化结果到../data/demo

```bash
cd ./detection/demo
rm ../../data/demo/*
python demo.py \
  --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml \
  --opts MODEL.WEIGHTS ../outputs/faster_rcnn_R_101_FPN_3x/model.pth
```

将图片中的gt截出

```bash
cd ./detection/tools
# 需要自己创建文件夹，可以在236-240行修改目标数据集和扩展大小
python ./crop_gt.py  --eval-only --num-gpus 1 --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml --eval-only
```

将图片中的预测结果截出

```bash
cd ./detection/demo
rm ../../data/demo/*
python crop.py --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml --opts MODEL.WEIGHTS ../outputs/faster_rcnn_R_101_FPN_3x/model.pth
```
