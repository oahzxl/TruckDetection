# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101),
    roi_head=dict(
        bbox_head=dict(num_classes=3)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('legal', 'illegal', 'empty',)
data = dict(
    train=dict(
        img_prefix='/home/zxl/TruckDetection/data/origin/train',
        classes=classes,
        ann_file='/home/zxl/TruckDetection/data/origin/train.json'),
    val=dict(
        img_prefix='/home/zxl/TruckDetection/data/origin/val',
        classes=classes,
        ann_file='/home/zxl/TruckDetection/data/origin/val.json'),
    test=dict(
        img_prefix='/home/zxl/TruckDetection/data/origin/val',
        classes=classes,
        ann_file='/home/zxl/TruckDetection/data/origin/val.json'))
