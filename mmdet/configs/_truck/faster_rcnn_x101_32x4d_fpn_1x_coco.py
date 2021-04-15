_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',),
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

runner = dict(type='EpochBasedRunner', max_epochs=15)
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Truck',)
data = dict(
    train=dict(
        img_prefix='/data01/zyh/truck/train',
        classes=classes,
        ann_file='/data01/zyh/truck/annotations/train_total.json'),
    val=dict(
        img_prefix='/data01/zyh/truck/val',
        classes=classes,
        ann_file='/data01/zyh/truck/annotations/val_total.json'),
    test=dict(
        img_prefix='/data01/zyh/truck/val',
        classes=classes,
        ann_file='/data01/zyh/truck/annotations/val_total.json'))
