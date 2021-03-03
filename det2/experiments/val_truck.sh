var_weight=/data01/zxl/Truck-Detection/outputs/faster_rcnn_R_50_FPN_3x/model.pth
cd ../tools
python ./plain_train_net.py  --eval-only --num-gpus 1 --config-file ../configs/Truck-Detection/faster_rcnn_R_50_FPN_3x.yaml MODEL.WEIGHTS $var_weight
