cd ../tools
var_config=../configs/Truck-Detection/faster_rcnn_R_50_FPN_1x.yaml
CUDA_VISIBLE_DEVICES=2 python ./train_net.py --resume --num-gpus 1 --dist-url tcp://0.0.0.0:12345 --config-file $var_config
