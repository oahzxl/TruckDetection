cd ../tools
tag_name=1121
which_model=../tools/output-1/model_0010999.pth
var_out=../tools/vis
cd ../demo  #../datasets/eval/baiguang/*
python demo.py  --output $var_out  --config-file ../configs/Truck-Detection/faster_rcnn_R_50_FPN_1x.yaml  --input ../datasets/truck/truck_train_imgs/*     --opts MODEL.WEIGHTS $which_model  #../pretrained/output/model_0004999.pth
