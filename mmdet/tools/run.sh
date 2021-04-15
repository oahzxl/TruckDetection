CUDA_VISIBLE_DEVICES=1 PORT=29503 bash tools/dist_train.sh configs/_truck/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco1.py 1
CUDA_VISIBLE_DEVICES=1 PORT=29503 bash tools/dist_train.sh configs/_truck/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco2.py 1
CUDA_VISIBLE_DEVICES=1 PORT=29503 bash tools/dist_train.sh configs/_truck/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco3.py 1
