# CUDA_VISIBLE_DEVICES=7 python  train_img_model_cent.py --root '/workshop/weihule/data/dl/reid' --stepsize 15 --train-batch 96 --save-dir '/workshop/weihule/data/reid_data/demo' --arch 'osnet_x1_0' --pre_trained '/workshop/weihule/data/classification_data/reid/osnet_x1_0_imagenet.pth'

CUDA_VISIBLE_DEVICES=7 python train_img_model_xent_htri.py\
--root '/workshop/weihule/data/dl/reid'\
--dataset 'market1501'\
--stepsize 20 --train-batch 96\
--save-dir '/workshop/weihule/data/reid_data/trip'\
--arch 'osnet_x1_0'\
--pre_trained '/workshop/weihule/data/classification_data/reid/osnet_x1_0_imagenet.pth'
