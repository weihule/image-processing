#python train.py  --dataset_name 'voc' \
#--num_classes 20 \
#--root_dir '/ssd/weihule/data/dl/VOCdataset' \
#--image_root_dir '/workshop/weihule/data/dl/COCO2017/images' \
#--annotation_root_dir '/workshop/weihule/data/dl/COCO2017/annotations' \
#--resize 640 \
#--max_epoch 50 \
#--train_batch 32 \
#--test_batch 32 \
#--gpu_devices '5' \
#--optim 'adam' \
#--max_epoch 150 \
#--lr 0.0001 \
#--arch 'resnet50_retinanet' \
#--pre_train_load_dir '/workshop/weihule/data/detection_data/retinanet/checkpoints/resnet50-acc76.322.pth' \
#--step_size 50 \
#--resume 'demo.pth' \
#--use_mosaic True \
#--eval_step 50 \
#--save_dir '/workshop/weihule/data/detection_data/new_retinanet'


python train.py  --dataset_name 'voc' \
--num_classes 20 \
--root_dir '/root/autodl-tmp/VOCdataset' \
--image_root_dir '/root/autodl-tmp/COCO2017/images' \
--annotation_root_dir '/root/autodl-tmp/COCO2017/annotations' \
--resize 640 \
--max_epoch 13 \
--train_batch 8 \
--test_batch 8 \
--gpu_devices '0' \
--optim 'adam' \
--max_epoch 150 \
--lr 0.0001 \
--arch 'resnet50_retinanet' \
--pre_train_load_dir '/root/autodl-nas/classification_data/resnet/pths/resnet50-acc76.322.pth' \
--num_workers 8 \
--step_size 50 \
--resume 'demo.pth' \
--use_mosaic \
--eval_step 50 \
--save_dir '/root/autodl-nas/detection_data/new_retinanet'

