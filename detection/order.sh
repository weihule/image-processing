python train.py  --dataset_name 'coco2017' \
--num_classes 80 \
--root_dir '/ssd/weihule/data/dl/VOCdataset' \
--image_root_dir '/workshop/weihule/data/dl/COCO2017/images' \
--annotation_root_dir '/workshop/weihule/data/dl/COCO2017/annotations' \
--resize 560 \
--max_epoch 30 \
--train_batch 16 \
--test_batch 16 \
--gpu_devices '6' \
--optim 'adam' \
--lr 0.0001 \
--arch 'resnet50_retinanet' \
--pre_train_load_dir '/workshop/weihule/data/detection_data/retinanet/checkpoints/resnet50-acc76.322.pth' \
--num_workers 8 \
--step_size 5 \
--print_freq 80 \
--resume 'checkpoint_ep1.pth' \
--eval_step 1 \
--use_mosaic \
--save_dir '/workshop/weihule/data/detection_data/retinanet_coco1024' \
--apex


#python train.py  --dataset_name 'voc' \
#--num_classes 20 \
#--root_dir '/root/autodl-tmp/VOCdataset' \
#--image_root_dir '/root/autodl-tmp/COCO2017/images' \
#--annotation_root_dir '/root/autodl-tmp/COCO2017/annotations' \
#--resize 640 \
#--max_epoch 30 \
#--train_batch 8 \
#--test_batch 8 \
#--gpu_devices '0' \
#--optim 'adam' \
#--lr 0.0001 \
#--arch 'resnet50_retinanet' \
#--pre_train_load_dir '/root/autodl-nas/classification_data/resnet/pths/resnet50-acc76.322.pth' \
#--num_workers 8 \
#--step_size 50 \
#--resume '/root/autodl-nas/detection_data/new_retinanet/checkpoint_ep1.pth' \
#--eval_step 2 \
#--use_mosaic \
#--save_dir '/root/autodl-nas/detection_data/new_retinanet'

