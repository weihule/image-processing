#python train_img.py --root 'D:\workspace\data\dl' --dataset 'dukemtmc' --optim 'sgd' --max_epoch 50 --train_batch 48 --test_batch 36 --arch 'resnet50' --pre_train_load_dir 'D:\workspace\data\classification_data\resnet\resnet50-19c8e357.pth' --loss_type 'softmax_trip' --resume 'checkpoint.pth' --eval_step 10 --save_dir 'D:\workspace\data\classification_data\resnet'


#python train_img.py --root '/root/autodl-tmp' \
#--dataset 'market1501' \
#--optim 'adam' \
#--max_epoch 120 \
#--train_batch 80 \
#--step_size 50 \
#--test_batch 64 \
#--arch 'resnet50' \
#--pre_train_load_dir '/root/autodl-nas/classification_data/resnet/pths/resnet50-19c8e357.pth' \
#--loss_type 'softmax_trip' \
#--resume 'checkpoint.pth' \
#--eval_step 25 \
#--save_dir '/root/autodl-nas/reid_data/resnet50_mine_trip_aligned_0928' \
#--aligned


# 128.55
python train_img.py --root '/workshop/weihule/data/dl/reid' \
--gpu_devices '5' \
--dataset 'market1501' \
--optim 'adam' \
--max_epoch 25 \
--train_batch 80 \
--step_size 50 \
--test_batch 64 \
--arch 'osnet_x1_0' \
--pre_train_load_dir  '/workshop/weihule/data/classification_data/reid/osnet_x1_0_imagenet.pth' \
--loss_type 'softmax' \
--resume 'demo.pth' \
--eval_step 10 \
--save_dir '/workshop/weihule/data/reid_data/osnet_x1_0' \

# '/workshop/weihule/data/classification_data/reid/osnet_x1_0_imagenet.pth'


