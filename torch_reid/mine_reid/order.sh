#python train_img.py --root 'D:\workspace\data\dl' --dataset 'dukemtmc' --optim 'sgd' --max_epoch 50 --train_batch 48 --test_batch 36 --arch 'resnet50' --pre_train_load_dir 'D:\workspace\data\classification_data\resnet\resnet50-19c8e357.pth' --loss_type 'softmax_trip' --resume 'checkpoint.pth' --eval_step 10 --save_dir 'D:\workspace\data\classification_data\resnet'


#python train_img.py --root '/root/autodl-tmp' \
#--dataset 'market1501' \
#--optim 'adam' \
#--max_epoch 180 \
#--train_batch 64 \
#--step_size 50 \
#--test_batch 64 \
#--arch 'osnet_x1_0' \
#--pre_train_load_dir '/root/autodl-nas/classification_data/osnet/osnet_x1_0_imagenet.pth' \
#--loss_type 'softmax_trip' \
#--resume 'checkpoint.pth' \
#--eval_step 25 \
#--save_dir '/root/autodl-nas/reid_data/osnet_1014_epoch0' \
#--aligned


# 128.55
python train_img.py --root '/workshop/weihule/data/dl/reid' \
--gpu_devices '6' \
--dataset 'market1501' \
--optim 'adam' \
--max_epoch 150 \
--train_batch 80 \
--step_size 50 \
--test_batch 64 \
--arch 'osnet_x1_0' \
--pre_train_load_dir  '/workshop/weihule/data/classification_data/reid/epoch7_7574.pth' \
--loss_type 'softmax_trip' \
--resume 'demo.pth' \
--eval_step 50 \
--save_dir '/workshop/weihule/data/reid_data/osnet_x1_0_1015_aligned_epoch7' \
--aligned

# '/workshop/weihule/data/classification_data/reid/osnet_x1_0_imagenet.pth'

