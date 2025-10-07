#python train_img.py --root 'D:\workspace\data\dl\reid' --dataset 'market1501' --optim 'adam' --max_epoch 50 --train_batch 48 --test_batch 36 --arch 'osnet_x1_0' --pre_train_load_dir 'D:\workspace\data\classification_data\osnet\pre_weights\osnet_x1_0_imagenet.pth' --loss_type 'softmax_trip' --resume 'checkpoint.pth' --eval_step 10 --save_dir 'D:\workspace\data\classification_data\resnet' --aligned


python train_img.py --root '/root/autodl-tmp' \
--dataset 'duke' \
--optim 'adam' \
--max_epoch 300 \
--train_batch 80 \
--test_batch 64 \
--lr 0.0005 \
--arch 'sc_osnet_x1_0_origin' \
--pre_train_load_dir '/root/autodl-nas/classification_data/osnet/pre_weights/osnet_x1_0_imagenet.pth' \
--loss_type 'softmax_trip' \
--resume 'checkpoint.pth' \
--eval_step 10 \
--save_dir '/root/autodl-nas/reid_train_data/duke/sc_osnet_x1_0_origin_relu_nam_epoch300_lr0005' \
--gpu_devices '1' \

# 128.55
#python train_img.py --root '/workshop/weihule/data/dl/reid' \
#--gpu_devices '6' \
#--dataset 'market1501' \
#--optim 'adam' \
#--max_epoch 150 \
#--train_batch 80 \
#--step_size 50 \
#--test_batch 64 \
#--arch 'osnet_x1_0' \
#--pre_train_load_dir  '/workshop/weihule/data/classification_data/reid/epoch7_7574.pth' \
#--loss_type 'softmax_trip' \
#--resume 'demo.pth' \
#--eval_step 50 \
#--save_dir '/workshop/weihule/data/reid_data/osnet_x1_0_1015_aligned_epoch7' \
#--aligned


