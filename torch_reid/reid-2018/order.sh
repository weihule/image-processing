#CUDA_VISIBLE_DEVICES=0 python  train_img_model_cent.py --root '/workshop/weihule/data/dl/reid' --stepsize 15 --train-batch 96 --save-dir '/workshop/weihule/data/reid_data/demo' --arch 'osnet_x1_0' --pre_trained '/workshop/weihule/data/classification_data/reid/osnet_x1_0_imagenet.pth'
#CUDA_VISIBLE_DEVICES=0 python train_img_model_cent.py --gpu-devices 0 --root '/root/autodl-tmp' --stepsize 15 --train-batch 96 --save-dir '/root/autodl-nas/reid_data/trip/cent' --arch 'osnet_x1_0' --pre_trained '/root/autodl-nas/classification_data/osnet/osnet_x1_0_imagenet.pth'

# 128.55
#python train_img_model_xent_htri.py \
#--gpu_devices 7 \
#--root '/workshop/weihule/data/dl/reid' \
#--dataset 'market1501' \
#--stepsize 20 --train_batch 96 \
#--max_epoch 150 \
#--save_dir '/workshop/weihule/data/reid_data/trip' \
#--arch 'osnet_x1_0' \
#--resume '' \
#--pre_trained '/workshop/weihule/data/classification_data/reid/osnet_x1_0_imagenet.pth'

# autodl
python train_img_model_xent_htri.py \
--gpu_devices 0 \
--root '/root/autodl-tmp' \
--dataset 'market1501' \
--stepsize 25 \
--train_batch 64 \
--max_epoch 20 \
--save_dir '/root/autodl-nas/reid_data/trip' \
--arch 'osnet_x1_0' \
--resume '/root/autodl-nas/reid_data/trip/checkpoint_ep5.pth' \
--pre_trained '/root/autodl-nas/classification_data/osnet/osnet_x1_0_imagenet.pth'

#python train_img_model_xent_htri.py --gpu_devices 7 --root 'D:\\workspace\\data\\dl' --dataset 'market1501' --stepsize 20 --train_batch 4 --save_dir 'D:\\workspace\\data\\reid_data\trip' --arch 'osnet_x1_0' --pre_trained 'D:\\workspace\\data\\classification_data\\osnet\\checkpoints\\osnet_x1_0_imagenet.pth'