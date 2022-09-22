#python train_img.py --root 'D:\workspace\data\dl' --dataset 'dukemtmc' --optim 'sgd' --max_epoch 50 --train_batch 48 --test_batch 36 --arch 'resnet50' --pre_train_load_dir 'D:\workspace\data\classification_data\resnet\resnet50-19c8e357.pth' --loss_type 'softmax_trip' --resume 'checkpoint.pth' --eval_step 10 --save_dir 'D:\workspace\data\classification_data\resnet'


python train_img.py --root '/root/autodl-tmp' \
--dataset 'market1501' \
--optim 'adam' \
--max_epoch 50 \
--train_batch 64 \
--test_batch 36 \
--arch 'resnet50' \
--pre_train_load_dir '/root/autodl-nas/classification_data/resnet/pths/resnet50-19c8e357.pth' \
--loss_type 'softmax_trip' \
--resume 'checkpoint.pth' \
--eval_step 20 \
--save_dir '/root/autodl-nas/reid_data/resnet50_mine_0922_trip'
