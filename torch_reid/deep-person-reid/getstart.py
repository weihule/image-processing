# 模块引入
import os
import torchreid
from torchreid import data, models, optim, engine

mode = 'local'
roots = {'autodl': '/root/autodl-tmp',
         'local': 'D:\\workspace\\data\\dl',
         'company': '/workshop/weihule/data/dl/reid'}
root = roots[mode]


# 存放预训练权重，log文件，训练好的pth文件
default_paths = {'autodl': '/root/autodl-tmp/osnet',
                 'local': 'D:\\workspace\\data\\weights\\osnet',
                 'company': '/workshop/weihule/data/dl/reid'}
default_path = default_paths[mode]

model_name = 'osnet_x1_0'

if __name__ == "__main__":

    # 加载数据管理器
    datamanager = torchreid.data.ImageDataManager(
        root=root,
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=4,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    )
    # 构建模型、优化器和lr_scheduler
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=datamanager.num_train_pids,
        default_path=default_path,
        loss='softmax',
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )
    # Build engine
    train_engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    save_folder = os.path.join(default_path, 'log', model_name)
    # 进行培训和测试
    train_engine.run(
        save_dir=save_folder,
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )

