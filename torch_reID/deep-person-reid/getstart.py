# 模块引入
import torchreid

root = '/workshop/weihule/data/dl/reid'

# 加载数据管理器
datamanager = torchreid.data.ImageDataManager(
    root=root,
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)
# 构建模型、优化器和lr_scheduler
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
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
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)
# 进行培训和测试
engine.run(
    save_dir='log/osnet_x1_0',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False
)