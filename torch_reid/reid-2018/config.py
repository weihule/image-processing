class Config:
    root ='D:\\workspace\\data\\dl'
    dataset_name = 'market1501'
    workers = 4
    height = 256
    width = 128
    split_id = 0
    
    # Optimization options
    max_epoch = 60
    start_epoch = 0
    train_batch = 32
    test_batch = 32
    learning_rate = 0.0003
    lr_cent = 0.5
    weight_cent = 0.0005 
    stepsize = 20
    gamma = 0.1
    weight_decay = 5e-04

    # Architecture
    arch = 'resnet50'

    # Miscs
    print_freq = 10
    seed = 1
    resume = ''
    evaluate = ''
    eval_step = 1
    start_eval = 0
    save_dir = 'D:\\workspace\\data\\reid_data\\demo'
    use_cpu = False
    gpu_devices = '0'

