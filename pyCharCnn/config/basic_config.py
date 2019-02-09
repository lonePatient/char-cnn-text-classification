#encoding:utf-8
from os import path
import multiprocessing


BASE_DIR = 'pyCharCnn'

configs = {
    'arch':'char_cnn',
    'train_data_path': path.sep.join([BASE_DIR,'dataset/raw/train.csv']),
    'valid_data_path': path.sep.join([BASE_DIR,'dataset/raw/test.csv']),
    'train_process_path': path.sep.join([BASE_DIR, 'dataset/processed/train.csv']),
    'valid_process_path': path.sep.join([BASE_DIR, 'dataset/processed/test.csv']),
    'vocab_path':path.sep.join([BASE_DIR, 'dataset/processed/vocab.pkl']),
    'log_dir': path.sep.join([BASE_DIR, 'output/log']), # 模型运行日志
    'figure_dir': path.sep.join([BASE_DIR, 'output/figure']), # 图形保存路径
    'checkpoint_dir': path.sep.join([BASE_DIR, 'output/checkpoints']),# 模型保存路径

    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    "num_of_char": 69,

    'valid_size': 0.3, # valid数据集大小
    'max_seq_len': 1014,  # word文本平均长度,按照覆盖95%样本的标准，取截断长度:np.percentile(list,95.0)
    "embedding_size": 128,
    'batch_size': 128,   # how many samples to process at once
    'epochs': 5000,       # number of epochs to train
    'start_epoch': 1,

    'learning_rate': 0.01,
    'n_gpus': [], # GPU个数,如果只写一个数字，则表示gpu标号从0开始，并且默认使用gpu:0作为controller,
                     # 如果以列表形式表示，即[1,3,5],则我们默认list[0]作为controller
    'weight_decay':5e-4,

    'num_workers': multiprocessing.cpu_count(), # 线程个数
    'resume':False,
    'seed': 2018,
    'lr_patience': 5, # number of epochs with no improvement after which learning rate will be reduced.
    'mode': 'min',    # one of {min, max}
    'monitor': 'val_loss',  # 计算指标
    'early_patience': 10,   # early_stopping
    'save_best_only': True, # 是否保存最好模型
    'best_model_name': '{arch}-best2.pth', #保存文件
    'epoch_model_name': '{arch}-{epoch}-{val_loss}.pth', #以epoch频率保存模型
    'save_checkpoint_freq': 10, #保存模型频率，当save_best_only为False时候，指定才有作用

    'label_to_id' : { # 标签映射
        'World':0,
        'Sports':1,
        'Business':2,
        'Sci/Tech':3
    }
}
