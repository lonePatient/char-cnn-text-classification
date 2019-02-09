#encoding:utf-8
import torch
import warnings
from torch import optim
from pyCharCnn.train.metrics import F1Score
from pyCharCnn.train.metrics import ClassReport
from pyCharCnn.train.losses import CrossEntropy
from pyCharCnn.train.trainer import Trainer
from torch.utils.data import DataLoader
from pyCharCnn.io.dataset import CreateDataset
from pyCharCnn.io.data_transformer import DataTransformer
from pyCharCnn.utils.logginger import init_logger
from pyCharCnn.utils.utils import seed_everything
from pyCharCnn.model.nn.character_cnn import CharacterCNN
from pyCharCnn.config.basic_config import configs as config
from pyCharCnn.preprocessing.preprocessor import Preprocessor
from pyCharCnn.callback.modelcheckpoint import ModelCheckpoint
from pyCharCnn.callback.trainingmonitor import TrainingMonitor
from pyCharCnn.callback.lrscheduler import ReduceLROnPlateau
warnings.filterwarnings("ignore")

# 主函数
def main():
    # **************************** 基础信息 ***********************
    logger = init_logger(log_name=config['arch'], log_dir=config['log_dir'])
    logger.info("seed is %d"%config['seed'])
    device = 'cuda:%d' % config['n_gpus'][0] if len(config['n_gpus']) else 'cpu'
    seed_everything(seed=config['seed'],device=device)
    logger.info('starting load data from disk')
    config['id_to_label'] = {v:k for k,v in config['label_to_id'].items()}
    # **************************** 数据生成 ***********************
    train_trf = DataTransformer(logger      = logger,
                                label_to_id = config['label_to_id'],
                                train_file  = config['train_process_path'],
                                valid_file  =None,
                                valid_size  = None,
                                seed        = config['seed'],
                                shuffle     = True,
                                skip_header = False,
                                data_path   = config['train_data_path'],
                                preprocess  = Preprocessor(),
                                vocab_path  = config['vocab_path'],
                                alphabet    = config['alphabet'])
    valid_trf = DataTransformer(logger      = logger,
                                label_to_id = config['label_to_id'],
                                train_file  = None,
                                valid_file  =config['valid_process_path'],
                                valid_size  = None,
                                seed        = config['seed'],
                                shuffle     = False,
                                skip_header = False,
                                data_path   = config['valid_data_path'],
                                preprocess  = Preprocessor(),
                                vocab_path  = config['vocab_path'],
                                alphabet    = config['alphabet'])
    # 读取数据集以及数据划分
    train_trf.read_data()
    valid_trf.read_data()
    # train
    train_dataset   = CreateDataset(data_path    = config['train_process_path'],
                                    max_seq_len  = config['max_seq_len'],
                                    seed         = config['seed'],
                                    example_type = 'train',
                                    num_of_char  = config['num_of_char'])
    # valid
    valid_dataset   = CreateDataset(data_path    = config['valid_process_path'],
                                    max_seq_len  = config['max_seq_len'],
                                    seed         = config['seed'],
                                    example_type = 'valid',
                                    num_of_char  = config['num_of_char'])
    #加载训练数据集
    train_loader = DataLoader(dataset     = train_dataset,
                              batch_size  = config['batch_size'],
                              num_workers = config['num_workers'],
                              shuffle     = True,
                              drop_last   = False,
                              pin_memory  = False)
    # 验证数据集
    valid_loader = DataLoader(dataset     = valid_dataset,
                              batch_size  = config['batch_size'],
                              num_workers = config['num_workers'],
                              shuffle     = False,
                              drop_last   = False,
                              pin_memory  = False)

    # **************************** 模型 ***********************
    logger.info("initializing model")
    model = CharacterCNN(num_classes = len(config['label_to_id']),
                         max_len_seq = config['max_seq_len'],
                         in_channels = config['num_of_char'])

    # ************************** 优化器 *************************

    optimizer = optim.Adam(params=model.parameters(), lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    # **************************** callbacks ***********************
    logger.info("initializing callbacks")
    # 模型保存
    model_checkpoint = ModelCheckpoint(checkpoint_dir   = config['checkpoint_dir'],
                                       mode             = config['mode'],
                                       monitor          = config['monitor'],
                                       save_best_only   = config['save_best_only'],
                                       best_model_name  = config['best_model_name'],
                                       epoch_model_name = config['epoch_model_name'],
                                       arch             = config['arch'],
                                       logger           = logger)
    # 监控训练过程
    train_monitor = TrainingMonitor(fig_dir  = config['figure_dir'],
                                    json_dir = config['log_dir'],
                                    arch     = config['arch'])
    # 学习率机制
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                     factor=0.5,
                                     patience=config['lr_patience'],
                                     min_lr=1e-9,
                                     epsilon=1e-5,
                                     verbose=1,
                                     mode=config['mode'])

    # **************************** training model ***********************
    logger.info('training model....')
    trainer = Trainer(model            = model,
                      train_data       = train_loader,
                      val_data         = valid_loader,
                      optimizer        = optimizer,
                      epochs           = config['epochs'],
                      criterion        = CrossEntropy(),
                      logger           = logger,
                      model_checkpoint = model_checkpoint,
                      training_monitor = train_monitor,
                      resume           = config['resume'],
                      lr_scheduler     = lr_scheduler,
                      n_gpu            = config['n_gpus'],
                      evaluate         = F1Score(),
                      class_report     = ClassReport(target_names=[config['id_to_label'][x]
                                                                   for x in range(len(config['label_to_id']))]))
    # 查看模型结构
    trainer.summary()
    # 拟合模型
    trainer.train()
    # 释放显存
    if len(config['n_gpus']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
