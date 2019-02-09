# encoding:utf-8
import os
import time
import numpy as np
import torch
from ..callback.progressbar import ProgressBar
from ..utils.utils import AverageMeter
from .train_utils import restore_checkpoint, model_device


# 训练包装器
class Trainer(object):
    def __init__(self, model,
                 train_data,
                 val_data,
                 optimizer,
                 epochs,
                 logger,
                 evaluate,
                 criterion,
                 class_report,
                 n_gpu=None,
                 lr_scheduler=None,
                 resume=None,
                 model_checkpoint=None,
                 training_monitor=None,
                 early_stopping=None,
                 verbose=1):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.optimizer = optimizer
        self.logger = logger
        self.verbose = verbose
        self.training_monitor = training_monitor
        self.early_stopping = early_stopping
        self.resume = resume
        self.model_checkpoint = model_checkpoint
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.class_report = class_report
        self.evaluate = evaluate
        self.n_gpu = n_gpu
        self._reset()

    def _reset(self):
        self.batch_num = len(self.train_data)
        self.progressbar = ProgressBar(n_batch=self.batch_num)
        self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model, logger=self.logger)
        self.start_epoch = 1
        # 重载模型，进行训练
        if self.resume:
            arch = self.model_checkpoint.arch
            resume_path = os.path.join(self.model_checkpoint.checkpoint_dir.format(arch=arch),
                                       self.model_checkpoint.best_model_name.format(arch=arch))
            self.logger.info("\nLoading checkpoint: {} ...".format(resume_path))
            resume_list = restore_checkpoint(resume_path=resume_path, model=self.model, optimizer=self.optimizer)
            self.model = resume_list[0]
            self.optimizer = resume_list[1]
            best = resume_list[2]
            self.start_epoch = resume_list[3]
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info("\nCheckpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))


    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # for p in model_parameters:
        #     print(p.size())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # 总的模型参数量
        self.logger.info('Model: trainable parameters: {:4}M'.format(params / 1000 / 1000))
        # 模型结构
        self.logger.info(self.model)

    # 保存模型信息
    def _save_info(self, epoch, val_loss):
        state = {
            'arch': self.model_checkpoint.arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': round(val_loss, 4)
        }
        return state

    # val数据集预测
    def _valid_epoch(self):
        val_loss, count = 0, 0
        predicts = []
        targets = []
        self.model.eval()
        with torch.no_grad():
            for step, (data, target) in enumerate(self.val_data):
                data, target = data.to(self.device), target.to(self.device)
                logits = self.model(data)
                loss = self.criterion(logits, target)
                val_loss += loss.item()
                predicts.append(logits)
                targets.append(target)
                count += 1
        predicts = torch.cat(predicts, dim=0).cpu()
        targets = torch.cat(targets, dim=0).cpu()
        val_acc, val_f1 = self.evaluate(output=predicts, target=targets)
        self.class_report(predicts, targets)

        return {
            'val_loss': val_loss / count,
            'val_acc': val_acc,
            'val_f1': val_f1
        }

    # epoch训练
    def _train_epoch(self):
        self.model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_f1 = AverageMeter()
        for step, (data, target) in enumerate(self.train_data):
            start = time.time()
            data = data.to(self.device)
            target = target.to(self.device)

            logits = self.model(data)
            loss = self.criterion(output=logits, target=target)
            acc, f1 = self.evaluate(output=logits, target=target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item())
            train_acc.update(acc.item())
            train_f1.update(f1)
            if self.verbose >= 1:
                self.progressbar.step(batch_idx=step,
                                      loss=loss.item(),
                                      acc=acc,
                                      f1=f1,
                                      use_time=time.time() - start)
            print("\ntraining result:")
            train_log = {
                'loss': train_loss.avg,
                'acc': train_acc.avg,
                'f1': train_f1.avg
            }
            return train_log

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            print("----------------- training start -----------------------")
            print("Epoch {i}/{epochs}......".format(i=epoch, epochs=self.start_epoch + self.epochs - 1))
            train_log = self._train_epoch()
            val_log = self._valid_epoch()
            logs = dict(train_log, **val_log)
            self.logger.info(
                '\nEpoch: %d - loss: %.4f acc: %.4f - f1: %.4f val_loss: %.4f - val_acc: %.4f - val_f1: %.4f' % (
                    epoch, logs['loss'], logs['acc'], logs['f1'], logs['val_loss'], logs['val_acc'], logs['val_f1']))

            if self.lr_scheduler:
                self.lr_scheduler.step(logs['loss'], epoch)
            if self.training_monitor:
                self.training_monitor.step(logs)
            if self.model_checkpoint:
                state = self._save_info(epoch, val_loss=logs['val_loss'])
                self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor], state=state)
            if self.early_stopping:
                self.early_stopping.step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break



