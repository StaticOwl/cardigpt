"""
File: train.py
Project: potluck
Author: malli
Created: 06-10-2024
Description: write_a_description
"""
import logging
import math
import os
import shutil
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
from dataset.ECGDataset import ECGDataset
from utils.freeze import set_freeze_by_id
from utils.metrics import cal_acc


class Trainer:
    def __init__(self, args):
        """
        Initialize the Trainer class

        Args:
            args: The arguments for the training process
            model: The model to be trained
            optimizer: The optimizer for the training process
            criterion: The loss function for the training process
            device: The device to be used for training
        """
        self.early_stop = args.early_stop
        self.patience = args.patience
        self.epochs_since_improvement = 0
        self.min_delta = args.min_delta
        self.num_classes = None
        self.save_dir = args.save_dir
        self.sigmoid = nn.Sigmoid()
        self.lr_scheduler = None
        self.dataloaders = None
        self.device_count = None
        self.dataset = None
        self.args = args
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = None
        self.start_epoch = 0
        self.cal_acc = cal_acc

    def setup(self):
        """
        Set up the training process
        """
        args = self.args

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))

        else:
            logging.warning("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        prepared_data = ECGDataset(args).data_preprare()

        self.dataset = {'train': prepared_data[0],
                        'val': prepared_data[1]}

        self.dataloaders = {
            x: DataLoader(self.dataset[x], batch_size=(args.batch_size if x == 'train' else 1),
                          shuffle=(True if x == 'train' else False),
                          num_workers=args.num_workers,
                          pin_memory=(True if self.device == 'cuda' else False),
                          drop_last=True)
            for x in ['train', 'val']
        }
        self.num_classes = ECGDataset.num_classes
        model_name = getattr(model, args.model_name)
        self.model = model_name(pretrained=True, in_channel=ECGDataset.input_channel, out_channel=self.num_classes)
        if args.layers_num_last is not None:
            if args.layers_num_last != 0:
                set_freeze_by_id(self.model, args.layers_num_last)
        if self.device_count > 1:
            self.model = nn.DataParallel(self.model)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                    weight_decay=args.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        self.model.to(self.device)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            for f in os.listdir(self.save_dir):
                file_path = os.path.join(self.save_dir, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logging.error(e)

    def train(self):
        """
        Train the model
        """
        global labels_all, logits_prob_all, logits_prob, loss_temp, loss
        args = self.args

        step = 0
        batch_count = 0
        best_acc = 0.0
        batch_loss = 0.0
        step_start = time.time()
        writer = SummaryWriter(log_dir="./logs")

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
                writer.add_scalar('Learning Rate', self.lr_scheduler.get_last_lr()[0], epoch)
            else:
                logging.info('current lr: {}'.format(args.lr))
                writer.add_scalar('Learning Rate', args.lr, epoch)

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_loss = 0.0

                # Iterate over data.
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                for batch_idx, (inputs, ag, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    ag = ag.to(self.device)
                    labels = labels.to(self.device)
                    # zero the parameter gradients
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        # track history if only in train
                        if phase == 'train':
                            logits = self.model(inputs, ag)
                            logits_prob = self.sigmoid(logits)
                            logits_prob = logits
                            if batch_idx == 0:
                                labels_all = labels
                                logits_prob_all = logits_prob
                            else:
                                labels_all = torch.cat((labels_all, labels), dim=0)
                                logits_prob_all = torch.cat((logits_prob_all, logits_prob), dim=0)

                            loss = self.criterion(logits, labels)
                            loss_temp = loss.item() * inputs.size(0)
                            epoch_loss += loss_temp

                            # backward + optimize only if in training phase
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_count += inputs.size(0)

                            if step % args.log_step == 0:
                                batch_loss_avg = batch_loss / batch_count
                                train_time = time.time() - step_start
                                step_start = time.time()
                                batch_time = train_time / args.log_step if step != 0 else train_time
                                samples_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f},'
                                             'samples/sec: {:.1f} examples/sec, time: {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss_avg, samples_per_sec, batch_time
                                ))
                                batch_loss = 0.0
                                batch_count = 0
                                writer.add_scalar('Train Loss', batch_loss_avg, step)

                            step += 1

                        else:
                            val_length = inputs.shape[2]
                            win_length = args.win_length if args.win_length else 4096
                            overlap = args.overlap if args.overlap else 256
                            patch_number = math.ceil(abs(val_length - win_length) / (win_length - overlap)) + 1
                            if patch_number > 1:
                                start = int((val_length - win_length) / (patch_number - 1))
                            for i in range(patch_number):
                                if i == 0:
                                    logits_prob, loss_temp = self.nn_forward(inputs[:, :, 0: val_length], ag, labels)
                                elif i == patch_number - 1:
                                    logits_prob_tmp, loss_temp_tmp = self.nn_forward(
                                        inputs[:, :, val_length - win_length: val_length], ag, labels)
                                    logits_prob = (logits_prob + logits_prob_tmp) / patch_number
                                    loss_temp = (loss_temp + loss_temp_tmp) / patch_number
                                else:
                                    logits_prob_tmp, loss_temp_tmp = self.nn_forward(
                                        inputs[:, :, i * start:i * start + win_length], ag, labels)
                                    logits_prob += logits_prob_tmp
                                    loss_temp += loss_temp_tmp

                            if batch_idx == 0:
                                labels_all = labels
                                logits_prob_all = logits_prob
                            else:
                                labels_all = torch.cat((labels_all, labels), dim=0)
                                logits_prob_all = torch.cat((logits_prob_all, logits_prob), dim=0)
                            epoch_loss += loss_temp

                metric = self.cal_acc(labels_all, logits_prob_all)
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)

                writer.add_scalar(f'{phase.capitalize()} Loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase.capitalize()} Loss', epoch_loss, epoch)
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-challenge_metric: {:.4f}, Cost {:.1f} sec'.
                             format(epoch, phase, epoch_loss, phase, metric, time.time() - epoch_start))

                if phase == 'val':
                    epoch_acc = metric
                    if epoch_acc > best_acc + self.min_delta:
                        best_acc = epoch_acc
                        self.epochs_since_improvement = 0

                        model_state = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                        torch.save(model_state,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))
                        logging.info(f'New best model saved at epoch {epoch} with accuracy {best_acc:.4f}')
                    else:
                        self.epochs_since_improvement += 1
                        if self.early_stop and self.epochs_since_improvement > self.patience:
                            logging.info(f'Early stopping triggered after {self.epochs_since_improvement} epochs with no improvement.')
                        writer.add_scalar(f'{phase.capitalize()} Loss', epoch_loss, epoch)
                        writer.close()
                        return

            # Update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            else:
                logging.info('current lr: {}'.format(args.lr))

        writer.close()

    def nn_forward(self, inputs, ag, labels):
        logits = self.model(inputs, ag)
        logits_prob = self.sigmoid(logits)
        tmp_loss = self.criterion(logits, labels).item() * inputs.size(0)
        return logits_prob, tmp_loss
