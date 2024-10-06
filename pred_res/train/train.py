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
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from pred_res.dataset import ECGDataset
from pred_res.model import resnet18
from pred_res.utils.freeze import set_freeze_by_id


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
        self.cal_acc = None

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
        self.model = resnet18(pretrained=True, in_channel=ECGDataset.input_channel, out_channel=self.num_classes)
        if args.layers_num_last is not None:
            if args.layers_num_last != 0:
                set_freeze_by_id(self.model, args.layers_num_last)
        if self.device_count > 1:
            self.model = nn.DataParallel(self.model)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                    weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        self.model.to(self.device)

    def train(self):
        """
        Train the model
        """
        global labels_all, logits_prob_all, logits_prob, loss_temp, loss, best_acc
        args = self.args

        step = 0
        step_start = time.time()

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_loss = 0.0
                batch_length = 0
                batch_count = 0
                batch_loss = 0.0
                # Iterate over data.
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # zero the parameter gradients
                    if phase == 'train':
                        logits = self.model(inputs)
                        logits_prob = self.sigmoid(logits)

                        if batch_idx == 0:
                            labels_all = labels
                            logits_prob_all = logits_prob
                        else:
                            labels_all = torch.cat((labels_all, labels), dim=0)
                            logits_prob_all = torch.cat((logits_prob_all, logits_prob), dim=0)

                        loss = self.criterion(logits, labels)
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                    else:
                        val_length = inputs.shape[2]
                        win_length = args.win_length if args.win_length else 4096
                        overlap = args.overlap if args.overlap else 256
                        patch_number = math.ceil(abs(val_length - win_length) / (win_length - overlap)) + 1
                        if patch_number > 1:
                            start = int((val_length - win_length) / (patch_number - 1))
                        else:
                            start = 0
                        for i in range(patch_number):
                            if i == 0:
                                logits_prob, loss_temp = self.nn_forward(inputs[:, :, start:start + win_length],
                                                                         labels)
                            elif i == patch_number - 1:
                                logits_prob_tmp, loss_temp_tmp = self.nn_forward(
                                    inputs[:, :, start + win_length:],
                                    labels)
                                logits_prob = (logits_prob + logits_prob_tmp) / patch_number
                                loss_temp = (loss_temp + loss_temp_tmp) / patch_number
                            else:
                                logits_prob_tmp, loss_temp_tmp = self.nn_forward(
                                    inputs[:, :, start + win_length:start + win_length + overlap],
                                    labels)
                                logits_prob = logits_prob + logits_prob_tmp
                                loss_temp = loss_temp + loss_temp_tmp
                            epoch_loss += loss_temp

                        if batch_idx == 0:
                            labels_all = labels
                            logits_prob_all = logits_prob
                        else:
                            labels_all = torch.cat((labels_all, labels), dim=0)
                            logits_prob_all = torch.cat((logits_prob_all, logits_prob), dim=0)
                        epoch_loss += loss_temp

                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_count += inputs.size(0)
                            batch_length += 1
                            if step % args.log_step == 0:
                                train_time = time.time() - step_start
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                logging.info('Epoch[{}/{}], Step[{}/{}], loss: {:.4f},'
                                             'lr: {:.1f} examples/sec {:.2f} sec/batch'
                                             .format(epoch, args.max_epoch, step,
                                                     args.max_epoch * len(self.dataloaders[phase].dataset),
                                                     batch_loss / batch_count, 1.0 * batch_count / train_time,
                                                     batch_time))
                                batch_loss = 0.0
                                batch_count = 0
                    step += 1

                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                logging.info('Epoch[{}/{}], Loss: {:.4f},'
                             'lr: {:.1f} examples/sec {:.2f} sec/batch'
                             .format(epoch, args.max_epoch, epoch_loss,
                                     1.0 * len(self.dataloaders[phase].dataset) / epoch_loss,
                                     1.0 * len(self.dataloaders[phase].dataset) / epoch_loss,
                                     1.0 * len(self.dataloaders[phase].dataset) / epoch_loss))

                epoch_acc = self.cal_acc(logits_prob_all, labels_all, threshold=0.5, num_classes=self.num_classes)

                if phase == 'val':
                    model_state = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model_state,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def nn_forward(self, inputs, labels):
        logits = self.model(inputs)
        logits_prob = self.sigmoid(logits)
        tmp_loss = self.criterion(logits, labels).item() * inputs.size(0)
        return logits_prob, tmp_loss
