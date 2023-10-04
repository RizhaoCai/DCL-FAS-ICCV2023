import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.utils import AverageMeter
from .network import build_net

pd.set_option('display.max_columns', None)

import logging

from data.transforms import VisualTransform, get_augmentation_transforms

from test import metric_report_from_dict
torch.cuda.empty_cache()
# try:
# from torch.utils.tensorboard import SummaryWriter
# except:
from tensorboardX import SummaryWriter

from FAS_DataManager.get_dataset import get_image_dataset_from_list
from .ewc import EWC
class Trainer():
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config):
        """
        Construct a new Trainer instance.
        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """

        self.config = config
        self.global_step = 1
        self.start_epoch = 1

        # Training control config
        self.epochs = self.config.TRAIN.EPOCHS
        self.batch_size = self.config.DATA.BATCH_SIZE

        self.counter = 0

        #  # Meanless at this version
        self.epochs = self.config.TRAIN.EPOCHS
        self.val_freq = config.TRAIN.VAL_FREQ
        #
        # # Network config

        self.val_metrcis = {
            'HTER@0.5': 1.0,
            'EER': 1.0,
            'MIN_HTER': 1.0,
            'AUC': 0
        }

        # Optimizer config
        self.momentum = self.config.TRAIN.MOMENTUM
        self.init_lr = self.config.TRAIN.INIT_LR
        self.lr_patience = self.config.TRAIN.LR_PATIENCE
        self.train_patience = self.config.TRAIN.PATIENCE

        self.network = build_net(self.config)
        if self.config.CUDA:
            # self.network = torch.nn.DataParallel(self.network )
            self.network.cuda()
        self.loss = torch.nn.CrossEntropyLoss().cuda()


    def get_dataset(self, data_file_list, transform):
        datasets = []
        for data_list_path in data_file_list:
            datasets.append(get_image_dataset_from_list(data_list_path, transform))
        return torch.utils.data.ConcatDataset(datasets)


    def get_dataloader(self, data_path, batch_size, num_workers, if_train=False):
        config = self.config

        if if_train:
            aug_transform = get_augmentation_transforms(config)
            data_transform = VisualTransform(config, aug_transform)
        else:
            data_transform = VisualTransform(config)

        dataset = self.get_dataset(data_path, data_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=num_workers,
                                                            shuffle=True, drop_last=if_train)

        return dataset, data_loader

    def train(self):

        protocol = self.config.TRAIN.PROTOCOL
        num_of_tasks = 10

        if protocol == 1:
            train_data_list = self.config.DATA.PROTOCOL1.TRAIN
            val_data_list = self.config.DATA.PROTOCOL1.VAL
            protocol_task_name = self.config.DATA.PROTOCOL1.TASK_NAME
        elif protocol==2:
            train_data_list = self.config.DATA.PROTOCOL2.TRAIN
            val_data_list = self.config.DATA.PROTOCOL2.VAL
            protocol_task_name = self.config.DATA.PROTOCOL2.TASK_NAME

        elif protocol == 105:
            train_data_list = self.config.DATA.PROTOCOL1.TRAIN_5SHOT
            val_data_list = self.config.DATA.PROTOCOL1.VAL
            protocol_task_name = self.config.DATA.PROTOCOL1.TASK_NAME
        elif protocol == 125:
            train_data_list = self.config.DATA.PROTOCOL1.TRAIN_25SHOT
            val_data_list = self.config.DATA.PROTOCOL1.VAL
            protocol_task_name = self.config.DATA.PROTOCOL1.TASK_NAME

        elif protocol == 205:
            train_data_list = self.config.DATA.PROTOCOL2.TRAIN_5SHOT
            val_data_list = self.config.DATA.PROTOCOL2.VAL
            protocol_task_name = self.config.DATA.PROTOCOL2.TASK_NAME
        elif protocol == 225:
            train_data_list = self.config.DATA.PROTOCOL2.TRAIN_25SHOT
            val_data_list = self.config.DATA.PROTOCOL2.VAL
            protocol_task_name = self.config.DATA.PROTOCOL2.TASK_NAME

        self.ckpt_list_of_previous_tasks = []

        ewc_lambda = self.config.TRAIN.EWC_LAMBDA
        if ewc_lambda > 0:
            ewc_onlne = self.config.TRAIN.EWC_ONLINE
            ewc_regularizer = EWC(ewc_lambda=ewc_lambda, if_online=ewc_onlne)
        else:
            ewc_regularizer = None

        self.init_weight(self.config.TRAIN.INIT)
        self.ckpt_list_of_previous_tasks.append(self.config.TRAIN.INIT)
        for task_id in range(1, num_of_tasks+1):

            # 0. task
            task_name = protocol_task_name[task_id-1]
            # 1. Set up data
            train_batch_size = self.config.DATA.BATCH_SIZE
            train_num_workers = self.config.DATA.NUM_WORKERS
            test_batch_size = self.config.DATA.VAL_BATCH_SIZE
            val_num_workers = self.config.DATA.VAL_NUM_WORKERS
            train_data_path = train_data_list[0:task_id] # TODO
            #train_data_path = train_data_list[task_id-1]  # TODO
            val_data_path = val_data_list[task_id-1]

            train_dataset, train_data_loader = self.get_dataloader(train_data_path, train_batch_size, train_num_workers, if_train=True)
            val_dataset, val_data_loader = self.get_dataloader([val_data_path], test_batch_size, val_num_workers, if_train=False)


            if self.config.TRAIN.OPTIM.TYPE == 'SGD':
                logging.info('Setting: Using SGD Optimizer')
                self.optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, self.network.parameters()),
                    lr=self.init_lr,
                )

            elif self.config.TRAIN.OPTIM.TYPE == 'Adam':
                logging.info('Setting: Using Adam Optimizer')


                self.optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, self.network.parameters()),
                    lr=self.init_lr,
                )
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.TRAIN.EPOCHS)

            # Set up data

            tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

            self.num_train = len(train_data_loader) * self.config.DATA.BATCH_SIZE
            self.num_valid = len(val_data_loader) * self.config.DATA.BATCH_SIZE
            logging.info("\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid))


            logging.info('Start task: {} {}'.format(task_id, task_name))

            consolidate = True
            if consolidate and task_id < len(train_data_list) and ewc_regularizer is not None:
                # estimate the fisher information of the parameters and consolidate
                # them in the network.
                logging.info(
                    '=> Estimating diagonals of the fisher information matrix...'
                )
                fisher_estimation_sample_size = 25
                ewc_regularizer.update_fisher_optpar(self.network, task_id, train_dataset,
                                                     sample_size=fisher_estimation_sample_size, consolidate=consolidate)
                logging.info(' Done!')

            self.train_a_task( self.network, ewc_regularizer, train_data_loader, val_data_loader, task_id, task_name)
            self.val_metrcis['AUC'] = -1


            best_ckpt_for_last_task = self.ckpt_list_of_previous_tasks[task_id]
            self.init_weight(best_ckpt_for_last_task)



    def train_a_task(self, network, ewc_regularizer, train_data_loader, val_data_loader, task_id, task_name):


        # Set up optimizer
        ckpt_path_to_save = os.path.join(self.config.OUTPUT_DIR, 'task_{}_{}'.format(task_id, task_name), 'ckpt',
                                        'best.ckpt')
        logging.info('Ckpt to save {}'.format(ckpt_path_to_save))
        self.ckpt_list_of_previous_tasks.append(ckpt_path_to_save)
        for epoch in range(self.start_epoch, self.epochs + 1):

            if self.tensorboard:
                self.tensorboard.add_scalar('lr', self.init_lr, self.global_step)
            logging.info('\nEpoch: {}/{} - LR: {:.6f}'.format(
                epoch, self.epochs, self.init_lr))

            # train for 1 epoch
            train_loss = AverageMeter()
            self.network.train()
            num_train = len(train_data_loader) * self.batch_size

            with tqdm(total=num_train, ncols=80) as pbar:
                for i, batch_data in enumerate(train_data_loader):


                    network_input, target = batch_data[1], batch_data[2]

                    cls_out = self.inference(network_input.cuda())
                    # compute losses for differentiable modules

                    ce_loss = self._total_loss_caculation(cls_out, target)



                    if ewc_regularizer is None:
                        ewc_loss = torch.tensor(0.0, device=ce_loss.device)
                    else:
                        ewc_loss = ewc_regularizer.regularize(network.named_parameters())

                    #import pdb; pdb.set_trace()
                    loss = ce_loss + ewc_loss

                    pbar.set_description(
                        (
                            " loss={:.4f}|ce_loss={:.4f}|ewc_loss={:.4f} ".format(loss.item(), ce_loss.item(), ewc_loss.item(),
                                                         )
                        )
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                    pbar.update(self.batch_size)
                    train_loss.update(loss.item(), self.batch_size)
                    # log to tensorboard
                    if self.tensorboard:
                        self.tensorboard.add_scalar('loss/train_total', loss.item(), self.global_step)

                    self.global_step += 1

                self.lr_scheduler.step()

            train_loss_avg = train_loss.avg

            logging.info("Avg Training loss = {} ".format(str(train_loss_avg)))
            # evaluate on validation set'


            if self.config.TRAIN.WITH_VAL and (epoch+1) % self.val_freq == 0:
                with torch.no_grad():
                    val_output = self.validate(epoch, val_data_loader)


                if val_output['AUC'] > self.val_metrcis['AUC']:
                    logging.info("Save models")
                    self.val_metrcis['MIN_HTER'] = val_output['MIN_HTER']
                    self.val_metrcis['AUC'] = val_output['AUC']
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': network.state_dict(),
                         'optim_state': self.optimizer.state_dict(),
                         'previous_task_ckpt': self.ckpt_list_of_previous_tasks,
                         'ewc_regularizer': ewc_regularizer
                         },
                        ckpt_path_to_save = ckpt_path_to_save

                    )

                logging.info('Current Best MIN_HTER={}%, AUC={}%'.format(100 * self.val_metrcis['MIN_HTER'],
                                                                      100 * self.val_metrcis['AUC']))
                if epoch > 20 and train_loss.avg < 0.00001:
                    logging.info("Early stop since Avg Training loss<0.00001. ".format(str(train_loss_avg)))
                    return
            if not self.config.TRAIN.WITH_VAL and epoch==100:
                self.save_checkpoint(
                    {'epoch': epoch,
                     'val_metrics': self.val_metrcis,
                     'global_step': self.global_step,
                     'model_state': network.state_dict(),
                     'optim_state': self.optimizer.state_dict(),
                     'previous_task_ckpt': self.ckpt_list_of_previous_tasks,
                     'ewc_regularizer': ewc_regularizer
                     },
                    ckpt_path_to_save=ckpt_path_to_save

                )
                logging.info('Stop at 100th epoch. Save checkpoint')


    def validate(self, epoch, val_data_loader):
        val_results = self.test(val_data_loader)
        val_loss = val_results['avg_loss']
        scores_gt_dict = val_results['scores_gt']
        scores_pred_dict = val_results['scores_pred']
        # log to tensorboard
        if self.tensorboard:
            self.tensorboard.add_scalar('loss/val_total', val_loss, self.global_step)

        frame_metric_dict, video_metric_dict = metric_report_from_dict(scores_pred_dict, scores_gt_dict, 0.5)
        df_frame = pd.DataFrame(frame_metric_dict, index=[0])
        df_video = pd.DataFrame(video_metric_dict, index=[0])

        logging.info("Frame level metrics: \n" + str(df_frame))
        logging.info("Video level metrics: \n" + str(df_video))

        return frame_metric_dict

    def test(self, test_data_loader):
        if not hasattr(self, 'network'):
            self.network = build_net(self.config)
            self.init_weight(self.config.TEST.CKPT)
            self.network.cuda()

        if test_data_loader is None:
            data_path = self.config.DATA.TEST
            batch_size = self.config.TEST.BATCH_SIZE
            num_workers = self.config.DATA.NUM_WORKERS
            dataset, test_data_loader = self.get_dataloader(data_path, batch_size, num_workers, if_train=False)
        avg_test_loss = AverageMeter()
        scores_pred_dict = {}
        spoofing_label_gt_dict = {}
        self.network.eval()

        with torch.no_grad():
            for data in tqdm(test_data_loader, ncols=80):
                network_input, target, video_ids = data[1], data[2], data[3]

                output_prob = self.inference(network_input.cuda())
                test_loss = self._total_loss_caculation(output_prob, target)
                pred_score = self._get_score_from_prob(output_prob)
                avg_test_loss.update(test_loss.item(), network_input.size()[0])

                gt_dict, pred_dict = self._collect_scores_from_loader(spoofing_label_gt_dict, scores_pred_dict,
                                                                      target['spoofing_label'].numpy(), pred_score,
                                                                      video_ids
                                                                      )

        test_results = {
            'scores_gt': gt_dict,
            'scores_pred': pred_dict,
            'avg_loss': avg_test_loss.avg,
        }
        return test_results

    def _collect_scores_from_loader(self, gt_dict, pred_dict, ground_truths, pred_scores, video_ids):
        batch_size = ground_truths.shape[0]

        for i in range(batch_size):
            video_name = video_ids[i]
            if video_name not in pred_dict.keys():
                pred_dict[video_name] = list()
            if video_name not in gt_dict.keys():
                gt_dict[video_name] = list()

            pred_dict[video_name] = np.append(pred_dict[video_name], pred_scores[i])
            gt_dict[video_name] = np.append(gt_dict[video_name], ground_truths[i])

        return gt_dict, pred_dict

    def save_checkpoint(self, state, ckpt_path_to_save=None):

        os.makedirs(os.path.dirname(ckpt_path_to_save), exist_ok=True)

        logging.info("[*] Saving model to {}".format(ckpt_path_to_save))
        torch.save(state, ckpt_path_to_save)

    def load_checkpoint(self, ckpt_path):

        logging.info("[*] Loading checkpoint from {}".format(ckpt_path))

        ckpt = torch.load(ckpt_path)
        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.valid_metric = ckpt['val_metrics']

        self.network.load_state_dict(ckpt['model_state'])

        if hasattr(self, 'optimizer') and 'optim_state' in ckpt.keys:
            self.optimizer.load_state_dict(ckpt['optim_state'])

        logging.info(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                ckpt_path, ckpt['epoch'])
        )

    def init_weight(self, ckpt_path):
        logging.info("[*] Initialize model weight from {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        # self.best_valid_acc = ckpt['best_valid_acc']
        self.network.load_state_dict(ckpt['model_state'])


    def inference(self, *args, **kargs):
        """
            Input images
            Output prob and scores
        """
        output_prob = self.network(*args, **kargs)  # By default: a binary classifier network
        return output_prob

    def _total_loss_caculation(self, output_prob, target):
        spoofing_label = target['spoofing_label'].cuda()
        return self.loss(output_prob, spoofing_label)

    def _get_score_from_prob(self, output_prob):
        output_scores = torch.softmax(output_prob, 1)
        output_scores = 1-output_scores.cpu().numpy()[:, 0] # The probability to be spoofing
        return output_scores

    def load_batch_data(self):
        pass
