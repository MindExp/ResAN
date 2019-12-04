from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from datasets.utils import dataset_loader
from models.init_model import InitModel
from utils import util, temporal_ensemble, inconsistency_target_sampler


# Training settings
class Solver(object):
    def __init__(self, config):
        self.source = config.source
        self.target = config.target
        self.num_classes = config.num_classes
        self.all_use = config.all_use
        self.backbone = config.backbone
        self.batch_size = config.batch_size
        self.checkpoint_dir = config.checkpoint_dir
        self.num_k = config.num_k
        self.lr = config.lr
        self.epoch_total = config.max_epoch
        self.optimizer = config.optimizer
        self.resume_epoch = config.resume_epoch
        self.batch_interval = config.batch_interval
        self.save_epoch_interval = config.save_epoch_interval
        self.use_abs_diff = config.use_abs_diff
        self.all_use = config.all_use
        self.ensemble_alpha = config.ensemble_alpha
        self.rampup_length = config.rampup_length
        self.grl_coefficient_upper = config.grl_coefficient_upper
        self.weight_consistency = config.weight_consistency
        self.weight_consistency_upper = config.weight_consistency_upper
        self.weight_discrepancy_upper = config.weight_discrepancy_upper
        self.mixup_beta = config.mixup_beta
        self.inconsistency_index_set = np.array([], dtype=np.int64)
        self.epoch_num, self.iter_num = 0, 0
        self.source_best_acc, self.target_best_acc = 0.0, 0.0

        print('{}_{} dataset loading...'.format(self.source, self.target))
        self.dataloader_source_train, self.dataloader_source_test = \
            dataset_loader.get_dataloader(domain=config.source, method=config.source_loader, config=config)
        self.dataloader_target_train, self.dataloader_target_test = \
            dataset_loader.get_dataloader(domain=config.target, method=config.target_loader, config=config)
        print('loading finished!')

        num_target_dataset_train = len(self.dataloader_target_train.dataset)
        self.current_output_t1, self.ensemble_output_t1, self.correction_output_t1 = \
            temporal_ensemble.init_variable_output_t(config, num_target_dataset_train)
        self.current_output_t2, self.ensemble_output_t2, self.correction_output_t2 = \
            temporal_ensemble.init_variable_output_t(config, num_target_dataset_train)

        model = InitModel().init(source_domain=self.source, target_domain=self.target)

        self.netF = model.get_netF(backbone=self.backbone)
        self.netC1 = model.get_netC(backbone=self.netF)
        self.netC2 = model.get_netC(backbone=self.netF)

        if config.resume_epoch or config.eval_only:
            self.netF = torch.load('%s/%s_to_%s_model_epoch%s_F.pt' %
                                   (self.checkpoint_dir, self.source, self.target, self.resume_epoch))
            self.netC1 = torch.load('%s/%s_to_%s_model_epoch%s_C1.pt' %
                                    (self.checkpoint_dir, self.source, self.target, self.resume_epoch))
            self.netC2 = torch.load('%s/%s_to_%s_model_epoch%s_C2.pt' %
                                    (self.checkpoint_dir, self.source, self.target, self.resume_epoch))

        self.netF.cuda()
        self.netC1.cuda()
        self.netC2.cuda()
        self.batch_interval = config.batch_interval

        self.opt_g, self.opt_c1, self.opt_c2 = None, None, None
        self.init_optimizer(which_opt=self.optimizer, lr=self.lr)
        self.lr = config.lr

    def init_optimizer(self, which_opt='sgd', lr=0.001, momentum=0.9):
        if which_opt == 'sgd':
            self.opt_g = optim.SGD(self.netF.get_parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_c1 = optim.SGD(self.netC1.get_parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_c2 = optim.SGD(self.netC2.get_parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.netF.get_parameters(), lr=lr, weight_decay=0.0005)
            self.opt_c1 = optim.Adam(self.netC1.get_parameters(), lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.netC2.get_parameters(), lr=lr, weight_decay=0.0005)

    # TODO: update optimizers dynamically --> FINISHED
    def update_optimizer(self):
        # TODO update gamma = 40
        gamma, power, weight_decay = 10, 0.75, 0.0005
        # optimizer=None, lr=0.01, iter_num=0, gamma=10, power=0.75, weight_decay=0.0005
        self.opt_g = util.update_optimizer(
            self.opt_g, lr=self.lr, epoch_num=self.epoch_num, epoch_total=self.epoch_total,
            gamma=gamma, power=power, weight_decay=weight_decay)
        self.opt_c1 = util.update_optimizer(
            self.opt_c1, lr=self.lr, epoch_num=self.epoch_num, epoch_total=self.epoch_total,
            gamma=gamma, power=power, weight_decay=weight_decay)
        self.opt_c2 = util.update_optimizer(
            self.opt_c2, lr=self.lr, epoch_num=self.epoch_num, epoch_total=self.epoch_total,
            gamma=gamma, power=power, weight_decay=weight_decay)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    # TODO: update for fair comparison. --> FINISHED
    def train(self, epoch, file_path_record=None):
        """
        Referenced from: https://github.com/mil-tokyo/adr_da
        :param epoch:
        :param file_path_record:
        :return:
        """
        criterion = nn.CrossEntropyLoss().cuda()
        self.netF.train()
        self.netC1.train()
        self.netC2.train()
        torch.cuda.manual_seed(1)

        # TODO update optimizer dynamically --> FINISHED
        # --> previous experimental results did not update optimizer in digit experiments
        office31_domains = ['A', 'W', 'D']
        if self.source in office31_domains and self.target in office31_domains:
            self.update_optimizer()

        iteration_source, iteration_target = len(self.dataloader_source_train), len(self.dataloader_target_train)
        iteration_total = iteration_source if iteration_source > iteration_target else iteration_target
        iter_dataloader_source_train, iter_dataloader_target_train = None, None

        for batch_idx in range(iteration_total):
            if batch_idx % iteration_source == 0:
                iter_dataloader_source_train = iter(self.dataloader_source_train)
            if batch_idx % iteration_target == 0:
                iter_dataloader_target_train = iter(self.dataloader_target_train)
            index_s, img_s, label_s = iter_dataloader_source_train.next()
            index_t, img_t, _ = iter_dataloader_target_train.next()

            img_s = img_s.cuda()
            img_t = img_t.cuda()
            imgs = torch.cat((img_s, img_t), 0)
            label_s = label_s.long().cuda()

            self.reset_grad()
            feat = self.netF(imgs)
            output = self.netC1(feat)
            output_s = output[:self.batch_size, :]
            loss_s1 = criterion(output_s, label_s)
            loss_s1.backward()
            self.opt_g.step()
            self.opt_c1.step()

            self.reset_grad()
            feat = self.netF(imgs)
            output = self.netC2(feat)
            output_s = output[:self.batch_size, :]
            loss_s2 = criterion(output_s, label_s)
            loss_s2.backward()
            self.opt_c2.step()
            self.reset_grad()

            feat = self.netF(imgs)

            output1 = self.netC1(feat)
            output1_s = output1[:self.batch_size, :]
            output1_t = output1[self.batch_size:, :]
            output1_t = F.softmax(output1_t)

            output2 = self.netC1(feat)
            output2_t = output2[self.batch_size:, :]
            output2_t = F.softmax(output2_t)
            loss = criterion(output1_s, label_s)
            loss_dis = util.adr_discrepancy(output1_t, output2_t)
            loss -= loss_dis
            loss.backward()
            self.opt_c1.step()
            self.reset_grad()

            for i in range(self.num_k):
                feat_t = self.netF(img_t)

                output1_t = self.netC1(feat_t)
                output2_t = self.netC1(feat_t)
                output1_t = F.softmax(output1_t)
                output2_t = F.softmax(output2_t)
                loss_dis = util.adr_discrepancy(output1_t, output2_t)
                G_loss = loss_dis
                G_loss.backward()
                self.opt_g.step()
                self.reset_grad()
            output = self.netF(img_s)
            output1_s = self.netC1(output)
            output2_s = self.netC1(output)
            output1_s = F.softmax(output1_s)
            output2_s = F.softmax(output2_s)

            output = self.netF(img_t)
            output1_t = self.netC1(output)
            output2_t = self.netC1(output)
            output1_t = F.softmax(output1_t)
            output2_t = F.softmax(output2_t)

            loss_dis = util.adr_discrepancy(output1_t, output2_t)
            entropy = util.entropy_loss(output1_t).detach()
            loss_dis = loss_dis.detach()
            loss_dis_s = util.adr_discrepancy(output1_s, output2_s)
            loss_dis_s = loss_dis_s.detach()

            if batch_idx % self.batch_interval == 0:
                record_info = 'Train Epoch:{} [batch_index:{}/{}]\tLoss1:{:.6f}\tDis:{:.6f}\tDis_s:{:.6f}'\
                    .format(epoch, batch_idx, iteration_total,
                            loss.data.item(), loss_dis.data.item(), loss_dis_s.data.item())
                print(record_info)
                if file_path_record:
                    util.record_log(file_path_record, record_info + '\n')
        return batch_idx

    # 应用 GRL 实现 one_step 训练
    def train_onestep(self, epoch, record_file_path=None):
        pass

    def test(self, epoch, record_file_path=None, save_model=False, test_domain='target'):
        self.netF.eval()
        self.netC1.eval()
        self.netC2.eval()
        test_loss_c1, test_loss_c2 = 0, 0
        correct_c1, correct_c2 = 0, 0
        correct_ensemble = 0
        size = 0
        if test_domain == 'target_testSet':
            dataloader_test = self.dataloader_target_test
        elif test_domain == 'source_testSet':
            dataloader_test = self.dataloader_source_test
        else:
            raise ValueError('Invalid test domain, expected to must be source or target testSet')

        for batch_idx, data in enumerate(dataloader_test):
            index, img, label = data
            img, label = img.cuda(), label.long().cuda()
            feat = self.netF(img)
            output_c1 = self.netC1(feat)
            output_c2 = self.netC2(feat)
            test_loss_c1 += F.nll_loss(output_c1, label).item()
            test_loss_c2 += F.nll_loss(output_c2, label).item()
            predict_c1 = output_c1.data.max(1)[1]
            predict_c2 = output_c2.data.max(1)[1]
            predict_ensemble = (output_c1 + output_c2).data.max(1)[1]
            k = label.data.size()[0]
            correct_c1 += predict_c1.eq(label.data).sum()
            correct_c2 += predict_c2.eq(label.data).sum()
            correct_ensemble += predict_ensemble.eq(label.data).sum()
            size += k
        test_loss_c1 = test_loss_c1 / size
        test_loss_c2 = test_loss_c2 / size

        current_ensemble_acc = float(correct_ensemble) / size

        if test_domain == 'source_testSet' and current_ensemble_acc >= self.source_best_acc:
            self.source_best_acc = current_ensemble_acc
        if test_domain == 'target_testSet' and save_model and current_ensemble_acc >= self.target_best_acc:
            self.target_best_acc = current_ensemble_acc
            torch.save(self.netF, '%s/%s_to_%s_model_F.pt' %
                       (self.checkpoint_dir, self.source, self.target))
            torch.save(self.netC1, '%s/%s_to_%s_model_C1.pt' %
                       (self.checkpoint_dir, self.source, self.target))
            torch.save(self.netC2, '%s/%s_to_%s_model_C2.pt' %
                       (self.checkpoint_dir, self.source, self.target))
        current_domain_best_acc = self.source_best_acc if test_domain == 'source_testSet' else self.target_best_acc
        if record_file_path:
            record_info = 'Test Epoch_{}:{}\tLoss1:{:.6f}, Loss2:{:.6f}, Current Best Accuracy:{:.4f}, ' \
                          'Accuracy C1:{:.4f}, Accuracy C2:{:.4f}, Accuracy Ensemble:{:.4f}\n'\
                .format(test_domain, epoch, test_loss_c1, test_loss_c2, current_domain_best_acc,
                        float(correct_c1) / size, float(correct_c2) / size, float(correct_ensemble) / size)
            print(record_info)
            util.record_log(record_file_path, record_info)
