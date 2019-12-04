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
        self.inconsistency_index_set = set()
        self.epoch_num = 0

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

        self.set_optimizer(which_opt=self.optimizer, lr=self.lr)
        self.lr = config.lr

    def set_optimizer(self, which_opt='sgd', lr=0.001, momentum=0.9):
        if which_opt == 'sgd':
            self.opt_g = optim.SGD(self.netF.get_parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_c1 = optim.SGD(self.netC1.get_parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_c2 = optim.SGD(self.netC2.get_parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.netF.get_parameters(), lr=lr, weight_decay=0.0005)
            self.opt_c1 = optim.Adam(self.netC1.get_parameters(), lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.netC2.get_parameters(), lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def train(self, epoch, file_path_record=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.netF.train()
        self.netC1.train()
        self.netC2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, (data_source, data_target) in enumerate(
                zip(self.dataloader_source_train, self.dataloader_target_train)):
            img_s, label_s = data_source
            img_t, _ = data_target
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = label_s.long().cuda()

            self.reset_grad()
            feat_s = self.netF(img_s)
            output_s1 = self.netC1(feat_s)
            output_s2 = self.netC2(feat_s)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()

            self.reset_grad()
            feat_s = self.netF(img_s)
            output_s1 = self.netC1(feat_s)
            output_s2 = self.netC2(feat_s)
            feat_t = self.netF(img_t)
            output_t1 = self.netC1(feat_t)
            output_t2 = self.netC2(feat_t)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_dis = util.loss_discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()

            for i in range(self.num_k):
                self.reset_grad()
                feat_t = self.netF(img_t)
                output_t1 = self.netC1(feat_t)
                output_t2 = self.netC2(feat_t)
                loss_dis = util.loss_discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()

            if batch_idx % self.batch_interval == 0:
                record_info = 'Train Epoch:{} [batch_index:{}]\tLoss1:{:.6f}\tLoss2:{:.6f}\tDiscrepancy:{:.6f}'.format(
                    epoch, batch_idx, loss_s1.data.item(), loss_s2.data.item(), loss_dis.data.item())
                print(record_info)
                if file_path_record:
                    util.record_log(file_path_record, record_info + '\n')
        return batch_idx

    # 应用 GRL 实现 one_step 训练
    def train_onestep(self, epoch, record_file_path=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.epoch_num += 1
        self.netF.train()
        self.netC1.train()
        self.netC2.train()
        torch.cuda.manual_seed(1)

        grl_coeff = util.calculate_grl_coefficient(epoch_num=self.epoch_num, epoch_total=self.epoch_total,
                                                   high=self.grl_coefficient_upper, low=0.0, alpha=10.0)

        weight_consistency = temporal_ensemble.get_current_consistency_weight(
            self.weight_consistency_upper, epoch, self.rampup_length)
        weight_consistency = torch.from_numpy(np.array([weight_consistency], dtype=np.float32)).cuda()
        weight_discrepancy_resample = temporal_ensemble.get_current_consistency_weight(
            self.weight_discrepancy_upper, epoch, self.rampup_length)
        weight_discrepancy_resample = torch.from_numpy(np.array([weight_discrepancy_resample], dtype=np.float32)).cuda()

        # TODO 研究点：重新设计 dataloader_target_resample， 采用样本 id 信息
        dataloader_target_resample = inconsistency_target_sampler.re_sample_inconsistency_target(
            list(self.inconsistency_index_set), self.dataloader_target_train, self.batch_size)

        iteration_source, iteration_target, iteration_resample = \
            len(self.dataloader_source_train), len(self.dataloader_target_train), len(dataloader_target_resample)
        iteration_total = iteration_source if iteration_source > iteration_target else iteration_target
        iter_dataloader_source_train, iter_dataloader_target_train, iter_dataloader_target_resample = None, None, None

        for batch_idx in range(iteration_total):
            if batch_idx % iteration_source == 0:
                iter_dataloader_source_train = iter(self.dataloader_source_train)
            if batch_idx % iteration_target == 0:
                iter_dataloader_target_train = iter(self.dataloader_target_train)
            resample_consideration = True
            if iteration_resample and batch_idx % iteration_resample == 0:
                iter_dataloader_target_resample = iter(dataloader_target_resample)

            index_s, img_s, label_s = iter_dataloader_source_train.next()
            index_t, img_t, _ = iter_dataloader_target_train.next()
            (index_r, img_r, _) = iter_dataloader_target_resample.next() if iteration_resample else (None, None, None)
            if iteration_resample == 0 or img_r.shape[0] == 1:
                resample_consideration = False

            img_s, img_t = img_s.cuda(), img_t.cuda()
            label_s = label_s.long().cuda()
            if img_t.shape[0] < self.batch_size:
                start, end = batch_idx % iteration_target * self.batch_size, \
                             batch_idx % iteration_target * self.batch_size + img_t.shape[0]
            else:
                start, end = batch_idx % iteration_target * self.batch_size, \
                             (batch_idx % iteration_target + 1) * self.batch_size
            correction_output_t1 = self.correction_output_t1[start: end]
            correction_output_t2 = self.correction_output_t2[start: end]

            self.reset_grad()
            feat_s = self.netF(img_s)
            output_s1 = self.netC1(feat_s)
            output_s2 = self.netC2(feat_s)
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_source = loss_s1 + loss_s2

            feat_t = self.netF(img_t)
            # TODO GRL方案选择 {method_1, method_2}
            output_t1 = self.netC1(feat_t, reverse=True, grl_coefficient=4.0)
            output_t2 = self.netC2(feat_t, reverse=True, grl_coefficient=4.0)

            inconsistency_index = torch.nonzero(
                torch.ne(output_t1.data.max(1)[1], output_t2.data.max(1)[1])).view(-1)
            consistency_index = torch.nonzero(
                torch.eq(output_t1.data.max(1)[1], output_t2.data.max(1)[1])).view(-1)
            # FIXME 直接使用以下策略效果也 ok, 但是部分逻辑问题
            self.inconsistency_index_set.update({start + i for i in inconsistency_index})
            self.inconsistency_index_set.difference_update({start + i for i in consistency_index})
            # TODO 研究点：自定义 discrepancy loss, e.g., symmetric_mse_loss.
            loss_discrepancy = util.loss_discrepancy(output_t1, output_t2)

            loss_consistency_output_t1 = temporal_ensemble.loss_softmax_mse(output_t1, correction_output_t1)
            loss_consistency_output_t2 = temporal_ensemble.loss_softmax_mse(output_t2, correction_output_t2)

            loss_discrepancy_resample = 0.0
            if resample_consideration:
                img_r = img_r.cuda()
                feat_r = self.netF(img_r)
                # TODO 研究点：再采样样本 grl_coefficient 待进一步研究
                output_r1 = self.netC1(feat_r, reverse=True, grl_coefficient=4.0)
                output_r2 = self.netC2(feat_r, reverse=True, grl_coefficient=4.0)
                loss_discrepancy_resample = util.loss_discrepancy(output_r1, output_r2)

            self.current_output_t1[start: end] = output_t1.data
            self.current_output_t2[start: end] = output_t2.data

            loss = loss_source - self.weight_consistency * loss_discrepancy + \
                   weight_consistency * (loss_consistency_output_t1 + loss_consistency_output_t2) + \
                   weight_discrepancy_resample * loss_discrepancy_resample
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_g.step()
            self.reset_grad()

            if batch_idx % self.batch_interval == 0:
                record_info = 'Train Epoch:{} [batch_index:{}/{}]\tLoss1:{:.6f}\tLoss2:{:.6f}\tDiscrepancy:{:.6f}'.format(
                    epoch + 1, batch_idx, iteration_total, loss_s1.data.item(), loss_s2.data.item(),
                    loss_discrepancy.data.item())
                train_info = '\tConsistency weight:{:.6f}\tConsistency loss_t1:{:.6f}\tConsistency loss_t2:{:.6f}' \
                             '\tDiscrepancy_Resample Weight:{:.6f}\tResample Discrepancy{:.6f}\tResample Pool Num:{}' \
                    .format(weight_consistency.item(),
                            loss_consistency_output_t1.data.item(),
                            loss_consistency_output_t2.data.item(),
                            weight_discrepancy_resample.item(),
                            loss_discrepancy_resample,
                            len(self.inconsistency_index_set))
                # string concatenate
                record_info = f'{record_info}{train_info}'
                print(record_info)
                if record_file_path:
                    util.record_log(record_file_path, record_info + '\n')

        self.ensemble_output_t1, self.correction_output_t1 = temporal_ensemble. \
            update_ensemble_and_correction_output_t(epoch, self.ensemble_alpha,
                                                    self.ensemble_output_t1,
                                                    self.current_output_t1)
        self.ensemble_output_t2, self.correction_output_t2 = temporal_ensemble. \
            update_ensemble_and_correction_output_t(epoch, self.ensemble_alpha,
                                                    self.ensemble_output_t2,
                                                    self.current_output_t2)

        return batch_idx

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

        if save_model and epoch % self.save_epoch_interval == 0:
            torch.save(self.netF, '%s/%s_to_%s_model_epoch%s_F.pt' %
                       (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.netC1, '%s/%s_to_%s_model_epoch%s_C1.pt' %
                       (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.netC2, '%s/%s_to_%s_model_epoch%s_C2.pt' %
                       (self.checkpoint_dir, self.source, self.target, epoch))
        if record_file_path:
            record_info = 'Test Epoch_{}:{}\tLoss1:{:.6f}, Loss2:{:.6f}, Accuracy C1:{:.4f}, Accuracy C2:{:.4f}, ' \
                          'Accuracy Ensemble:{:.4f}\n'.format(test_domain, epoch,
                                                              test_loss_c1, test_loss_c2,
                                                              float(correct_c1) / size, float(correct_c2) / size,
                                                              float(correct_ensemble) / size)
            print(record_info)
            util.record_log(record_file_path, record_info)
