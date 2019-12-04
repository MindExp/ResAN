from __future__ import print_function

import argparse

import torch
import os
import torch.backends.cudnn as cudnn

from solver_fixed import Solver
from utils import util

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation')
parser.add_argument('--source', type=str, metavar='N', required=True,
                    help='source dataset')
parser.add_argument('--target', type=str, metavar='N', required=True,
                    help='target dataset')
parser.add_argument('--source_loader', type=str, metavar='N', required=True,
                    help='source dataset loader')
parser.add_argument('--target_loader', type=str, metavar='N', required=True,
                    help='target dataset loader')
parser.add_argument('--num_classes', type=int, metavar='N', required=True,
                    help='number of classes in source domain')
parser.add_argument('--all_use', action='store_true', default=False,
                    help='use all training data? ')
parser.add_argument('--backbone', type=str, default='default', metavar='N',
                    help='base network')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--image_size', type=int, default=32, metavar='N',
                    help='input image size for training (default: 32)')
parser.add_argument('--checkpoint', type=str, default='checkpoint', metavar='N',
                    help='checkpoint directory')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update, '
                         'only used when the model is trained by multi_step iterating')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=160, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N',
                    help='model optimizer')
parser.add_argument('--resume_epoch', type=int, default=0, metavar='N',
                    help='epoch to resume')
parser.add_argument('--batch_interval', type=int, default=10, metavar='N',
                    help='batch interval to record')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--save_epoch_interval', type=int, default=20, metavar='N',
                    help='epoch interval to save model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
parser.add_argument('--ensemble_alpha', type=float, metavar='N', required=True,
                    help='initial weight of ensemble output')
parser.add_argument('--rampup_length', type=int, default=80, metavar='N',
                    help='ramp-up length of weight consistency loss')
parser.add_argument('--grl_coefficient_upper', type=float, default=1.0, metavar='N',
                    help='gradient reverse layer coefficient upper bound')
parser.add_argument('--weight_consistency', type=float, default=1.0, metavar='N',
                    help='weight of consistency loss upper bound')
parser.add_argument('--weight_consistency_upper', type=float, default=1.0, metavar='N',
                    help='weight of consistency loss upper bound')
parser.add_argument('--weight_discrepancy_upper', type=float, default=1.0, metavar='N',
                    help='weight of discrepancy loss upper bound')
parser.add_argument('--mixup_beta', type=float, metavar='N', required=True,
                    help='initial weight of ensemble output')
parser.add_argument('--supplementary_info', type=str, default=None, metavar='N',
                    help='supplementary information for this algorithm')

config = parser.parse_args()
config.cuda = not config.no_cuda and torch.cuda.is_available()
torch.manual_seed(config.seed)
if config.cuda:
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True


def main():
    record_directory, record_train_file_path, record_test_file_path = util.init_record_file_name(config)
    config.checkpoint_dir = os.path.join(record_directory, config.checkpoint)
    util.record_log(record_train_file_path, '{}\n'.format(config))
    util.record_log(record_test_file_path, '{}\n'.format(config))

    solver = Solver(config)

    if config.eval_only:
        solver.test(0)
    else:
        iteration = 0
        for epoch in (range(config.max_epoch) if not config.resume_epoch else range(
                config.resume_epoch, config.max_epoch)):
            if config.one_step:
                num = solver.train_onestep(epoch, record_file_path=record_train_file_path)
            else:
                num = solver.train(epoch, file_path_record=record_train_file_path)
            iteration += num
            if epoch % 1 == 0:
                solver.test(epoch + 1, record_file_path=record_test_file_path, save_model=config.save_model,
                            test_domain='target_testSet')
                solver.test(epoch + 1, record_file_path=record_test_file_path, save_model=False,
                            test_domain='source_testSet')


if __name__ == '__main__':
    source_domains, target_domains = ['W', 'D'], ['W', 'D']
    for source_domain in source_domains:
        for target_domain in target_domains:
            config.source, config.target = source_domain, target_domain
            if config.source == config.target:
                continue
            if (config.source == 'A' and config.target == 'D') or (config.source == 'D' and config.target == 'W'):
                config.lr = 0.0003
            else:
                config.lr = 0.001
            weight_consistencies, weight_consistency_uppers, weight_discrepancy_uppers = [0.0, 0.1, 1.0], [0.0, 1.0], [
                0.0, 1.0]
            for weight_consistency in weight_consistencies:
                for weight_consistency_upper in weight_consistency_uppers:
                    for weight_discrepancy_upper in weight_discrepancy_uppers:
                        config.weight_consistency, config.weight_consistency_upper, config.weight_discrepancy_upper = \
                            weight_consistency, weight_consistency_upper, weight_discrepancy_upper
                        main()
    # weight_consistencies, weight_consistency_uppers, weight_discrepancy_uppers = 1.0, 1.0, 1.0
    # main()
