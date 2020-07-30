import argparse
import os
from enum import Enum

import matplotlib.pyplot as plt

from exp_record import algorithms

dir_path_reproduction = os.path.join(os.getcwd(), 'REPRODUCTION')
dir_path_anon = os.getcwd()
process_multi_task_list = ['office_31']
digit_tasks = ['svhn_mnist', 'mnist_usps', 'usps_mnist', 'synsig_gtsrb']
office_31_tasks = algorithms.office_31_tasks


class AccType(Enum):
    acc_c1 = 'acc_c1'
    acc_c2 = 'acc_c2'
    acc_ensemble = 'acc_ensemble'


class DataGain(object):
    """
    单任务实验数据处理
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def data_gain_universal(self):
        with open(self.file_path, mode='r') as file:
            for line in file.readlines():
                print(line)


parser = argparse.ArgumentParser(description='PyTorch Implementation')
parser.add_argument('--algorithm', type=str, metavar='N', required=True,
                    help='process algorithm')
parser.add_argument('--process_task', type=str, metavar='N', required=True,
                    help='process task')
parser.add_argument('--specific_task', type=str, metavar='N', required=True,
                    help='specific task')
parser.add_argument('--all_use', action='store_true', default=False,
                    help='use all training data? ')
parser.add_argument('--supplementary_info', type=str, default=None, metavar='N',
                    help='supplementary information for current processing')

config = parser.parse_args()

algorithms_dict = {'DAN': algorithms.ProcessDAN, 'DANN': algorithms.ProcessDANN, 'JAN': algorithms.ProcessJAN,
                   'CDAN': algorithms.ProcessCDAN, 'ADR': algorithms.ProcessADR, 'MCD': algorithms.ProcessMCD,
                   'ResAN': algorithms.ProcessAnon}


def statistic_mean_std(algorithm=''):
    process_algorithm = algorithms_dict[algorithm](config=config)
    exp_setting_nums = 22 if algorithm == 'ResAN' else 1

    for dataset in algorithms.dataset_tasks_dict.keys():
        if dataset == 'digits' and algorithm not in ['ResAN', 'MCD']:
            continue
        process_algorithm.statistic_all_tasks_acc(dataset=dataset, exp_setting_nums=exp_setting_nums)


office_31_domain_dict = {'A': 'Amazon', 'D': 'DSLR', 'W': 'Webcam'}


def plot_acc_action(specific_task, algorithms_acc_dict):
    """
    绘制 Acc 收敛曲线
    :param specific_task: 任务名，i.e., A_W
    :param algorithms_acc_dict: 输入数据格式 {'algorithm_name': algorithm_acc_list}
    :return:
    """
    epoch_start, epoch_stop = 0, min([len(algorithm_acc) for algorithm_acc in algorithms_acc_dict.values()])
    x_epoch = list(range(epoch_start, epoch_stop))
    for algorithm_name in algorithms_acc_dict.keys():
        plt.plot(x_epoch, algorithms_acc_dict[algorithm_name][epoch_start: epoch_stop], label=algorithm_name)

    domain = specific_task.split('_')
    if specific_task in office_31_tasks:
        title_name = "{} to {}".format(office_31_domain_dict[domain[0]], office_31_domain_dict[domain[1]])
    elif specific_task in digit_tasks:
        title_name = "{} to {}".format(domain[0].upper(), domain[1].upper())
    else:
        title_name = specific_task
    plt.title(title_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.ylim(ymin=0.6)

    plt.legend()
    # plt.show()
    plt.savefig('.\\Visualization\{}.png'.format(specific_task), bbox_inches='tight')
    plt.close()


office_31_tasks_best_setting_dict = {'A_D': 6, 'A_W': 6, 'D_A': 2, 'D_W': 7, 'W_A': 8, 'W_D': 11}
digit_tasks_best_setting_dict = {'svhn_mnist_all_use': 7, 'mnist_usps_all_use': 7, 'mnist_usps_not_all': 10,
                                 'usps_mnist_not_all': 7, 'synsig_gtsrb_all_use': 7}


def plot_algorithms_acc_curve(specific_task='A_W'):
    """
    测试版本
    :return:
    """
    config.specific_task = specific_task
    process_algorithms_dict, algorithms_target_acc_dict = dict(), dict()
    process_algorithms_dict['ResAN'] = algorithms.ProcessAnon(config=config)
    process_algorithms_dict['MCD'] = algorithms.ProcessMCD(config=config)
    # process_algorithms_dict['CDAN'] = algorithms.ProcessCDAN(config=config)
    # process_algorithms_dict['JAN'] = algorithms.ProcessJAN(config=config)
    # process_algorithms_dict['DANN'] = algorithms.ProcessDANN(config=config)
    # process_algorithms_dict['DAN'] = algorithms.ProcessDAN(config=config)
    process_algorithms_dict['Source Only'] = algorithms.ProcessAnon(config=config)

    for process_algorithm in process_algorithms_dict.keys():
        if specific_task in office_31_tasks and process_algorithm == 'ResAN':
            exp_setting_num = office_31_tasks_best_setting_dict[specific_task]
        elif specific_task in digit_tasks and process_algorithm == 'ResAN':
            all_use_info = '_all_use' if config.all_use else '_not_all'
            best_setting_key = "{}{}".format(specific_task, all_use_info)
            exp_setting_num = digit_tasks_best_setting_dict[best_setting_key]
        else:
            exp_setting_num = 0

        process_algorithms_dict[process_algorithm].exp_setting_num = exp_setting_num
        target_acc_list = process_algorithms_dict[process_algorithm].get_acc_list()
        if process_algorithm in ['ResAN', 'MCD', 'Source Only']:
            target_acc_list = target_acc_list[AccType.acc_ensemble.value]
        else:
            target_acc_list = target_acc_list[1: -1]
        print("Plot algorithm:{}, experimental setting:{}.".format(process_algorithm, exp_setting_num))
        algorithms_target_acc_dict[process_algorithm] = target_acc_list

    plot_acc_action(specific_task=specific_task, algorithms_acc_dict=algorithms_target_acc_dict)


if __name__ == '__main__':
    statistic_mean_std(algorithm='ResAN')

    # for algorithm in algorithms_dict.keys():
    #     statistic_mean_std(algorithm=algorithm)
    #
    # for specific_task in office_31_tasks:
    #     plot_algorithms_acc_curve(specific_task=specific_task)

    # plot_algorithms_acc_curve(specific_task='svhn_mnist')
