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


'''
class Process(object):
    def __init__(self, config):
        self.process_task = config.process_task
        self.all_use = config.all_use

    def get_specific_task_list(self):
        """

        :return:
        """
        specific_task_list = [self.process_task] if self.process_task in process_single_task_list\
            else office_31_tasks if self.process_task == 'office_31' else None
        if not specific_task_list:
            raise ValueError('invalid specific_task_list!')

        return specific_task_list

    def get_exp_dir_path(self, algorithm):
        """
        获取当前任务实验记录路径
        :return:
        """
        process_tasks = ['svhn_mnist', 'mnist_usps', 'usps_mnist', 'synsig_gtsrb', 'office_31']
        specific_tasks = ['mnist_usps', 'usps_mnist']
        dir_path_reproduction_mcd = os.path.join(dir_path_reproduction, 'MCD')
        dir_path = dir_path_anon if algorithm == 'Anon' else dir_path_reproduction_mcd if algorithm == 'MCD' else ''

        if self.process_task in process_tasks:
            if self.process_task in specific_tasks:
                sub_file_name = 'mnist_usps'
            else:
                sub_file_name = self.process_task
        else:
            raise ValueError('invalid process task parameters!')

        if sub_file_name in specific_tasks:
            append_file_name = '_all_use' if self.all_use else '_not_all'
            sub_file_name = sub_file_name + append_file_name
            exp_dir_path = os.path.join(dir_path, sub_file_name)
        else:
            exp_dir_path = os.path.join(dir_path, sub_file_name)

        print('processing:', exp_dir_path)

        return exp_dir_path

    def filter_train_and_test_file_list(self, file_list, specific_task=''):
        """
        文件过滤
        :param file_list:
        :param specific_task:
        :return:
        """
        all_use_type = 'alluse_True' if self.all_use else 'alluse_False'
        train_file_list = sorted(list(filter(
            lambda file_name: ('train' in file_name) and (specific_task in file_name) and (all_use_type in file_name),
            file_list)))
        test_file_list = sorted(list(filter(
            lambda file_name: ('test' in file_name) and (specific_task in file_name) and (all_use_type in file_name),
            file_list)))
        return train_file_list, test_file_list

    def get_exp_train_test_path_list(self, algorithm='anon', specific_task=''):
        """
        获取单任务多参数配置下实验记录文件路径列表
        :return:
        """
        dir_path = self.get_exp_dir_path(algorithm=algorithm)
        file_list = os.listdir(dir_path)
        train_file_list, test_file_list = self.filter_train_and_test_file_list(
            file_list=file_list, specific_task=specific_task)
        train_file_path_list = list(map(lambda file_name: os.path.join(dir_path, file_name), train_file_list))
        test_file_path_list = list(map(lambda file_name: os.path.join(dir_path, file_name), test_file_list))

        return train_file_path_list, test_file_path_list

    @staticmethod
    def process_hyper_parameters_analysis(test_file_list):
        """
        单任务超参数分析
        :param test_file_list:
        :return:
        """

        return

    @staticmethod
    def get_mean_std(acc_list, acc_type, last_n=5):
        """
        计算均值与方差
        :param acc_list:
        :param acc_type:
        :param last_n:
        :return:
        """
        calculate_data = np.asarray(acc_list[acc_type][-1: -(last_n + 1): -1])
        mean = round(float(np.mean(calculate_data)), 4)
        std = round(float(np.std(calculate_data)), 5)
        return mean, std

    def statistic_acc(self, file_path_list):
        """

        :param file_path_list:
        :return:
        """
        acc_mean_list = []
        for test_file_path in file_path_list:
            source_test_acc_list, target_test_acc_list = self.get_exp_acc_list(test_file_path)
            source_mean_std = self.get_mean_std(
                acc_list=source_test_acc_list, acc_type=AccType.acc_ensemble.value, last_n=5)
            target_mean_std = self.get_mean_std(
                acc_list=target_test_acc_list, acc_type=AccType.acc_ensemble.value, last_n=5)

            print('source: {}, target: {}'.format(source_mean_std, target_mean_std))
            acc_mean_list.append((target_mean_std, test_file_path))
        print('best target acc:{}\n'.format(max(acc_mean_list) if acc_mean_list else 'To be ignored!'))

    def get_exp_acc_list(self, file_path):
        """
        获取单任务单设置下 ACC
        :param file_path:
        :return:
        """
        print('processing:', file_path)
        source_test_acc_list, target_test_acc_list = \
            dict({'acc_c1': list(), 'acc_c2': list(), 'acc_ensemble': list()}), \
            dict({'acc_c1': list(), 'acc_c2': list(), 'acc_ensemble': list()})
        with open(file_path, 'r') as file:
            file.__next__()
            line_counter = 1
            for line in file.readlines():
                separated_list = line.split(sep=':')
                split_index = 5
                current_acc_c1 = separated_list[split_index].split(sep=',')[0]
                current_acc_c2 = separated_list[split_index + 1].split(sep=',')[0]
                current_acc_ensemble = separated_list[split_index + 2].split(sep='\n')[0]

                if line_counter % 2 == 1:
                    nicked_name_dict = target_test_acc_list
                else:
                    nicked_name_dict = source_test_acc_list

                nicked_name_dict['acc_c1'].append(float(current_acc_c1))
                nicked_name_dict['acc_c2'].append(float(current_acc_c2))
                nicked_name_dict['acc_ensemble'].append(float(current_acc_ensemble))

                line_counter = line_counter + 1

        return source_test_acc_list, target_test_acc_list

    def process_acc_curves(self, specific_task, algorithms_acc_dict):
        """
        绘制不同算法 ACC 曲线
        :param specific_task:
        :param algorithms_acc_dict:
        :return:
        """
        epoch_start, epoch_stop = 0, min([len(algorithm_acc) for algorithm_acc in algorithms_acc_dict.values()])
        x_epoch = np.linspace(start=epoch_start, stop=epoch_stop)
        for key in algorithms_acc_dict.keys():
            plt.plot(x_epoch, algorithms_acc_dict[key][x_epoch], label=key)

        plt.title(specific_task)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.legend()
        plt.show()

    def process_misclassified_samples_curves(self, algorithms_misclassified_samples_lists):
        pass
'''

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

'''
1. Acc 收敛对比曲线
2. Misclassified samples curves
3. T-SNE 可视化
'''

'''
def main_anon():
    process_task_list = process_single_task_list
    for process_task in process_task_list:
        config.process_task = process_task
        process = Process(config=config)
        specific_task_list = process.get_specific_task_list()
        for specific_task in specific_task_list:
            train_file_path_list, test_file_path_list = process.get_exp_train_test_path_list(
                algorithm=config.algorithm, specific_task=specific_task)
            process.statistic_acc(file_path_list=test_file_path_list)

'''

algorithms_dict = {'DAN': algorithms.ProcessDAN, 'DANN': algorithms.ProcessDANN, 'JAN': algorithms.ProcessJAN,
                   'CDAN': algorithms.ProcessCDAN, 'ADR': algorithms.ProcessADR, 'MCD': algorithms.ProcessMCD,
                   'Anon': algorithms.ProcessAnon}


def statistic_mean_std(algorithm=''):
    process_algorithm = algorithms_dict[algorithm](config=config)
    exp_setting_nums = 12 if algorithm == 'Anon' else 1

    for dataset in algorithms.dataset_tasks_dict.keys():
        if dataset == 'digits' and algorithm not in ['Anon', 'MCD']:
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
    process_algorithms_dict['Anon'] = algorithms.ProcessAnon(config=config)
    process_algorithms_dict['MCD'] = algorithms.ProcessMCD(config=config)
    process_algorithms_dict['CDAN'] = algorithms.ProcessCDAN(config=config)
    process_algorithms_dict['JAN'] = algorithms.ProcessJAN(config=config)
    process_algorithms_dict['DANN'] = algorithms.ProcessDANN(config=config)
    process_algorithms_dict['DAN'] = algorithms.ProcessDAN(config=config)
    # process_algorithms_dict['Source Only'] = algorithms.ProcessAnon(config=config)

    for process_algorithm in process_algorithms_dict.keys():
        if specific_task in office_31_tasks and process_algorithm == 'Anon':
            exp_setting_num = office_31_tasks_best_setting_dict[specific_task]
        elif specific_task in digit_tasks and process_algorithm == 'Anon':
            all_use_info = '_all_use' if config.all_use else '_not_all'
            best_setting_key = "{}{}".format(specific_task, all_use_info)
            exp_setting_num = digit_tasks_best_setting_dict[best_setting_key]
        else:
            exp_setting_num = 0

        process_algorithms_dict[process_algorithm].exp_setting_num = exp_setting_num
        target_acc_list = process_algorithms_dict[process_algorithm].get_acc_list()
        if process_algorithm in ['Anon', 'MCD', 'Source Only']:
            target_acc_list = target_acc_list[AccType.acc_ensemble.value]
        else:
            target_acc_list = target_acc_list[1: -1]
        print("Plot algorithm:{}, experimental setting:{}.".format(process_algorithm, exp_setting_num))
        algorithms_target_acc_dict[process_algorithm] = target_acc_list

    plot_acc_action(specific_task=specific_task, algorithms_acc_dict=algorithms_target_acc_dict)


if __name__ == '__main__':
    statistic_mean_std(algorithm='DAN')

    # for algorithm in algorithms_dict.keys():
    #     statistic_mean_std(algorithm=algorithm)
    #
    # for specific_task in office_31_tasks:
    #     plot_algorithms_acc_curve(specific_task=specific_task)

    # plot_algorithms_acc_curve(specific_task='D_W')
