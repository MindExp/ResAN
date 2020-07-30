import os
from enum import Enum

import numpy as np

dir_path_reproduction = os.path.join(os.getcwd(), 'REPRODUCTION')
office_31_tasks = ['A_D', 'A_W', 'D_A', 'D_W', 'W_A', 'W_D']
digits_tasks = ['svhn_mnist', 'mnist_usps', 'usps_mnist', 'synsig_gtsrb']
dataset_tasks_dict = {'digits': digits_tasks, 'office_31': office_31_tasks}


class AccType(Enum):
    acc_c1 = 'acc_c1'
    acc_c2 = 'acc_c2'
    acc_ensemble = 'acc_ensemble'


class ProcessAlgorithm(object):
    def __init__(self, config):
        self.specific_task = config.specific_task
        self.dir_path = os.path.join(dir_path_reproduction, 'CDAN')

    def get_acc_list(self):
        """
        获取单任务单设置下 ACC
        :return:
        """
        pass

    @staticmethod
    def get_mean_std(acc_list, last_n=5):
        """
        计算均值与方差
        :param acc_list:
        :param last_n:
        :return:
        """
        calculate_data = np.asarray(acc_list[-1: -(last_n + 1): -1])
        mean = round(float(np.mean(calculate_data)), 4)
        std = round(float(np.std(calculate_data)), 5)
        return mean, std

    def statistic_specific_task_acc(self, specific_task):
        """
        统计单任务均值与方差
        :param specific_task:
        :return:
        """
        assert specific_task in office_31_tasks

        self.specific_task = specific_task
        acc_list = self.get_acc_list()
        target_test_mean_std = self.get_mean_std(acc_list, last_n=5)

        print("target domain:{}".format(target_test_mean_std))
        return target_test_mean_std

    def statistic_all_tasks_acc(self, dataset='office_31', exp_setting_nums=0):
        """
        统计 Office_31 数据集所有任务对应均值与方差
        :return:
        """
        process_tasks = dataset_tasks_dict[dataset]

        tasks_mean = list()
        for specific_task in process_tasks:
            mean, std = self.statistic_specific_task_acc(specific_task)
            tasks_mean.append(mean)
        average_mean = round(float(np.mean(tasks_mean)), ndigits=4)
        print("{} average mean:{}".format(dataset, average_mean))


class ProcessMCD(ProcessAlgorithm):
    def __init__(self, config):
        ProcessAlgorithm.__init__(self, config=config)
        self.ancestor_path = os.path.join(dir_path_reproduction, 'MCD')
        self.dir_path = self.ancestor_path
        self.all_use = config.all_use
        self.invalid_setting = False
        self.exp_setting_num = 0

    def get_acc_list(self):
        """

        :return:
        """
        file_name = self.get_target_file_path()
        if self.invalid_setting:
            return
        file_path = os.path.join(self.dir_path, file_name)
        print('processing:', file_path)
        target_test_acc_list = dict({'acc_c1': list(), 'acc_c2': list(), 'acc_ensemble': list()})
        with open(file_path, 'r') as file:
            file.__next__()
            line_counter = 0
            for line in file.readlines():
                separated_list = line.split(sep=':')
                split_index = 5
                current_acc_c1 = float(separated_list[split_index].split(sep=',')[0])
                current_acc_c2 = float(separated_list[split_index + 1].split(sep=',')[0])
                current_acc_ensemble = float(separated_list[split_index + 2].split(sep='\n')[0])

                line_counter = line_counter + 1
                if line_counter % 2 == 0:
                    continue

                target_test_acc_list['acc_c1'].append(current_acc_c1)
                target_test_acc_list['acc_c2'].append(current_acc_c2)
                target_test_acc_list['acc_ensemble'].append(current_acc_ensemble)

        return target_test_acc_list

    def get_mean_std(self, acc_list, last_n=5):
        """
        计算均值与方差
        :param acc_list:
        :param last_n:
        :return:
        """
        calculate_data = np.asarray(acc_list[AccType.acc_ensemble.value][-1: -(last_n + 1): -1])
        mean = round(float(np.mean(calculate_data)), 4)
        std = round(float(np.std(calculate_data)), 5)
        return mean, std

    def statistic_specific_task_acc(self, specific_task):
        """
        统计单任务均值与方差
        :param specific_task:
        :return:
        """
        assert (specific_task in office_31_tasks) or (specific_task in digits_tasks)
        self.specific_task = specific_task
        acc_list = self.get_acc_list()
        if self.invalid_setting:
            return None, None

        target_test_mean_std = self.get_mean_std(acc_list, last_n=5)

        print("target domain:{}".format(target_test_mean_std))
        return target_test_mean_std

    def statistic_all_tasks_acc(self, dataset='office_31', exp_setting_nums=1):
        """
        统计 Office_31 数据集所有任务对应均值与方差
        :param dataset:
        :param exp_setting_nums:
        :return:
        """
        if not (dataset in dataset_tasks_dict.keys()):
            raise ValueError("statistic dataset must in ['digits', 'office_31']")
        process_tasks = dataset_tasks_dict[dataset]

        all_tasks_mean_list = list()
        for specific_task in process_tasks:
            specific_task_all_settings_mean_list = list()
            for exp_setting_num in range(exp_setting_nums):
                self.exp_setting_num = exp_setting_num
                mean, std = self.statistic_specific_task_acc(specific_task)
                if self.invalid_setting:
                    continue
                specific_task_all_settings_mean_list.append(mean)

            best_mean = max(specific_task_all_settings_mean_list) if specific_task_all_settings_mean_list \
                else 'To be ignored, invalid experiment record setting.'
            print("{} best exp setting mean:{}".format(specific_task, best_mean))
            all_tasks_mean_list.append(best_mean)
        if self.specific_task in office_31_tasks:
            average_mean = round(float(np.mean(all_tasks_mean_list)), ndigits=4)
            print("{} average mean:{}".format(dataset, average_mean))

    def get_target_file_path(self):
        """

        :return:
        """
        self.invalid_setting = False
        self.dir_path = self.ancestor_path
        self.get_dir_path()
        print('processing:', self.dir_path)
        file_name_list = os.listdir(self.dir_path)
        target_file_name = list(filter(
            lambda file_name: (self.specific_task in file_name) and
                              ('record_num_{}_'.format(self.exp_setting_num) in file_name) and 'test' in file_name,
            file_name_list))
        file_path = None
        if target_file_name:
            target_file_name = target_file_name[0]
            file_path = os.path.join(self.dir_path, target_file_name)
        else:
            self.invalid_setting = True
        return file_path

    def get_dir_path(self):
        """

        :return:
        """
        assert self.specific_task in digits_tasks or self.specific_task in office_31_tasks
        sub_dir_dict = {'svhn_mnist': 'svhn_mnist', 'synsig_gtsrb': 'synsig_gtsrb'}
        mnist_usps_task = ['mnist_usps', 'usps_mnist']

        sub_dir = sub_dir_dict[self.specific_task] if self.specific_task in sub_dir_dict.keys() \
            else 'office_31' if self.specific_task in office_31_tasks \
            else 'mnist_usps_all_use' if (self.all_use and self.specific_task in mnist_usps_task) \
            else 'mnist_usps_not_all' if (not self.all_use and self.specific_task in mnist_usps_task) else None
        if not sub_dir:
            raise ValueError('sub directory error!')
        else:
            self.dir_path = os.path.join(self.dir_path, sub_dir)


class ProcessAnon(ProcessMCD):
    def __init__(self, config):
        ProcessMCD.__init__(self, config=config)
        self.ancestor_path = os.getcwd()
        self.dir_path = self.ancestor_path


class ProcessADR(ProcessMCD):
    def __init__(self, config):
        ProcessMCD.__init__(self, config=config)
        self.ancestor_path = os.path.join(dir_path_reproduction, 'ADR')


class ProcessCDAN(ProcessAlgorithm):
    def __init__(self, config):
        ProcessAlgorithm.__init__(self, config=config)
        self.dir_path = os.path.join(dir_path_reproduction, 'CDAN')

    def get_acc_list(self):
        """

        :return:
        """
        file_name = "{}_log.txt".format(self.specific_task)
        file_path = os.path.join(self.dir_path, file_name)
        print('processing:', file_path)
        target_test_acc_list = list()
        with open(file_path, 'r') as file:
            for line in file.readlines():
                separated_list = line.split(sep=':')
                split_index = 3
                target_test_acc_list.append(float(separated_list[split_index]))

        return target_test_acc_list


class ProcessDANN(ProcessCDAN):
    def __init__(self, config):
        ProcessCDAN.__init__(self, config=config)
        self.dir_path = os.path.join(dir_path_reproduction, 'DANN')


class ProcessJAN(ProcessCDAN):
    def __init__(self, config):
        ProcessCDAN.__init__(self, config=config)
        self.dir_path = os.path.join(dir_path_reproduction, 'JAN')


class ProcessDAN(ProcessCDAN):
    def __init__(self, config):
        ProcessCDAN.__init__(self, config=config)
        self.dir_path = os.path.join(dir_path_reproduction, 'DAN')
