import numpy as np
import torch
import torch.nn.functional as F


def init_variable_output_t(config, num_target_dataset_train):
    """

    :param config:
    :param num_target_dataset_train:
    :return:
    """
    current_output_t, ensemble_output_t, correction_output_t = \
        torch.zeros(num_target_dataset_train, config.num_classes, dtype=torch.float).cuda(), \
        torch.zeros(num_target_dataset_train, config.num_classes, dtype=torch.float).cuda(), \
        torch.zeros(num_target_dataset_train, config.num_classes, dtype=torch.float).cuda()
    return current_output_t, ensemble_output_t, correction_output_t


def update_ensemble_and_correction_output_t(epoch, alpha, ensemble_output_t, current_output_t):
    """

    :param epoch:
    :param alpha:
    :param ensemble_output_t:
    :param current_output_t:
    :return:
    """
    ensemble_output_t = alpha * ensemble_output_t + (1 - alpha) * current_output_t
    correction_output_t = ensemble_output_t * (1. / (1. - alpha ** (epoch + 1)))
    return ensemble_output_t, correction_output_t


def rampup_sigmoid(current_epoch, rampup_length):
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242
    :param current_epoch:
    :param rampup_length:
    :return:
    """
    if rampup_length == 0:
        return 1.0
    else:
        current_epoch = np.clip(current_epoch, 0.0, rampup_length)
        phase = 1.0 - current_epoch / rampup_length
        return np.exp(-5.0 * phase * phase)


def rampup_linear(current, rampup_length):
    """
    Linear rampup
    :param current:
    :param rampup_length:
    :return:
    """
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def rampdown_cosine(current, rampdown_length):
    """
    Cosine rampdown from https://arxiv.org/abs/1608.03983
    :param current:
    :param rampdown_length:
    :return:
    """
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def loss_softmax_mse(input_logits, target_logits):
    """
    Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.mse_loss(input_softmax, target_softmax)


def softmax_kl_loss(input_logits, target_logits):
    """
    Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2) ** 2) / num_classes


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Use the true average until the exponential average is more correct
    :param model:
    :param ema_model:
    :param alpha:
    :param global_step:
    :return:
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def adjust_learning_rate(config, optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = config.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = rampup_linear(epoch, config.lr_rampup) * (config.lr - config.initial_lr) + config.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if config.lr_rampdown_epochs:
        assert config.lr_rampdown_epochs >= config.epochs
        lr *= rampdown_cosine(epoch, config.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(weight_consistency_upper, current_epoch, rampup_length):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return weight_consistency_upper * rampup_sigmoid(current_epoch, rampup_length)


def rampup(config, epoch):
    if epoch < config.rampup_length:
        p = max(0.0, float(epoch)) / float(config.rampup_length)
        p = 1.0 - p
        return np.math.exp(-p * p * 5.0)
    else:
        return 1.0


def rampdown(config, epoch):
    if epoch >= (config.num_epochs - config.rampdown_length):
        ep = (epoch - (config.num_epochs - config.rampdown_length)) * 0.5
        return np.math.exp(-(ep * ep) / config.rampdown_length)
    else:
        return 1.0


class EMAWeightOptimizer(object):
    def __init__(self, target_net, source_net, alpha=0.999):
        self.target_net = target_net
        self.source_net = source_net
        self.ema_alpha = alpha
        self.target_params = list(target_net.state_dict().values())
        self.source_params = list(source_net.state_dict().values())

        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p[:] = src_p[:]

        target_keys = set(target_net.state_dict().keys())
        source_keys = set(source_net.state_dict().keys())
        if target_keys != source_keys:
            raise ValueError('Source and target networks do not have the same state dict keys; do they have different '
                             'architectures?')

    def step(self):
        one_minus_alpha = 1.0 - self.ema_alpha
        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p.mul_(self.ema_alpha)
            tgt_p.add_(src_p * one_minus_alpha)
