from datasets import svhn, mnist, usps, synsig, gtsrb, office_31, office_home


def get_dataloader(domain, method, config):
    """
    specific domain description:
        A: Amazon, W: Webcam, D: DSLR
        Ar: Art, Cl: Clipart, Pr: Product, Rw: Real-World
    :param domain: source domain or target domain name
    :param method: domain dataset loading method, e.g. mcd, gta etc.
    :param config:
    :return:
    """
    office_31_domain = ['A', 'W', 'D']
    office_home_domain = ['Ar', 'Cl', 'Pr', 'Rw']

    dataset = svhn if domain == 'svhn' else mnist if domain == 'mnist' else usps if domain == 'usps' \
        else synsig if domain == 'synsig' else gtsrb if domain == 'gtsrb' else office_31 if domain in office_31_domain\
        else office_home if domain in office_home_domain else None
    if not dataset:
        raise ValueError('invalid domain parameter, not in expected datasets!')

    if method == 'gta':
        dataloader_train, dataloader_test = dataset.get_loader_gta(config)
    elif method == 'mcd':
        dataloader_train, dataloader_test = dataset.get_loader_mcd(config)
    elif method == 'cogan':
        dataloader_train, dataloader_test = dataset.get_loader_cogan(config)
    elif method == 'normal':
        dataloader_train, dataloader_test = dataset.get_loader_normal(config, domain=domain)
    else:
        raise ValueError('invalid domain loading method, not in expected loading methods!')

    return dataloader_train, dataloader_test
