from models import svhn_mnist, mnist_usps, office_31, synsig_gtsrb


class InitModel:
    def __init__(self):
        self.model = None

    def init(self, source_domain, target_domain):
        if source_domain in ['svhn', 'mnist'] and target_domain in ['svhn', 'mnist']:
            self.model = svhn_mnist
        elif source_domain in ['usps', 'mnist'] and target_domain in ['usps', 'mnist']:
            self.model = mnist_usps
        elif source_domain in ['synsig', 'gtsrb'] and target_domain in ['synsig', 'gtsrb']:
            self.model = synsig_gtsrb
        elif source_domain in ['A', 'W', 'D'] and target_domain in ['A', 'W', 'D']:
            self.model = office_31
        else:
            raise ValueError('invalid source domain or target domain parameter!')

        return self.model
