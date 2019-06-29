from models import svhn_mnist, mnist_svhn, mnist_usps, office_31, synsig_gtsrb


class InitModel:
    def __init__(self):
        self.model = None

    def init(self, source_domain, target_domain):
        if source_domain == 'svhn' and target_domain == 'mnist':
            self.model = svhn_mnist
        elif source_domain == 'mnist' and target_domain == 'svhn':
            self.model = mnist_svhn
        elif source_domain == 'usps' or target_domain == 'usps':
            self.model = mnist_usps
        elif source_domain == 'synsig' and target_domain == 'gtsrb':
            self.model = synsig_gtsrb
        elif source_domain in ['A', 'W', 'D'] or target_domain in ['A', 'W', 'D']:
            self.model = office_31
        else:
            raise ValueError('invalid source domain or target domain parameter!')

        return self.model
