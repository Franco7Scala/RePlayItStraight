from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch import tensor, long
import numpy as np


class CIFAR100Subset(datasets.CIFAR100):
    def __init__(self, subset, **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        assert max(subset) <= max(self.targets)
        assert min(subset) >= min(self.targets)

        self.aligned_indices = []
        for idx, label in enumerate(self.targets):
            if label in subset:
                self.aligned_indices.append(idx)

    def get_class_names(self):
        return [self.classes[i] for i in self.subset]

    def __len__(self):
        return len(self.aligned_indices)

    def __getitem__(self, item):
        return super().__getitem__(self.aligned_indices[item])


def CIFAR50(args):
    channel = 3
    im_size = (32, 32)
    num_classes = 50
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]

    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    subset = list(range(0, num_classes))
    dst_train = CIFAR100Subset(subset=subset, root=args.data_path+'/cifar100', train=True, download=True, transform=train_transform)
    dst_unlabeled = CIFAR100Subset(subset=subset, root=args.data_path+'/cifar100', train=True, download=True, transform=test_transform)
    dst_test = CIFAR100Subset(subset=subset, root=args.data_path+'/cifar100', train=False, download=True, transform=test_transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_unlabeled, dst_test
