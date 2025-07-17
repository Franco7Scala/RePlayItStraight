from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch import tensor, long
from PIL import Image


class ImagewoofDataset(Dataset):
    def __init__(self, file_path, transform=None, resolution=160):
        self.transform = transform
        self.resolution = resolution
        print(f"Resizing Initial Data into {self.resolution}x{self.resolution}")
        transform_resize = T.Resize(size=(self.resolution, self.resolution))
        self.data = ImageFolder(file_path, transform_resize, is_valid_file=self.checkImage)
        self.classes = self.data.classes
        self.targets = self.data.targets

    def __getitem__(self, index):
        # id = self.id_sample[index]
        img, label = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, label#, index

    def __len__(self):
        return len(self.data)

    def checkImage(self, path):
        try:
            Image.open(path)
            return True

        except:
            return False


def get_augmentations_32(T_normalize):
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T_normalize])
    test_transform = T.Compose([T.ToTensor(), T_normalize])
    return train_transform, test_transform


def Imagewoof(args):
    channel = 3
    im_size = (32, 32)  # (160, 160) TODO
    num_classes = 10
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    T_normalize = T.Normalize(mean, std)
    #if args.resolution == 32: TODO
    train_transform, test_transform = get_augmentations_32(T_normalize) #TODO

    dst_train = ImagewoofDataset(args.data_path + '/Imagewoof/train/', transform=train_transform, resolution=args.resolution)
    dst_unlabeled = ImagewoofDataset(args.data_path + '/Imagewoof/train/', transform=test_transform, resolution=args.resolution)
    dst_test = ImagewoofDataset(args.data_path + '/Imagewoof/val/', transform=test_transform, resolution=args.resolution)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_unlabeled, dst_test
