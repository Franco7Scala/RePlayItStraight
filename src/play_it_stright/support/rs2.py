import torch
import random

from src.play_it_stright.support.utils import DataLoaderX


def split_dataset_for_rs2(dst_train, args):
    result = []
    size_batches = int(len(dst_train) / args.n_split)
    indices = list(range(len(dst_train)))
    random.shuffle(indices)

    for i in range(args.n_split):
        split_set = indices[i * size_batches:(i + 1) * size_batches]
        dst_subset = torch.utils.data.Subset(dst_train, split_set)
        if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
            train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

        else:
            train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

        result.append(train_loader)

    return result
