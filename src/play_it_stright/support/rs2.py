import torch
import random

from src.play_it_stright.support.utils import DataLoaderX


def split_dataset_for_rs2(dst_train, args):
    result = []
    print(f"Discount RS2: {args.discount_rs2}")
    size_batches = int(len(dst_train) / args.n_split / args.discount_rs2)
    indices = list(range(len(dst_train)))
    random.shuffle(indices)
    indices = indices[:int(len(indices) / args.discount_rs2)]

    for i in range(args.n_split):
        split_set = indices[i * size_batches:(i + 1) * size_batches]
        dst_subset = torch.utils.data.Subset(dst_train, split_set)
        if len(dst_subset) > 0:
            if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
                train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

            else:
                train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

        result.append(train_loader)

    return result


def rs2_split_dataset(dst_train, indices, n_split):
    if len(dst_train) == 0 or len(indices) == 0 or n_split == 0:
        return []

    result = []
    size_batches = int(len(indices) / n_split)
    random.shuffle(indices)

    for i in range(n_split):
        split_set = indices[i * size_batches:(i + 1) * size_batches]
        dst_subset = torch.utils.data.Subset(dst_train, split_set)
        #train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
        result.append(dst_subset)

    return result
