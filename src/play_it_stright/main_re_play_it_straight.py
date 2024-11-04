import random
import nets
import torch
import copy
import datasets as datasets
import methods as methods
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from src.play_it_stright.support.support import clprint, Reason
from src.play_it_stright.support.rs2 import split_dataset_for_rs2, rs2_split_dataset
from src.play_it_stright.support.utils import *
from src.play_it_stright.support.arguments import parser
from ptflops import get_model_complexity_info


random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


def rs2_training(dst_train, args, network, train_loader, test_loader, boot_epochs, n_split):
    splits_for_rs2 = split_dataset_for_rs2(dst_train, args)
    criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, round(len(train_loader) / n_split) * args.epochs, eta_min=args.min_lr)
    epoch = 0
    accs = []
    precs = []
    recs = []
    f1s = []
    while epoch < boot_epochs:
        for split in splits_for_rs2:
            print("performing RS2 boot epoch n.{}/{}".format(epoch + 1, boot_epochs))
            train(split, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
            epoch += 1
            if epoch % 10 == 0:
                accuracy, precision, recall, f1 = test(test_loader, network, criterion, epoch, args, rec)
                accs.append(accuracy)
                precs.append(precision)
                recs.append(recall)
                f1s.append(f1)
                clprint("Boot epoch {}/{} | Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(epoch, boot_epochs, accuracy, precision, recall, f1), reason=Reason.OUTPUT_TRAINING)
                # if accuracy >= 60:
                #     clprint(f"Early stopping on boot, a nice accuracy was reached after {epoch} epochs!", reason=Reason.INFO_TRAINING)
                #     epoch = args.boot_epochs
                #     break

        print("Finished splits, reshuffling data and resplitting!")
        splits_for_rs2 = split_dataset_for_rs2(dst_train, args)

    accuracy, precision, recall, f1 = test(test_loader, network, criterion, epoch, args, rec)
    clprint("Boot completed | Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(accuracy, precision, recall, f1), reason=Reason.OTHER)
    print("Accuracies:")
    print(accs)
    print("Precisions:")
    print(precs)
    print("Recalls:")
    print(recs)
    print("F1s:")
    print(f1s)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    args = parser.parse_args()
    cuda = ""
    if len(args.gpu) > 1:
        cuda = "cuda"

    elif len(args.gpu) == 1:
        cuda = "cuda:"+str(args.gpu[0])

    if args.dataset == "ImageNet":
        args.device = cuda if torch.cuda.is_available() else "cpu"

    else:
        args.device = cuda if torch.cuda.is_available() else "cpu"

    print("args: ", args)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_u_all, dst_test = datasets.__dict__[args.dataset](args)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
    print("im_size: ", dst_train[0][0].shape)
    # BackgroundGenerator for ImageNet to speed up dataloaders
    if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
        train_loader = DataLoaderX(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    else:
        train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    print("| Training on model %s" % args.model)
    network = get_model(args, nets, args.model)
    macs, params = get_model_complexity_info(network, (channel, im_size[0], im_size[1]), as_strings=True, print_per_layer_stat=False, verbose=False)
    print("{:<30}  {:<8}".format("MACs: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
    # Tracker for energy consumption calculation 
    #tracker = EmissionsTracker()
    #tracker.start()
    # RS2 boot training
    print("==================== RS2 boot training ====================")
    print("RS2 split size: {}".format(int(len(dst_train) / args.n_split)))
    accuracy, precision, recall, f1 = rs2_training(dst_train, args, network, train_loader, test_loader, args.boot_epochs, args.n_split)

    # Active learning cycles
    # Initialize Unlabeled Set & Labeled Set
    indices = list(range(len(dst_train)))
    random.shuffle(indices)
    labeled_set = []
    unlabeled_set = indices
    logs_accuracy = []
    logs_precision = []
    logs_recall = []
    logs_f1 = []
    cycle = 0
    while accuracy < args.target_accuracy:
        cycle += 1
        print("====================Cycle: {}====================".format(cycle))
        print("==========Start Querying==========")
        selection_args = dict(selection_method=args.uncertainty, balance=args.balance, greedy=args.submodular_greedy, function=args.submodular)
        ALmethod = methods.__dict__[args.method](dst_u_all, unlabeled_set, network, args, **selection_args)
        Q_indices, Q_scores = ALmethod.select()
        # Update the labeled dataset and the unlabeled dataset, respectively
        new_labeled_set = []
        for idx in Q_indices:
            new_labeled_set.append(idx)
            unlabeled_set.remove(idx)

        # improving efficiency when increases the amount of the selected data
        if True:#len(labeled_set) >= len(unlabeled_set) * 15:
            clprint("Optimized run: # of Labeled: {}, # of new Labeled: {}, # of Unlabeled: {}".format(len(labeled_set), len(new_labeled_set), len(unlabeled_set)), Reason.SETUP_TRAINING)
            # Get optim configurations for Distrubted SGD
            criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
            # Updating scheduler according to RS2
            n_split = min(int(len(labeled_set) / len(new_labeled_set)), args.epochs) # <- ad ora soluzione migliore
            if len(labeled_set) > 0:
                t_max = max(1, int((len(new_labeled_set) + (len(labeled_set)/n_split)) / args.batch_size) * args.epochs)

            else:
                t_max = max(1, int(len(new_labeled_set) / args.batch_size) * args.epochs)


            #min_lr = args.min_lr if cycle < 15 else args.min_lr * 2
            #lr = args.lr if cycle < 15 else args.lr * 2
            #args.optimizer = torch.optim.SGD(network.parameters(), lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=args.min_lr)

            if n_split == 0:
                splitted_old_labeled_set = []

            else:
                # if len(labeled_set) > (len(new_labeled_set) * args.epochs):
                #     args2 = copy.deepcopy(args)
                #     args2.n_query = len(new_labeled_set) * args.epochs
                #     ALmethod = methods.__dict__[args.method](dst_u_all, labeled_set, network, args2, **selection_args)
                #     Q_indices, _ = ALmethod.select()
                #     splitted_old_labeled_set = rs2_split_dataset(dst_train=dst_train, indices=Q_indices, n_split=n_split)
                #
                # else:
                splitted_old_labeled_set = rs2_split_dataset(dst_train=dst_train, indices=labeled_set, n_split=n_split)

            first = True
            print("==========Start Training==========")
            for epoch in range(args.epochs):
                # taking a different chunk for each epoch from old datas
                if len(splitted_old_labeled_set) > 0:
                    j = epoch % len(splitted_old_labeled_set)
                    if not first and j == 0:
                        splitted_old_labeled_set = rs2_split_dataset(dst_train=dst_train, indices=labeled_set, n_split=int(len(labeled_set) / len(new_labeled_set)))

                    else:
                        first = False

                    dataset = torch.utils.data.Subset(dst_train, new_labeled_set + splitted_old_labeled_set[j].indices)

                else:
                    dataset = torch.utils.data.Subset(dst_train, new_labeled_set)

                if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
                    train_loader = DataLoaderX(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

                else:
                    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

                train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)

            labeled_set = labeled_set + new_labeled_set

            if cycle < 12 and cycle % 10 == 0:
                rs2_training(dst_train, args, network, train_loader, test_loader, 10, 10)

            elif cycle >= 12 and cycle % 5 == 0:
                rs2_training(dst_train, args, network, train_loader, test_loader, 10, 10)

        else:
            labeled_set = labeled_set + new_labeled_set
            clprint("Base run: # of Labeled: {}, # of Unlabeled: {}".format(len(labeled_set), len(unlabeled_set)), Reason.LIGHT_INFO_TRAINING)
            assert len(labeled_set) == len(list(set(labeled_set))) and len(unlabeled_set) == len(list(set(unlabeled_set)))

            dst_subset = torch.utils.data.Subset(dst_train, labeled_set)
            if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
                train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

            else:
                train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

            # Get optim configurations for Distrubted SGD
            criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
            print("==========Start Training==========")
            for epoch in range(args.epochs):
                train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)

        accuracy, precision, recall, f1 = test(test_loader, network, criterion, epoch, args, rec)
        clprint("Cycle {} || Label set size {} | Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(cycle, len(labeled_set), accuracy, precision, recall, f1), reason=Reason.OUTPUT_TRAINING)

        logs_accuracy.append([accuracy])
        logs_precision.append([precision])
        logs_recall.append([recall])
        logs_f1.append([f1])

    print("========== Final logs ==========")
    print("-"*100)
    print("Accuracies:")
    logs_accuracy = np.array(logs_accuracy).reshape((-1, 1))
    print(logs_accuracy, flush=True)
    print("Precisions:")
    logs_precision = np.array(logs_precision).reshape((-1, 1))
    print(logs_precision, flush=True)
    print("Recalls:")
    logs_recall = np.array(logs_recall).reshape((-1, 1))
    print(logs_recall, flush=True)
    print("F1s:")
    logs_f1 = np.array(logs_f1).reshape((-1, 1))
    print(logs_f1, flush=True)

    #tracker.stop()
