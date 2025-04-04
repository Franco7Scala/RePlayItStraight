import random
import nets
import torch
import datasets as datasets
import methods as methods
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from src.play_it_stright.support.support import clprint, Reason
from src.play_it_stright.support.rs2 import split_dataset_for_rs2
from src.play_it_stright.support.utils import *
from src.play_it_stright.support.arguments import parser
from ptflops import get_model_complexity_info


random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


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
    tot_backward_steps = 0
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
    splits_for_rs2 = split_dataset_for_rs2(dst_train, args)
    criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, round(len(train_loader)/args.n_split) * args.epochs, eta_min=args.min_lr)
    epoch = 0
    accs = []
    precs = []
    recs = []
    f1s = []
    while epoch < args.boot_epochs:
        for split in splits_for_rs2:
            print("performing RS2 boot epoch n.{}/{}".format(epoch + 1, args.boot_epochs))
            _, backward_steps = train(split, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
            tot_backward_steps += backward_steps
            epoch += 1
            if epoch % 10 == 0:
                accuracy, precision, recall, f1 = test(test_loader, network, criterion, epoch, args, rec)
                accs.append(accuracy)
                precs.append(precision)
                recs.append(recall)
                f1s.append(f1)
                clprint("Boot epoch {}/{} | Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(epoch, args.boot_epochs, accuracy, precision, recall, f1), reason=Reason.OUTPUT_TRAINING)

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
        for idx in Q_indices:
            labeled_set.append(idx)
            unlabeled_set.remove(idx)

        print("# of Labeled: {}, # of Unlabeled: {}".format(len(labeled_set), len(unlabeled_set)))
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
            _, backward_steps = train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
            tot_backward_steps += backward_steps

        accuracy, precision, recall, f1 = test(test_loader, network, criterion, epoch, args, rec)
        clprint("Cycle {} || Label set size {} | Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(cycle, len(labeled_set), accuracy, precision, recall, f1), reason=Reason.OUTPUT_TRAINING)
        clprint(f"Done {tot_backward_steps} backward steps to reach {accuracy} of accuracy!", reason=Reason.OUTPUT_TRAINING)

        logs_accuracy.append([accuracy])
        logs_precision.append([precision])
        logs_recall.append([recall])
        logs_f1.append([f1])

    print("========== Final logs ==========")
    print("-"*100)
    print("Backward steps:")
    print(tot_backward_steps, flush=True)
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
