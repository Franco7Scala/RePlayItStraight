import random
import nets
import torch
import datasets_
import methods as methods
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from src.re_play_it_straight.support.support import clprint, Reason
from src.re_play_it_straight.support.rs2 import split_dataset_for_rs2, rs2_split_dataset
from src.re_play_it_straight.support.utils import *
from src.re_play_it_straight.support.arguments import parser
from ptflops import get_model_complexity_info


def rs2_training(dst_train, args, network, train_loader, test_loader, boot_epochs, n_split, type="boot", check_accuracy=False):
    splits_for_rs2 = split_dataset_for_rs2(dst_train, args)
    criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, round(len(train_loader) / n_split) * args.epochs, eta_min=args.min_lr)
    epoch = 0
    accs = []
    precs = []
    recs = []
    f1s = []
    tot_backward_steps = 0
    while epoch < boot_epochs:
        for split in splits_for_rs2:
            print("performing RS2 {} epoch n.{}/{}".format(type, epoch + 1, boot_epochs))
            _, backward_steps = train(split, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
            tot_backward_steps += backward_steps
            epoch += 1

            if epoch % 10 == 0:
                accuracy, precision, recall, f1 = test(test_loader, network, criterion, epoch, args, rec)
                accs.append(accuracy)
                precs.append(precision)
                recs.append(recall)
                f1s.append(f1)
                clprint(f"{type} epoch {epoch}/{boot_epochs} | Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}", reason=Reason.OUTPUT_TRAINING)
                if check_accuracy and accuracy >= args.target_accuracy:
                    clprint("Early stopping RS2 due target accuracy reached!", reason=Reason.OUTPUT_TRAINING)
                    return accuracy, precision, recall, f1, tot_backward_steps

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

    return accuracy, precision, recall, f1, tot_backward_steps


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
    seed_everything(args.seed)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_u_all, dst_test = datasets_.__dict__[args.dataset](args)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
    print("im_size: ", dst_train[0][0].shape)
    # BackgroundGenerator for ImageNet to speed up dataloaders
    train_size = int(len(dst_train) * 0.80)
    validation_size = len(dst_train) - train_size
    train_lengths = [train_size, validation_size]
    subset_train, subset_validation = torch.utils.data.random_split(dst_train, train_lengths)

    if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
        train_loader = DataLoaderX(subset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        validation_loader = DataLoaderX(subset_validation, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        evaluation_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    else:
        train_loader = torch.utils.data.DataLoader(subset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        validation_loader = torch.utils.data.DataLoader(subset_validation, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        evaluation_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    print("| Training on model %s" % args.model)
    tot_backward_steps = 0
    network = get_model(args, nets, args.model)
    macs, params = get_model_complexity_info(network, (channel, im_size[0], im_size[1]), as_strings=True, print_per_layer_stat=False, verbose=False)
    print("{:<30}  {:<8}".format("MACs: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
    # RS2 boot training
    print("==================== RS2 boot training ====================")
    print("RS2 split size: {}".format(int(len(dst_train) / args.n_split)))
    accuracy, precision, recall, f1, steps = rs2_training(dst_train, args, network, train_loader, validation_loader, args.boot_epochs, args.n_split)
    tot_backward_steps += steps

    # Active learning cycles
    # Initialize Unlabeled Set & Labeled Set
    indices = list(range(len(dst_train)))
    random.shuffle(indices)
    labeled_set = []
    new_labeled_set = []
    unlabeled_set = indices
    logs_accuracy = []
    logs_precision = []
    logs_recall = []
    logs_f1 = []
    cycle = 0
    current_loss = None
    previous_loss = None
    n_rs2_refresh = 0
    while accuracy < args.target_accuracy:
        cycle += 1
        print("====================Cycle: {}====================".format(cycle))
        print("==========Start Querying==========")
        selection_args = dict(selection_method=args.uncertainty, balance=args.balance, greedy=args.submodular_greedy, function=args.submodular)
        ALmethod = methods.__dict__[args.method](dst_u_all, unlabeled_set, network, args, **selection_args)
        Q_indices, Q_scores = ALmethod.select()
        # Update the labeled datasets_ and the unlabeled datasets_, respectively
        for idx in Q_indices:
            new_labeled_set.append(idx)
            unlabeled_set.remove(idx)

        clprint("Run: # of Labeled: {}, # of new Labeled: {}, # of Unlabeled: {}".format(len(labeled_set), len(new_labeled_set), len(unlabeled_set)), Reason.SETUP_TRAINING)
        print("==========Start Training==========")
        if len(labeled_set) >= 25000:
            clprint("Performing an optimized run due to the excessively large size of the labeled datasets_...", Reason.INFO_TRAINING)
            # Updating scheduler according to RS2
            if len(labeled_set) == 0:
                n_split = 1

            else:
                if len(new_labeled_set) == 0:
                    n_split = args.epochs

                else:
                    n_split = min(int(len(labeled_set) / len(new_labeled_set)), args.epochs)

            if n_split == 0:
                t_max = int(len(new_labeled_set) / args.batch_size) * args.epochs

            else:
                t_max = int((len(new_labeled_set) + (len(labeled_set) / n_split)) / args.batch_size) * args.epochs

            # Get optim configurations for Distrubted SGD
            criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=args.min_lr)
            splitted_old_labeled_set = rs2_split_dataset(dst_train=dst_train, indices=labeled_set, n_split=n_split)
            first = True
            for epoch in range(args.epochs):
                # taking a different chunk for each epoch from old datas
                if len(splitted_old_labeled_set) > 0:
                    j = epoch % len(splitted_old_labeled_set)
                    dataset = torch.utils.data.Subset(dst_train, new_labeled_set + splitted_old_labeled_set[j].indices)

                else:
                    dataset = torch.utils.data.Subset(dst_train, new_labeled_set)

                if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
                    train_loader = DataLoaderX(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

                else:
                    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

                current_loss, backward_steps = train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
                tot_backward_steps += backward_steps

        else:
            clprint("Performing standard run...", Reason.INFO_TRAINING)
            dst_subset = torch.utils.data.Subset(dst_train, labeled_set + new_labeled_set)
            if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
                train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

            else:
                train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

            # Get optim configurations for Distrubted SGD
            criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
            for epoch in range(args.epochs):
                _, backward_steps = train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
                tot_backward_steps += backward_steps

        labeled_set += new_labeled_set
        new_labeled_set = []
        # Performing RS2 training on the whole datasets_ if it is slowing training
        if previous_loss is not None:
            relative_change = abs(previous_loss - current_loss) / previous_loss

        else:
            relative_change = 0

        if relative_change != 0 and relative_change < args.boost_threshold:
            clprint(f"Performing n {n_rs2_refresh} boost epoch...", reason=Reason.INFO_TRAINING)
            bkp_lr = args.lr
            args.lr = args.lr * 0.1
            accuracy, precision, recall, f1, steps = rs2_training(dst_train, args, network, train_loader, validation_loader, 10, 10, "boost", check_accuracy=True)
            args.lr = bkp_lr
            tot_backward_steps += steps
            n_rs2_refresh += 1

        else:
            accuracy, precision, recall, f1 = test(validation_loader, network, criterion, epoch, args, rec)

        previous_loss = current_loss
        clprint(f"Cycle {cycle} || Label set size {len(labeled_set)} | Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}", reason=Reason.OUTPUT_TRAINING)
        clprint(f"Done {tot_backward_steps} backward steps to reach {accuracy} of accuracy!", reason=Reason.OUTPUT_TRAINING)

        logs_accuracy.append([accuracy])
        logs_precision.append([precision])
        logs_recall.append([recall])
        logs_f1.append([f1])

    print("========== Pre-Final logs ==========")
    print("-" * 100)
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
    accuracy, precision, recall, f1 = test(evaluation_loader, network, criterion, epoch, args, rec)

    clprint("========== Final logs ==========", Reason.OUTPUT_TRAINING)
    clprint("-" * 100, Reason.OUTPUT_TRAINING)
    clprint("Backward steps:", Reason.OUTPUT_TRAINING)
    clprint(tot_backward_steps, Reason.OUTPUT_TRAINING)
    clprint("Accuracy:", Reason.OUTPUT_TRAINING)
    clprint(accuracy, Reason.OUTPUT_TRAINING)
    clprint("Precision:", Reason.OUTPUT_TRAINING)
    clprint(precision, Reason.OUTPUT_TRAINING)
    clprint("Recall:", Reason.OUTPUT_TRAINING)
    clprint(recall, Reason.OUTPUT_TRAINING)
    clprint("F1:", Reason.OUTPUT_TRAINING)
    clprint(f1, Reason.OUTPUT_TRAINING)
