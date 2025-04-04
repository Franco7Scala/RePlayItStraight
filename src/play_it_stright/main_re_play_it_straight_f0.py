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
from PIL import Image, ImageFile


random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
                # if accuracy >= 60:
                #     clprint(f"Early stopping on boot, a nice accuracy was reached after {epoch} epochs!", reason=Reason.INFO_TRAINING)
                #     epoch = args.boot_epochs
                #     break
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
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_u_all, dst_test = datasets.__dict__[args.dataset](args)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
    print("im_size: ", dst_train[0][0].shape)
    # BackgroundGenerator for ImageNet to speed up dataloaders
    if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
        whole_train_loader = DataLoaderX(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    else:
        whole_train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
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
    accuracy, precision, recall, f1, steps = rs2_training(dst_train, args, network, whole_train_loader, test_loader, args.boot_epochs, args.n_split)
    tot_backward_steps += steps

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
    current_loss = None
    previous_loss = None
    n_rs2_refresh = 0
    accuracy_thresholds = list(range(0, 100, 5))
    last_threshold = 0
    while accuracy < args.target_accuracy:
        cycle += 1
        print("====================Cycle: {}====================".format(cycle))
        print("==========Start Querying==========")
        selection_args = dict(selection_method=args.uncertainty, balance=args.balance, greedy=args.submodular_greedy, function=args.submodular)
        al_method = methods.__dict__[args.method](dst_u_all, unlabeled_set, network, args, **selection_args)
        Q_indices, Q_scores = al_method.select()
        # Update the labeled dataset and the unlabeled dataset, respectively
        new_labeled_set = []
        for idx in Q_indices:
            new_labeled_set.append(idx)
            unlabeled_set.remove(idx)

        clprint("Optimized run: # of Labeled: {}, # of new Labeled: {}, # of Unlabeled: {}".format(len(labeled_set), len(new_labeled_set), len(unlabeled_set)), Reason.SETUP_TRAINING)
        # Get optim configurations for Distributed SGD
        criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, whole_train_loader)
        # Updating scheduler according to RS2
        if len(new_labeled_set) == 0:
            n_split = args.epochs

        else:
            #n_split = min(int(len(labeled_set) / len(new_labeled_set)), args.epochs) TODO
            n_split = 1#min(int(len(labeled_set) / len(new_labeled_set) / 3), 1)

        if len(labeled_set) > 0:
            t_max = max(1, int((len(new_labeled_set) + (len(labeled_set)/n_split)) / args.batch_size) * args.epochs)

        else:
            t_max = max(1, int(len(new_labeled_set) / args.batch_size) * args.epochs)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=args.min_lr)

        if n_split == 0:
            splitted_old_labeled_set = []

        else:
            splitted_old_labeled_set = rs2_split_dataset(dst_train=dst_train, indices=labeled_set, n_split=n_split)

        first = True
        print("==========Start Training==========")
        for epoch in range(args.epochs):
            # taking a different chunk for each epoch from old datas
            if len(splitted_old_labeled_set) > 0:
                j = epoch % len(splitted_old_labeled_set)
                if not first and j == 0:
                    splitted_old_labeled_set = rs2_split_dataset(dst_train=dst_train, indices=labeled_set, n_split=n_split) #max(1, int((len(labeled_set) / len(new_labeled_set))))) TODO

                else:
                    first = False

                dataset = torch.utils.data.Subset(dst_train, new_labeled_set + splitted_old_labeled_set[j].indices)

            else:
                dataset = torch.utils.data.Subset(dst_train, new_labeled_set)

            if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
                train_loader = DataLoaderX(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

            else:
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

            current_loss, backward_steps = train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
            tot_backward_steps += backward_steps

        labeled_set = labeled_set + new_labeled_set
        # Performing RS2 training on the whole dataset if it is slowing training
        if previous_loss is not None:
            relative_change = abs(previous_loss - current_loss) / previous_loss

        else:
            relative_change = 0

        if relative_change != 0 and relative_change < args.boost_threshold:
            clprint(f"-------------------------- previous loss: {previous_loss}", Reason.WARNING)
            clprint(f"-------------------------- current loss: {current_loss}", Reason.WARNING)
            clprint(f"-------------------------- relative change: {relative_change}", Reason.WARNING)

            clprint(f"Performing n {n_rs2_refresh} boost epoch...", reason=Reason.INFO_TRAINING)
            bkp_lr = args.lr
            args.lr = args.lr * 0.1
            accuracy, precision, recall, f1, steps = rs2_training(dst_train, args, network, train_loader, test_loader, 10, 10, "boost", check_accuracy=True)
            args.lr = bkp_lr

            # bkp_scheduler = args.scheduler
            # bkp_lr = args.lr
            # args.scheduler = "LinearLR"
            # args.lr = args.lr / 4
            # criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, whole_train_loader)
            # _, backward_steps = train(whole_train_loader, network, criterion, optimizer, scheduler, 1, args, rec, if_weighted=False)
            # args.scheduler = bkp_scheduler
            # args.lr = bkp_lr
            tot_backward_steps += steps
            n_rs2_refresh += 1

        # if cycle < 12 and cycle % 10 == 0:
        #     rs2_training(dst_train, args, network, train_loader, test_loader, 10, 10)
        #
        # elif cycle >= 12 and cycle % 5 == 0:
        #     rs2_training(dst_train, args, network, train_loader, test_loader, 10, 10)

        else:
            accuracy, precision, recall, f1 = test(test_loader, network, criterion, epoch, args, rec)

        previous_loss = current_loss
        clprint(f"Cycle {cycle} || Label set size {len(labeled_set)} | Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}", reason=Reason.OUTPUT_TRAINING)

        #if accuracy >= accuracy_thresholds[last_threshold]:
        clprint(f"Done {tot_backward_steps} backward steps to reach {accuracy} of accuracy!", reason=Reason.OUTPUT_TRAINING)
        last_threshold += 1

        logs_accuracy.append([accuracy])
        logs_precision.append([precision])
        logs_recall.append([recall])
        logs_f1.append([f1])

    print("========== Final logs ==========")
    print(f"Performed n {n_rs2_refresh} refresh epochs!")
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
