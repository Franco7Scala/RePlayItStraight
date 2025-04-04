import random
import nets
import torch.optim as optim
import datasets as datasets

from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from src.play_it_stright.support.support import clprint
from src.play_it_stright.support.rs2 import split_dataset_for_rs2
from src.play_it_stright.support.utils import *
from src.play_it_stright.support.arguments import parser
from ptflops import get_model_complexity_info


random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


# Main
if __name__ == "__main__":
    args = parser.parse_args()
    cuda = ""
    if len(args.gpu) > 1:
        cuda = "cuda"

    elif len(args.gpu) == 1:
        cuda = "cuda:" + str(args.gpu[0])

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
    ##tracker.start()

    print("====================RS2 training====================")

    print("RS2 split size: {}".format(int(len(dst_train) / args.n_split)))
    print("Epochs: {}".format(args.epochs))
    splits_for_rs2 = split_dataset_for_rs2(dst_train, args)
    criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, round(len(train_loader)/args.n_split) * args.epochs, eta_min=args.min_lr)
    epoch = 0
    accs = []
    precs = []
    recs = []
    f1s = []
    logs = []
    tot_backward_steps = 0
    while epoch < args.epochs:
        for split in splits_for_rs2:
            print("Performing RS2 training epoch n.{}".format(epoch + 1))
            _, backward_steps = train(split, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
            tot_backward_steps += backward_steps
            epoch += 1

            if epoch % 10 == 0:
                accuracy, precision, recall, f1 = test(test_loader, network, criterion, epoch, args, rec)
                accs.append([accuracy])
                precs.append([precision])
                recs.append([recall])
                f1s.append([f1])
                clprint("Training epoch {}/{} | Accuracy: {}, Precision: {}, Recall: {}, F1: {}, Steps: {}".format(epoch, args.epochs, accuracy, precision, recall, f1, tot_backward_steps), reason=Reason.OUTPUT_TRAINING)

        print("Finished splits, reshuffling data and resplitting!")
        splits_for_rs2 = split_dataset_for_rs2(dst_train, args)

    clprint("Boot completed | Accuracy: {}, Precision: {}, Recall: {}, F1: {}, Steps: {}".format(accuracy, precision, recall, f1, tot_backward_steps), reason=Reason.OTHER)
    print("========== Final logs ==========")
    print("Backward steps:")
    print(tot_backward_steps, flush=True)
    print("Accuracies:")
    print(accs)
    print("Precisions:")
    print(precs)
    print("Recalls:")
    print(recs)
    print("F1s:")
    print(f1s)
    #tracker.stop()
