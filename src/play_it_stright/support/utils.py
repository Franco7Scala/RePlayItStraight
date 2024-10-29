import time
import os
import torch
import torch.nn as nn

from argparse import ArgumentTypeError
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
from src.play_it_stright.support.support import clprint, Reason


class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], self.weights[[i for i in idx]]

        return self.dataset[self.indices[idx]], self.weights[idx]


def train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted: bool = False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    network.train()
    end = time.time()
    for i, contents in enumerate(train_loader):
        optimizer.zero_grad()
        if if_weighted:
            target = contents[0][1].to(args.device)
            input = contents[0][0].to(args.device)
            # Compute output
            output = network(input)
            weights = contents[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)

        else:
            target = contents[1].to(args.device)
            input = contents[0].to(args.device)
            # Compute output
            output = network(input)
            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    record_train_stats(rec, epoch, losses.avg, top1.avg, optimizer.state_dict()['param_groups'][0]['lr'])


def test(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy_am = AverageMeter('Accuracy', ':6.2f')
    precision_am = AverageMeter('Precision', ':6.2f')
    recall_am = AverageMeter('Recall', ':6.2f')
    f1_am = AverageMeter('F1', ':6.2f')
    # Metrics for Precision, Recall, and F1
    all_preds = []
    all_targets = []
    # Switch to evaluate mode
    network.eval()
    network.no_grad = True
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.to(args.device)
        input = input.to(args.device)
        # Compute output
        with torch.no_grad():
            output = network(input)
            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        accuracy_am.update(prec1.item(), input.size(0))
        # Store predictions and targets for later calculations
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())  # Move to CPU before converting
        all_targets.extend(target.cpu().numpy())
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # Calculate precision, recall, and F1-score
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=1)
    precision_am.update(precision)
    recall_am.update(recall)
    f1_am.update(f1)

    clprint('Test | Accuracy: {top1.avg:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}'.format(top1=accuracy_am, precision=precision, recall=recall, f1=f1), reason=Reason.OUTPUT_TRAINING)

    network.no_grad = False
    return accuracy_am.avg, precision_am.avg, recall_am.avg, f1_am.avg


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def str_to_bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    else:
        raise ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, path, epoch, prec):
    print("=> Saving checkpoint for epoch %d, with Prec@1 %f." % (epoch, prec))
    torch.save(state, path)


def init_recorder():
    from types import SimpleNamespace
    rec = SimpleNamespace()
    rec.train_step = []
    rec.train_loss = []
    rec.train_acc = []
    rec.lr = []
    rec.test_step = []
    rec.test_loss = []
    rec.test_acc = []
    rec.ckpts = []
    return rec


def record_train_stats(rec, step, loss, acc, lr):
    rec.train_step.append(step)
    rec.train_loss.append(loss)
    rec.train_acc.append(acc)
    rec.lr.append(lr)
    return rec


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_more_args(args):
    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if args.resume != "":
        try:
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            assert {"exp", "epoch", "state_dict", "opt_dict", "best_acc1", "rec", "subset", "sel_args"} <= set(checkpoint.keys())
            assert 'indices' in checkpoint["subset"].keys()
            start_exp = checkpoint['exp']
            start_epoch = checkpoint["epoch"]

        except AssertionError:
            try:
                assert {"exp", "subset", "sel_args"} <= set(checkpoint.keys())
                assert 'indices' in checkpoint["subset"].keys()
                print("=> The checkpoint only contains the subset, training will start from the begining")
                start_exp = checkpoint['exp']
                start_epoch = 0
            except AssertionError:
                print("=> Failed to load the checkpoint, an empty one will be created")
                checkpoint = {}
                start_exp = 0
                start_epoch = 0
    else:
        checkpoint = {}
        start_exp = 0
        start_epoch = 0
    
    return args, checkpoint, start_exp, start_epoch


def get_configuration(args, nets, model, checkpoint, train_loader, start_epoch):
    network = nets.__dict__[model](args.channel, args.num_classes, args.im_size).to(args.device)
    if args.device == "cpu":
        print("Using CPU.")

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu[0])
        network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)

    elif torch.cuda.device_count() > 1:
        network = nets.nets_utils.MyDataParallel(network).cuda()

    if "state_dict" in checkpoint.keys():
        network.load_state_dict(checkpoint["state_dict"])

    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)

    else:
        optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # LR scheduler
    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs, eta_min=args.min_lr)

    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size, gamma=args.gamma)

    else:
        scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)

    scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

    if "opt_dict" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["opt_dict"])

    if "rec" in checkpoint.keys():
        rec = checkpoint["rec"]

    else:
        rec = init_recorder()

    return network, criterion, optimizer, scheduler, rec


def get_model(args, nets, model):
    network = nets.__dict__[model](args.channel, args.num_classes, args.im_size).to(args.device)
    if args.device == "cpu":
        print("Using CPU.")

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu[0])
        network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)

    elif torch.cuda.device_count() > 1:
        network = nets.nets_utils.MyDataParallel(network).cuda()

    return network


def get_optim_configurations(args, network, train_loader, start_epoch=0):
    print("lr: {}, momentum: {}, decay: {}".format(args.lr, args.momentum, args.weight_decay))
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    # Optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)

    else:
        optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs, eta_min=args.min_lr)

    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)

    rec = init_recorder()

    return criterion, optimizer, scheduler, rec
