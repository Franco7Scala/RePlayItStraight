import random

import torch
import numpy as np

from .almethod import ALMethod
from enum import Enum


class SelectionPolicy(Enum):
    L2 = 1
    KL = 3


class ImprovedUncertainty(ALMethod):
    def __init__(self, dst_u_all, unlabeled_set, model, args, selection_method="LeastConfidence", balance=False, **kwargs):
        super().__init__(dst_u_all, unlabeled_set, model, args, **kwargs)
        selection_choices = ["LeastConfidence", "Entropy", "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")

        self.selection_method = selection_method
        self.balance = balance
        self.selection_policy = SelectionPolicy.L2

    def run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            scores = []
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_unlabeled)[self.dst_unlabeled.targets == c]
                scores.append(self.rank_uncertainty(class_index))
                selection_result = np.append(selection_result, class_index[np.argsort(scores[-1])[:round(len(class_index) * self.args.n_query)]])

        else:
            scores = self.rank_uncertainty()
            #TODO weighted random sampling here
            selection_result = np.argsort(scores)[:self.args.n_query]
            #selection_result = self.random_distributed_selection(scores, self.args.n_query)
            #selection_result = self.random_distributed_selection_mixed(scores, self.args.n_query)

        return selection_result, scores

    def random_distributed_selection(self, scores, n_query):
        total_score = sum(scores)
        normalized_scores = [val / total_score for val in scores]
        thresholds = [sum(normalized_scores[:i + 1]) for i in range(len(normalized_scores))]
        selected_indexes = []
        for i in range(n_query):
            hook = random()
            for j in range(len(thresholds)):
                if hook < thresholds[j]:
                    if j in selected_indexes:
                        i -= 1

                    else:
                        selected_indexes.append(j)
                        break

        return selected_indexes

    def random_distributed_selection_mixed(self, scores, n_query):
        max_num = max(scores)
        scores = [abs((num * 2) - max_num) for num in scores]

        total_score = sum(scores)
        normalized_scores = [val / total_score for val in scores]
        thresholds = [sum(normalized_scores[:i + 1]) for i in range(len(normalized_scores))]
        selected_indexes = []
        for i in range(n_query):
            hook = random()
            for j in range(len(thresholds)):
                if hook < thresholds[j]:
                    if j in selected_indexes:
                        i -= 1

                    else:
                        selected_indexes.append(j)
                        break

        return selected_indexes

    def rank_uncertainty(self, index=None):
        self.model.eval()
        with (torch.no_grad()):

            # indices = list(range(len(self.dst_unlabeled)))  # Create a list of indices
            # random.shuffle(indices)
            # indices = indices[:len(indices) // 2]  # Integer division to get half
            # train_loader = torch.support.data.DataLoader(torch.support.data.Subset(self.dst_unlabeled, indices), batch_size=self.args.test_batch_size, num_workers=self.args.workers)

            train_loader = torch.utils.data.DataLoader(self.dst_unlabeled if index is None else torch.utils.data.Subset(self.dst_unlabeled, index), batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            scores = np.array([])
            batch_num = len(train_loader)
            for i, (input, output) in enumerate(train_loader):
                y_pred = self.model(input.to(self.args.device))
                if i % self.args.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))

                if self.selection_method == "LeastConfidence":
                    current_scores = y_pred.max(axis=1).values.cpu().numpy()

                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(y_pred, dim=1).cpu().numpy()
                    current_scores = (np.log(preds + 1e-6) * preds).sum(axis=1)

                elif self.selection_method == "Margin":
                    preds = torch.nn.functional.softmax(y_pred, dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    current_scores = (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy()

                # Adding label knowledge
                y_target = (torch.eye(self.args.n_class)[output]).to(self.args.device)
                if self.selection_policy == SelectionPolicy.L2:
                    diff = torch.linalg.norm(y_target - y_pred, dim=1)

                elif self.selection_policy == SelectionPolicy.KL:
                    diff = torch.nn.functional.kl_div(y_pred, y_target, reduction="none").sum(dim=1)

                current_scores = current_scores + diff.cpu().numpy()
                scores = np.append(scores, current_scores)

        return scores

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_indices = [self.unlabeled_set[idx] for idx in selected_indices]
        return Q_indices, scores
