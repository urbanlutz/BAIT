import time

import torch
from tqdm import tqdm
from utils.monitor import MONITOR
from utils.state import State


def train(state: State, device):
    state.model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(state.train_loader):
        state.inc_iterations()
        data, target = data.to(device), target.to(device)
        state.optimizer.zero_grad()
        output = state.model(data)
        train_loss = state.loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        state.optimizer.step()

        MONITOR.track('Loss/Train', train_loss, state.total_iterations)

    return total / len(state.train_loader.dataset)

def eval(state: State, device):
    state.model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in state.test_loader:
            data, target = data.to(device), target.to(device)
            output = state.model(data)
            total += state.loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()

    test_loss = total / len(state.test_loader.dataset)
    MONITOR.track('Loss/Test', test_loss, state.total_iterations)
    accuracy = 100. * correct1 / len(state.test_loader.dataset)
    MONITOR.track('Accuracy/Top1', accuracy, state.total_iterations)
    MONITOR.track('Accuracy/Top5', 100. * correct5 / len(state.test_loader.dataset), state.total_iterations)
    state.set_best_accuracy(accuracy)
    return test_loss, accuracy


class EarlyStopper:
    # inspired by https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.last_losses = []
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def should_stop(self, val_loss) -> bool:
        return self.eval_min(val_loss)

    def eval_min(self, val_loss):
        if not self.best_loss:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train_eval_loop(state:State, device, epochs) -> None:
    early_stopper = EarlyStopper()
    # capture the inits
    for name, weight in state.model.named_parameters():
        MONITOR.track_hist(name, weight, state.total_epochs)


    for epoch in tqdm(range(epochs)):

        start = time.perf_counter()
        train(state, device)
        test_loss, accuracy = eval(state, device)
        stop = time.perf_counter()
        

        MONITOR.track('Performance/Train-Eval Loop', stop-start, state.total_epochs)
        for name, weight in state.model.named_parameters():
            MONITOR.track_hist(name, weight, state.total_epochs)
        state.scheduler.step()

        if early_stopper.should_stop(test_loss):
            print(f"Stopped early at epoch {epoch}")
            break

        state.inc_epochs()
        


def get_metrics(state: State, device):
    test_loss, accuracy = eval(state, device)
    return {
        "accuracy": accuracy
    }