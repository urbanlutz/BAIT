import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from utils.monitor import MONITOR
def train(model, loss, optimizer:torch.optim.Optimizer, dataloader, device, epoch, verbose, log_interval=1):
    
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        # if verbose & (batch_idx % log_interval == 0):
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(dataloader.dataset),
        #         100. * batch_idx / len(dataloader), train_loss.item()))
        if (batch_idx % log_interval == 0):
            MONITOR.track('Loss/Train', train_loss, state.total_iteratoins )
            MONITOR.inc_iteration()
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5


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
        return self.eval_min_avg(val_loss)

    def eval_min(self, val_loss):
        if not self.best_loss:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                return True
        return False
    
    def eval_min_avg(self, val_loss):
        # is it lower than the rolling average over the last n (patience) losses?
        self.last_losses = [val_loss] + self.last_losses[:self.patience - 1]
        return val_loss > (sum(self.last_losses)/len(self.last_losses))


def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]

    early_stopper = EarlyStopper()

    for epoch in tqdm(range(epochs)):
        start = time.perf_counter()


        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        stop = time.perf_counter()
        MONITOR.track('Loss/Test', test_loss)
        MONITOR.track('Accuracy/Top1', accuracy1)
        MONITOR.track('Accuracy/Top5', accuracy5)
        MONITOR.track('Performance/Train-Eval Loop', stop-start)
        MONITOR.track_hist("W", model.fc.weight)
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)

        if early_stopper.should_stop(test_loss):
            print(f"Stopped early at epoch {epoch}")
            break
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)


