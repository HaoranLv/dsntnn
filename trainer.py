import torch
import os
import numpy as np


def fit(train_loader, val_loader, model, euc_loss_fn, reg_loss_fn, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, save_path='./models'):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    Val_loss=10000000
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, euc_loss_fn, reg_loss_fn, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, euc_loss_fn, reg_loss_fn, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        if Val_loss>val_loss:
            Val_loss=val_loss
            torch.save(model, os.path.join(save_path, '{}-{}.ckpt'.format(epoch,val_loss)))
            print('save model to {}'.format(os.path.join(save_path, '{}-{}.ckpt'.format(epoch,val_loss))))
        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        # for metric in metrics:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model, euc_loss_fn, reg_loss_fn, loss_fn, optimizer, cuda, log_interval, metrics):
    # for metric in metrics:
    #     metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        coords, heatmaps =  model(*data)
        euc_losses = euc_loss_fn(coords, target)
        # Per-location regularization losses
        sigma_t=torch.Tensor([1.0]).cuda()
        reg_losses = reg_loss_fn(heatmaps, target, sigma_t=sigma_t)
        # Combine losses into an overall loss
        loss = loss_fn(euc_losses + reg_losses)
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # for metric in metrics:
        #     metric(outputs, target, loss)

        # if batch_idx % log_interval == 0:
        #     message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         batch_idx * len(data[0]), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), np.mean(losses))
        #     for metric in metrics:
        #         message += '\t{}: {}'.format(metric.name(), metric.value())

        #     print(message)
        #     losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, euc_loss_fn, reg_loss_fn, loss_fn, cuda, metrics):
    with torch.no_grad():
        # for metric in metrics:
        #     metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            coords, heatmaps =  model(*data)
            euc_losses = euc_loss_fn(coords, target)
            # Per-location regularization losses
            sigma_t=torch.Tensor([1.0]).cuda()
            reg_losses = reg_loss_fn(heatmaps, target, sigma_t=sigma_t)
            # Combine losses into an overall loss
            loss = loss_fn(euc_losses + reg_losses)

            # if type(outputs) not in (tuple, list):
            #     outputs = (outputs,)
            # loss_inputs = outputs
            # if target is not None:
            #     target = (target,)
            #     loss_inputs += target

            # loss_outputs = loss_fn(*loss_inputs)
            # loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            # for metric in metrics:
            #     metric(outputs, target, loss_outputs)

    return val_loss, metrics
