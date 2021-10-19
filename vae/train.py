import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
# folder scripts
from vae.loss import elbo_loss
from vae.optimizer import Adam
from vae.scheduler import StepLR
from vae.earlystopping import EarlyStopping




class Train():
    def __init__(self, epochs=2, batch_size=1, log_interval=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.device = self.set_device()
        self.num_workers = self.set_num_workers()
        self.pin_memory = self.set_pin_memory()

    def set_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        return self.device

    def set_num_workers(self):
        if self.device == "cuda":
            return 1
        else:
            return 0

    def set_pin_memory(self):
        if self.device == "cuda":
            return True
        else:
            return False
    

    def train_epoch(self, model, train_loader, optimizer):
        model.train()

        losses = []

        pbar = tqdm(total=len(train_loader.dataset))

        for batch_idx, (x, _) in enumerate(train_loader):

            x = x.to(self.device)

            # apply model on whole batch directly on device
            x_hat, mu, sigma, z = model(x)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = elbo_loss(x, mu, sigma, z, x_hat, model.log_scale)     # no need to squeeze the variables since all are on the device

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            losses.append(loss.item())

            # print training stats
            pbar.update(len(x))
            if batch_idx % self.log_interval == 0:
                s = "-- TRAIN Loss: {loss:.4f}"
                d = {
                    'loss': loss.item()
                }
                pbar.set_description(s.format(**d))

        pbar.close()

        return np.mean(losses)

    def test_epoch(self, model, test_loader):
        # put the model in the evaluation mode
        model.eval()

        loss = 0
        with torch.no_grad():  # stops autograd engine from calculating the gradients
            for x, _ in test_loader:

                x = x.to(self.device)

                # apply model on whole batch directly on device
                x_hat, mu, sigma, z = model(x)

                # save losses
                loss += elbo_loss(x, mu, sigma, z, x_hat, model.log_scale)/len(test_loader)

        s = "-- TEST  Loss: {loss:.4f}"
        d = {
            'loss': loss.item()
        }
        print(s.format(**d))

        return loss.item()

    def earlystop(self, model, loss, earlystopping=None):
        earlystop = False
        # check if earlystopping class exists
        if earlystopping:
            # update earlystopping parameters
            earlystopping(loss, model)
            # check if to stop
            if earlystopping.early_stop:
                # load best model at the end of training
                model.load_state_dict(torch.load(earlystopping.checkpoint_path)) 
                print("Early stopping")
                # force training stop
                earlystop = True
        return earlystop

    def train(self, train_set, test_set, model):
        # associate the architecture parameters to the chosen optimizer class
        optimizer = Adam()(model.parameters())
        # associate optimizer to the scheduler
        scheduler = StepLR()(optimizer)
        # create earlystopping class instance
        earlystopping = EarlyStopping(checkpoint_path='checkpoint/vae.pth',  patience=np.inf, mode="min", delta=0.001)
        # send architecture to device
        model.to(self.device)
        # set dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        train_loss = []
        test_loss = []

        for epoch in range(1, self.epochs + 1):
            print(f'Epoch: {epoch}')
            # append train metrics
            loss = self.train_epoch(model, train_loader, optimizer)
            train_loss.append(loss)
            # append test metrics
            loss = self.test_epoch(model, test_loader)
            test_loss.append(loss)
            
            # update learning rate
            scheduler.step()
            
            # check early stopping
            if self.earlystop(model, loss, earlystopping):  #change criterion to change metric to check
                break

        columns = ['train_loss', 'test_loss']
        df = pd.DataFrame(
            np.array([train_loss, test_loss]).T,
            index = np.arange(1, len(train_loss)+1),
            columns=columns
            )

        return df

    def __call__(self, train_set, test_set, model):
        return self.train(train_set, test_set, model)

