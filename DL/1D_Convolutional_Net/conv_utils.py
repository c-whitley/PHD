import torch
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score, accuracy_score


class FTIR_Dataset_C(Dataset):

    def __init__(self, dataframe, y_label, transform=None, target_transform=None):
        self.y = y_label
        self.X = dataframe
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # Make X into tensor compatible with CONV1D layers
        spectra = torch.tensor(self.X[idx,:], dtype=torch.float)
        spectra = torch.rot90(spectra)#, 1, 0)#.unsqueeze(0)
        #spectra = self.X[idx,:]
        
        # Make y compatible with binary cross entropy loss
        label = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(0)#.unsqueeze(0)
    

        if self.transform:
            spectra = self.transform(spectra)
        if self.target_transform:
            label = self.target_transform(label)
        return spectra, label


class FTIR_Dataset(Dataset):

    def __init__(self, dataframe, y_label, transform=None, target_transform=None):
        self.y = y_label
        self.X = dataframe
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # Make X into tensor compatible with CONV1D layers
        spectra = torch.tensor(self.X[idx,:], dtype=torch.float).unsqueeze(-2)
        #spectra = self.X[idx,:]
        
        # Make y compatible with binary cross entropy loss
        #label = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(0)#.unsqueeze(0)
        label = torch.tensor(self.y[idx], dtype=torch.float)#.unsqueeze(0)#.unsqueeze(0)


        if self.transform:
            spectra = self.transform(spectra)
        if self.target_transform:
            label = self.target_transform(label)
        return spectra, label


class OptunaJob:

    def __init__(self):

        pass

    def conv_objective(trial, epochs=5):

        h_params = {}

        h_params['n_conv_layers'] = trial.suggest_int("n_conv_layers", 2, 5)

        h_params['chan_list'] = [1] # Single channel for first input
        h_params['chan_list'].extend([int(trial.suggest_discrete_uniform(f"num_filter_{i}", 16, 128, 16))
                        for i in range(h_params['n_conv_layers'])])

        h_params['pool_list'] = [int(trial.suggest_discrete_uniform(f"drop_{i}", 3, 9, 2))
                        for i in range(h_params['n_conv_layers'])]

        h_params['fc_neurons'] = int(trial.suggest_discrete_uniform(f"fc_neurons", 16, 512, 16))
        h_params['dropout'] = trial.suggest_float('dropout', 0.25, 0.9)

        # --------------------------------------------------------------------------------------------------

        model = SpecConvNet()
        model.Optuna_build(h_params)

        loss_fn = nn.BCELoss()

        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True) 

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            
            print(f"Epoch {epoch+1}\n-------------------------------")

            cu.train_loop(train_dataloader, model, loss_fn, optimizer)
            score = cu.test_loop(test_dataloader, model, loss_fn)

            # For pruning (stops trial early if not promising)
            trial.report(score, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return score

    def train_loop(dataloader, model, loss_fn, optimizer):

        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % int(size/10) == 0:
                loss, current = loss.item(), batch * len(X)
                
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn):
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():

            for X, y in dataloader:

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                probs = pred.numpy().squeeze()#[:,1]

                auc = roc_auc_score(y.numpy().squeeze(), probs)
                #accuracy = accuracy_score(y.numpy().squeeze(), (probs>0.5))

        test_loss /= num_batches
        correct /= size

        return auc