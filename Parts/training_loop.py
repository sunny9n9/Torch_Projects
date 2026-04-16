from .imports import torch, tqdm, numpy

__all__ = ['TrainingLoop', 'TrainingLoopAdvanced', 'SaveState', 'EarlyStopping']

class EarlyStopping:
    def __init__(self, patience = 3, delta = 0):
        self.patience = patience
        self.best_loss = numpy.inf
        self.current_count = 0
        self.early_stop = False
        self.delta = delta # to enforce bit stronger regularization
        self.current_state = {}

    def __call__(self, current_loss, model):
        # makes instance of this class callable
        
        if current_loss >= (self.best_loss - self.delta):
            self.current_count += 1
            if self.current_count > self.patience:
                self.early_stop = True
        else:
            self.current_count = 0
            self.best_loss = self.current_loss
            self.current_state = {'model_state_dict' : model.state_dict(),
                                  'lowest_loss' : self.best_loss}
            self.early_stop = False
        return self.early_stop

class SaveState:
    def __init__(self, name : str, colab : bool):
        self.model_states = {}
        self.model_name = name
        self.is_runtime_colab = colab

    def save(self, model, optmizer, epoch, current_loss):
        self.model_states = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optmizer.state_dict(),
            'loss': current_loss
        }
        torch.save(self.model_states, self.model_name)
        if self.is_runtime_colab:
            from google.colab import files
            files.download(self.model_name)
    def __call__(self, *args):
        self.save(*args)

class TrainingLoop:
    def __init__(self, model, loss, optimizer, epoch):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.loss_history = [] # <- for per batch loss
        self.loss_hitory_epoch = [] # <- for per epoch loss
        self.epoch = epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, data_loader):
        self.model.to(self.device)
        # self.model.train()

        batch_size = len(data_loader)
        total_steps = batch_size * self.epoch
        with tqdm.tqdm(total=total_steps) as bar:
            for epoch in range(self.epoch):
                loss_epoch = 0
                for data, label in data_loader:
                    data = data.to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad()

                    prediction = self.model(data)
                    loss = self.loss(label, prediction)
                    loss.backward()

                    self.optimizer.step()

                    loss_epoch += loss.item()

                    bar.update(1)
                self.loss_history.append(loss.item())
            self.loss_history.append(loss_epoch)
            print(f":: current loss :: {loss} :: current epoch {epoch}")
        

class TrainingLoopAdvanced:
    def __init__(self, model, loss, optimizer, epoch, earlystopping, save_iteration):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.loss_history = [] # <- for per batch loss
        self.loss_history_epoch = [] # <- for per epoch loss
        self.loss_val = [] # <- for validation loss (per epoch)
        self.epoch = epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.early_stopping = earlystopping
        self.save = save_iteration
        
        self.improvement = 0.05 # <- 5% improvement at least on each iteration

    def train(self, data_loader, val_data_loader):
        '''
        beware all losses are accumulated, not averaged per sample, but per batch loss
        may cause problem if batch size vary in future
        '''
        self.model.to(self.device)
        self.model.train()

        batch_size = len(data_loader)
        total_steps = batch_size * self.epoch
        with tqdm.tqdm(total=total_steps) as bar:
            for epoch in range(self.epoch):
                loss_epoch = 0
                for data, label in data_loader:
                    data = data.to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad()

                    prediction = self.model(data)
                    loss = self.loss(label, prediction)
                    loss.backward()

                    self.optimizer.step()

                    loss_epoch += loss.item()
                    bar.update(1)
                    self.loss_history.append(loss.item())
                self.loss_history_epoch.append(loss_epoch)
                if val_data_loader:
                    val_loss = self.validate(val_data_loader)
                    print(f":: current loss :: {loss} :: current validaton {val_loss} :: current epoch {epoch}")
                    if self.early_stopping(val_loss, self.model):
                        # then we need to stop the training loop
                        self.save(self.model, self.optimizer, epoch, val_loss)
                        break
                else:  
                    print(f":: current loss :: {loss} :: current epoch {epoch}")

        
    def validate(self, val_data_loader):
        self.model.eval() # <<<==== need to put it out of eval mode
        val_loss = 0 # <- per epoch val loss
        with torch.no_grad():
            for data, label in val_data_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                prediction = self.model(data)
                loss = self.loss(label, prediction)
                val_loss += loss.item()
        self.loss_val.append(val_loss)
        self.model.train() # <==== ready to train again
        return val_loss