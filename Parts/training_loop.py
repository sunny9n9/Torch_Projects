from .imports import torch, tqdm, numpy
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ['TrainingLoop', 'TrainingLoopAdvanced', 'SaveState', 'EarlyStopping']

class EarlyStopping:
    def __init__(self, patience = 3, delta = 0.05):
        self.patience = patience
        self.best_loss = numpy.inf
        self.current_count = 0
        self.early_stop = False
        self.delta = delta # to enforce bit stronger regularization
        self.current_state = {}

    def __call__(self, current_loss, model):
        # makes instance of this class callable
        
        if current_loss >= (self.best_loss - (self.delta * self.best_loss)):
            self.current_count += 1
            if self.current_count > self.patience:
                self.early_stop = True
        else:
            self.current_count = 0
            self.best_loss = current_loss
            self.current_state = {'model_state_dict' : model.state_dict(),
                                  'lowest_loss' : self.best_loss}
            self.early_stop = False
        return self.early_stop

class SaveState:
    def __init__(self, name : str, colab : bool):
        self.model_states = {}
        self.model_name = name
        self.is_runtime_colab = colab

    def save(self, model, optmizer, epoch, trainer):
        self.model_states = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optmizer.state_dict(),
            'loss_history_epoch' : trainer.loss_history_epoch,
            'batch_loss' : trainer.loss_history, # (per epoch)
            'val_loss' : trainer.loss_val
        }
        torch.save(self.model_states, self.model_name)

        if self.is_runtime_colab: # <-----------------------------------
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
        self.loss_history_epoch = [] # <- for per epoch loss
        self.epoch = epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, data_loader):
        self.model.to(self.device)
        self.model.train() # redundancy is better than regret

        batch_size = len(data_loader)
        total_steps = batch_size * self.epoch
        with tqdm.tqdm(total=total_steps) as bar:
            for epoch in range(self.epoch):
                loss_this_epoch = 0
                for batch_number, (data, label) in enumerate(data_loader):
                    data = data.to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad()

                    prediction = self.model(data)
                    loss_this_batch = self.loss(label, prediction)
                    loss_this_batch.backward()

                    self.optimizer.step()

                    loss_this_epoch += loss_this_batch.item()

                    bar.update(1)
                    bar.set_postfix_str(f":: current loss :: {loss_this_batch.item()} :: current epoch {epoch} :: current batch {batch_number}")
                    self.loss_history.append(loss_this_batch.item())
                self.loss_history_epoch.append(loss_this_epoch)
        

class TrainingLoopAdvanced:
    def __init__(self, model, loss, optimizer, epoch, earlystopper : EarlyStopping = None, saver : SaveState = None, lrscheduler : torch.optim.lr_scheduler = None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.loss_history = [] # <- for per batch loss
        self.loss_history_epoch = [] # <- for per epoch loss (avg)
        self.loss_val = [] # <- for validation loss (per epoch, avg)
        self.epoch = epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.early_stopping = earlystopper
        self.save = saver
        self.save_epoch = 2
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=2) if lrscheduler is None else lrscheduler 
        
        self.improvement = 0.05 # <- 5% improvement at least on each iteration

    def train(self, data_loader, val_data_loader = None):
        '''
        beware all losses are accumulated, not averaged per sample, but per batch loss
        may cause problem if batch size vary in future
        '''
        self.model.to(self.device)
        self.model.train() # redundancy is better than regret

        batch_size = len(data_loader)
        total_steps = batch_size * self.epoch
        
        with tqdm.tqdm(total=total_steps) as bar:
            for epoch in range(self.epoch):
                loss_this_epoch = 0
                
                for batch_number, (data, label) in enumerate(data_loader): ### BEWARE IN CASE OF DIFFERENT LEN BATCHES (for logging loss)
                    data = data.to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad()

                    prediction = self.model(data)
                    loss_this_batch = self.loss(label, prediction)
                    loss_this_batch.backward()

                    self.optimizer.step()

                    loss_this_epoch += loss_this_batch.item()

                    bar.update(1)
                    # Update bar in-place using postfix_str
                    bar.set_postfix_str(f"Loss:: {loss_this_batch.item()} :: Epoch :: {epoch} :: Batch :: {batch_number}")
                    
                    self.loss_history.append(loss_this_batch.item() * len(data)) # * len() to have weighted append incase diff sized batches
                
                loss_this_epoch_normalized = loss_this_epoch / len(data_loader.dataset) # .dataset as each batch = all samples
                self.loss_history_epoch.append(loss_this_epoch_normalized)

                # Validation and Early Stopping Logic
                if val_data_loader:
                    val_loss = self.validate(val_data_loader)
                    bar.write(f":: Epoch {epoch} :: Train Loss {loss_this_epoch_normalized} :: Val Loss {val_loss}")

                    # might as well tune learning rate
                    if self.lr_scheduler is not None:
                        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                            self.lr_scheduler.step(val_loss)
                        else:
                            self.lr_scheduler.step()  # CosineAnnealingLR
                    
                    if self.early_stopping is not None:
                        if self.early_stopping(val_loss, self.model):
                            self._save(epoch)
                            bar.write(f":: Early Stopping Triggered @ epoch {epoch}")
                            break
                else:  
                    bar.write(f":: current loss :: {loss_this_epoch_normalized} :: current epoch {epoch}")

                # Periodic Saving
                if self.save:
                    if epoch % self.save_epoch == 0:
                        # Use val_loss if available, else use train loss
                        monitor_loss = self.loss_val[-1] if self.loss_val else loss_this_epoch_normalized
                        self._save(epoch)

            # Final Save after loop completion (if no early stopping used)
            if self.save:
                final_monitor_loss = self.loss_val[-1] if self.loss_val else self.loss_history_epoch[-1]
                self._save(numpy.inf)
                
        
    def validate(self, val_data_loader):
        self.model.eval() 
        val_loss = 0 
        with torch.no_grad():
            for data, label in val_data_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                prediction = self.model(data)
                loss = self.loss(label, prediction)
                val_loss += (loss.item() * len(data))
                
        val_loss_normalized = val_loss / len(val_data_loader.dataset)
        self.loss_val.append(val_loss_normalized)
        self.model.train() 
        return val_loss_normalized
    
    def _save(self, epoch):
        self.save(self.model, self.optimizer, epoch, self)
        