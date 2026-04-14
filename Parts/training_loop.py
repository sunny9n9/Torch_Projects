from .imports import torch, tqdm

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

        batch_size = len(data_loader)
        total_steps = batch_size * self.epoch
        with tqdm.tqdm(total=total_steps) as bar:
            for epoch in tqdm.tqdm(range(self.epoch)):
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
    def __init__(self, model, loss, optimizer, epoch):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.loss_history = [] # <- for per batch loss
        self.loss_hitory_epoch = [] # <- for per epoch loss
        self.loss_val = [] # <- for validation loss (per epoch)
        self.epoch = epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.improvement = 0.05 # <- 5% improvement at least on each iteration

    def train(self, data_loader, val_data_loader):
        self.model.to(self.device)

        batch_size = len(data_loader)
        total_steps = batch_size * self.epoch
        with tqdm.tqdm(total=total_steps) as bar:
            for epoch in tqdm.tqdm(range(self.epoch)):
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
            if val_data_loader:
                val_loss = self.validate(val_data_loader)
                print(f":: current loss :: {loss} :: current validaton {val_loss} :: current epoch {epoch}")
            else:  
                print(f":: current loss :: {loss} :: current epoch {epoch}")

        
    def validate(self, val_data_loader):
        self.model.eval()
        val_loss = 0 # <- per epoch val loss
        with torch.no_grad():
            for data, label in val_data_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                prediction = self.model(data)
                loss = self.loss(label, prediction)
                val_loss += loss.item()
        self.loss_val.append(val_loss)
        return val_loss