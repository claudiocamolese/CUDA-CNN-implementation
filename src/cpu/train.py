from tqdm import tqdm
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss


class Trainer:
    def __init__(self, epochs, lr, train_loader, device):
        self.epochs = epochs
        self.lr = lr
        self.train_loader = train_loader
        self.device = device

        self.criterion = CrossEntropyLoss()

    def train(self, model):
        model.to(self.device)

        optimizer = SGD(model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-5
        )

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0

            loop = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]")

            for img, label in loop:
                img = img.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()

                pred = model(img)
                loss = self.criterion(pred, label)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(loss=running_loss / (loop.n + 1))

            scheduler.step()
            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch avg loss: {avg_loss:.6f}")

            