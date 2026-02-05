import torch

from tqdm import tqdm
from torch.nn import CrossEntropyLoss

class Tester:
    def __init__(self, test_loader, device):
        self.test_loader = test_loader
        self.device = device
        self.criterion = CrossEntropyLoss

    @torch.no_grad()
    def test(self, model):
        model.to(self.device)
        model.eval()

        running_loss = 0.0
        num_batches = len(self.test_loader)

        loop = tqdm(self.test_loader, desc="Testing")

        for img, label in loop:
            img = img.to(self.device)
            label = label.to(self.device)

            pred = model(img)

            if self.criterion is not None:
                loss = self.criterion(pred, label)
                running_loss += loss.item()
                loop.set_postfix(loss=running_loss / (loop.n + 1))

        if self.criterion is not None:
            avg_loss = running_loss / num_batches
            print(f"Test avg loss: {avg_loss:.6f}")
            return avg_loss

        return None
