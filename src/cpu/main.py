import time
import torch
import argparse
import os
import torch
import random
import numpy as np

from torchvision import datasets, transforms

from model import Net
from train import Trainer


def main(args):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.device
    num_workers = max(1, os.cpu_count() // 2)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './datasets/mnist/',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size= 64,      
        shuffle= True,
        num_workers= 2,       
        pin_memory= True)     

    trainer = Trainer(epochs=5, lr=1e-2, train_loader= train_loader, device= device)

    model = Net()

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    trainer.train(model=model)

    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"Training time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parser.parse_args()
    main(args)