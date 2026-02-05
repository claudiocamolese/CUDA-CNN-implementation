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
from test import Tester


def main(args):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.device
    epoch = 5
    lr = 1e-2

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
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './datasets/mnist/',
            train= False, 
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size= 64,      
        shuffle= False,
        num_workers= 2,       
        pin_memory= True)     


    trainer = Trainer(epochs= epoch, lr= lr, train_loader= train_loader, device= device)
    tester = Tester(test_loader= test_loader, device= device)
    model = Net()

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    model = trainer.train(model=model)

    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"Time for training {epoch} is: {end_time - start_time:.3f} s")

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    tester.test(model= model)

    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parser.parse_args()
    main(args)