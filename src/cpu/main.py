import time
import torch
import argparse
from torchvision import datasets, transforms

from model import Net
from train import Trainer


def main(args):
    device = args.device

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './datasets/mnist/',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size=64,      # aumenta batch size per sfruttare meglio la GPU
        #shuffle=True,
        #num_workers=4,       # dipende dal numero di core CPU
        #pin_memory=True      # trasferimento più veloce verso GPU
    )

    trainer = Trainer(
        epochs=5,
        lr=1e-2,    
        train_loader=train_loader,
        device=device
    )

    model = Net()

    # -------------------
    # TIMING START
    # -------------------
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    trainer.train(model=model)

    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    # -------------------
    # TIMING END
    # -------------------

    print(f"Training time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parser.parse_args()
    main(args)
