import os
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CSVJSONImageDataset

from LeNet_5 import  LeNet5
from compared_nets import UNet
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for handwritten digit classification")
    parser.add_argument('--data_json', type=str, default='splits.json',
                        help='Path to the JSON file with train/val/test splits')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train',
                        help='Which split to train on (use val for validation during training)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((64, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Datasets and loaders
    train_dataset = CSVJSONImageDataset(splits_json=args.data_json, split='train', csv_path=r"/home/SSD1_4T/datasets/English-Handwritten-Characters-Dataset/english.csv",
                                        transform=transform)
    val_dataset   = CSVJSONImageDataset(splits_json=args.data_json, split='val', csv_path=r"/home/SSD1_4T/datasets/English-Handwritten-Characters-Dataset/english.csv",
                                        transform=transform)

    print(len(train_dataset))
    print(len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model placeholder
    # TODO: replace `YourModelClass` with the actual model class
    model = LeNet5()
    model = model.to(device)
    model_name = model._get_name()
    save_path = os.path.join(args.save_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / total
        val_acc = correct / total
        print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, model_name, f"{model_name}_best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best val acc: {best_val_acc:.4f}. Model saved to {save_path}\n")

if __name__ == '__main__':
    main()