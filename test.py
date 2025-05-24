import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CSVJSONImageDataset
from LeNet_5 import LeNet5


def parse_args():
    parser = argparse.ArgumentParser(description="Test script for handwritten character classification")
    parser.add_argument('--data_json', type=str, default='/home/lyh/models/DL_Project1/splits.json',
                        help='Path to the JSON file with train/val/test splits')
    parser.add_argument('--csv_path', type=str, required=False,
                        default="/home/SSD1_4T/datasets/English-Handwritten-Characters-Dataset/english.csv",
                        help='Path to the CSV file with filename,label_char columns')
    parser.add_argument('--checkpoint', type=str, required=False,
                        default="/home/lyh/models/DL_Project1/checkpoints/LeNet5/LeNet5_best_model.pth",
                        help='Path to the saved model checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Data transforms (must match training)
    transform = transforms.Compose([
        transforms.Resize((64, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2) Test dataset and loader
    test_dataset = CSVJSONImageDataset(
        splits_json=args.data_json,
        csv_path=args.csv_path,
        split='test',
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Loaded {len(test_dataset)} test samples.")

    # 3) Model
    model = LeNet5()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # 4) Loss and metrics
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == '__main__':
    main()
