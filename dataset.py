import os
import json
import csv
from PIL import Image
from torch.utils.data import Dataset

# 字符到标签的映射：0-9 -> 0-9, A-Z -> 10-35
import string
digits = [str(i) for i in range(10)]
uppercase = list(string.ascii_uppercase)
chars = digits + uppercase
char2idx = {ch: idx for idx, ch in enumerate(chars)}

class CSVJSONImageDataset(Dataset):
    """
    结合 CSV 和 JSON 划分的 Dataset，支持：
      - splits.json 中的绝对路径
      - CSV (filename, label_char) 中的相对路径或子路径
      - 通过 basename 匹配实现映射
    """
    def __init__(self,
                 splits_json: str,
                 csv_path: str,
                 split: str = 'train',
                 transform=None):
        assert split in ('train', 'val', 'test'), "split must be 'train', 'val', or 'test'"
        self.transform = transform

        # 1) 读取 JSON 划分，获取绝对路径列表
        with open(splits_json, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"Split '{split}' not found in {splits_json}")
        json_paths = splits[split]

        # 2) 读取 CSV 标签映射（使用 basename 作为 key）
        basename2label = {}
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row['image']
                label_char = row['label'].strip().upper()
                if label_char not in char2idx:
                    raise ValueError(f"Unknown label character '{label_char}' in CSV for file '{rel_path}'")
                base = os.path.basename(rel_path)
                basename2label[base] = char2idx[label_char]

        # 3) 构建 samples，仅保留在 JSON 列表中且 CSV 中有标签的项
        self.samples = []  # 存放 (absolute_path, int_label)
        missing = []
        for abs_path in json_paths:
            base = os.path.basename(abs_path)
            if base in basename2label:
                self.samples.append((abs_path, basename2label[base]))
            else:
                missing.append(abs_path)

        if missing:
            print(f"Warning: {len(missing)} images in split '{split}' not found in CSV labels. e.g.: {missing[:5]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # 改成 'RGB' 如需彩色
        if self.transform:
            img = self.transform(img)
        return img, label
