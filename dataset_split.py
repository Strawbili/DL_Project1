import os
import glob
import random
import json

def split_images_to_json(
    image_dir: str,
    output_json: str,
    ratios: tuple = (0.7, 0.1, 0.2),
    seed: int = 42
):
    """
    按 ratios (train, val, test) 划分 image_dir 下的所有图片，
    并保存到 output_json 文件中。
    """
    # 支持的图片后缀
    exts = ["jpg", "jpeg", "png", "bmp", "gif"]
    # 收集所有图片路径
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))
    images = sorted(images)  # 保证一致的初始顺序

    # 打乱
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    # n_test 剩余
    n_test  = n - n_train - n_val

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train + n_val]
    test_imgs  = images[n_train + n_val:]

    assert len(train_imgs) == n_train
    assert len(val_imgs)   == n_val
    assert len(test_imgs)  == n_test

    # 构造字典
    split_dict = {
        "train": train_imgs,
        "val":   val_imgs,
        "test":  test_imgs
    }

    # 写入 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, ensure_ascii=False, indent=2)

    print(f"共 {n} 张图片，划分为：")
    print(f"  train: {n_train}")
    print(f"  val:   {n_val}")
    print(f"  test:  {n_test}")
    print(f"结果保存在：{output_json}")

if __name__ == "__main__":
    # 示例用法：将当前目录下的 images/ 划分后保存到 splits.json
    split_images_to_json(
        image_dir="E:\HFUT\大三下\深度学习导论\English-Handwritten-Characters-Dataset\Img",
        output_json="splits.json",
        ratios=(0.7, 0.1, 0.2),
        seed=123
    )
