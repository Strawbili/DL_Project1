import torch

def print_cuda_is_available():
    """
    检测当前环境是否支持 CUDA（GPU 加速），并打印结果。
    返回值：
        bool：如果支持 CUDA 则为 True，否则为 False。
    """
    available = torch.cuda.is_available()
    print(f"CUDA Available: {available}")
    return available

if __name__ == "__main__":
    print_cuda_is_available()