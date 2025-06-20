# config.py
import torch


class CFG:
    data_dir = r"H:\glaucoma\test"
    batch_size = 64
    epochs = 100
    lr = 0.03
    momentum = 0.9
    weight_decay = 1e-4
    moco_momentum = 0.999
    queue_size = 65536
    num_workers = 8
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
