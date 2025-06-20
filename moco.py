# moco.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MoCo(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 在线 encoder
        self.encoder_q = models.resnet50(pretrained=False)
        self.encoder_q.fc = nn.Identity()
        # 动量 encoder
        self.encoder_k = models.resnet50(pretrained=False)
        self.encoder_k.fc = nn.Identity()
        # 初始化 momentum encoder 参数
        for q_param, k_param in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False

        # 队列 (queue)
        self.register_buffer("queue", torch.randn(cfg.queue_size, 2048))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.m = cfg.moco_momentum  # EMA momentum
        self.T = 0.2                # Temperature

    @torch.no_grad()
    def _momentum_update(self):
        """用 EMA 更新 momentum encoder 参数"""
        for q_param, k_param in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k_param.data = k_param.data * self.m + q_param.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """把新的 key 特征放入队列，并移除最早的一批"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[ptr:ptr+batch_size, :] = keys
        ptr = (ptr + batch_size) % self.queue.shape[0]
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # 1) 计算 query
        q = self.encoder_q(im_q)      # [B, 2048]
        q = nn.functional.normalize(q, dim=1)
        # 2) 计算 key
        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        # 3) 构造 logits：正样本对、负样本队列
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)      # [B,1]
        l_neg = torch.einsum(
            'nc,kc->nk', [q, self.queue.clone().detach()])  # [B, Q]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.size(
            0), dtype=torch.long).to(logits.device)

        # 4) 更新队列
        self._dequeue_and_enqueue(k)
        return logits, labels
