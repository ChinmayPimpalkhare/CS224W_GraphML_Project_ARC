import torch


def bpr_loss(s_pos, s_neg):
    return -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-12).mean()
