import torch


def padding_mask(lens):
    """Mask out the blank (padding) values
    Args:
        lens: (bs,)
    Returns:
        mask: (bs, 1, max_len)
    """
    bs, max_len = len(lens), max(lens)
    mask = torch.zeros(bs, 1, max_len)
    for i, l in enumerate(lens):
        mask[i, :, :l] = 1
    mask = mask > 0
    return mask
