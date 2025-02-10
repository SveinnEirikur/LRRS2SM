import torch
import json
import numpy as np

from einops import reduce


def ConvCM(X,FKM):
    return torch.fft.ifft2(torch.fft.fft2(X)*FKM).real


def StandardizeChannels(x, ch_mean=None, ch_std=None, un_mean=None, un_std=None):
    if isinstance(x, (list, tuple)):
        y_list = [StandardizeChannels(im) for im in x]
        y, ch_mean, ch_std = zip(*y_list)
        return y, torch.stack(ch_mean).unsqueeze(0), torch.stack(ch_std).unsqueeze(0)

    if un_mean is None:
        if ch_mean is None:
            ch_std = torch.std(x, dim=(-2, -1), keepdim=True)
            ch_mean = torch.mean(x, dim=(-2, -1), keepdim=True)
        y = (x - ch_mean) / ch_std
        return y, ch_mean, ch_std
    else:
        y = x * un_std + un_mean
        return y


def CalcGradientNorm(module):
    """
    Calculates the L2 norm of gradients for all parameters in a module.
    
    Args:
        module (torch.nn.Module): Module containing parameters with gradients
        
    Returns:
        float: The L2 norm of all gradients concatenated into a single vector
    """
    total_norm = 0.0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def MtMx(x, d):
    if not torch.is_tensor(d):
        d = torch.tensor(d)
    M = torch.zeros([1, len(d), x.shape[-2], x.shape[-1]]).type_as(x)
    for s in torch.unique(d):
        M[:, d == int(s), :: int(s), :: int(s)] = 1
    y = M * x
    return y


def gaussian_filter(N=18, sigma=0.5):
    if not isinstance(sigma, (list, tuple, np.ndarray)) and not isinstance(
        N, (list, tuple, np.ndarray)
    ):
        n = (N - 1) / 2.0
        y, x = np.ogrid[-n : n + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma**2))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    elif not isinstance(N, (list, tuple, np.ndarray)):
        return [gaussian_filter(N, s) for s in sigma]
    elif not isinstance(sigma, (list, tuple, np.ndarray)):
        return [gaussian_filter(n, sigma) for n in N]
    else:
        return [gaussian_filter(n, s) for n, s in zip(N, sigma)]


def gaussian_filters(N=7, sigmas=[0.42, 0.35, 0.35, 0.36, 0.2, 0.24]):
    return np.stack([gaussian_filter(N, sigma) for sigma in sigmas])


def create_conv_kernel(
    sdf,
    nl,
    nc,
    d=[6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2],
    N=[18, 0, 0, 0, 12, 12, 12, 0, 12, 18, 12, 12],
):
    L = len(d)
    B = torch.zeros([L, nl, nc])
    for i in range(L):
        if d[i] == 1 or N[i] == 0:
            B[i, 0, 0] = 1
        else:
            h = torch.tensor(gaussian_filter(N[i], sdf[i]))
            B[
                i,
                int(nl % 2 + 1 + (nl - N[i] - d[i]) // 2) : int(
                    nl % 2 + 1 + (nl + N[i] - d[i]) // 2
                ),
                int(nc % 2 + 1 + (nc - N[i] - d[i]) // 2) : int(
                    nc % 2 + 1 + (nc + N[i] - d[i]) // 2
                ),
            ] = h

            B[i, :, :] = torch.fft.fftshift(B[i, :, :])
            B[i, :, :] = torch.divide(B[i, :, :], torch.sum(B[i, :, :]))
    FBM = torch.fft.fft2(B)

    return FBM


def imgrad_weights(x, bands=[1, 2, 3, 7], sigma=1.0, clip_min=0.5):
    W = torch.sqrt(
        torch.square(x[:, bands, :, :] - torch.roll(x[:, bands, :, :], -1, -1))
        + torch.square(x[:, bands, :, :] - torch.roll(x[:, bands, :, :], -1, -2))
    ).square()
    W = reduce(W, "b c h w -> b 1 h w", "max").sqrt()
    W = W / W.quantile(0.95)
    W = torch.exp(-0.5 * W**2 / sigma**2).clip(clip_min)
    return W


class NumpyJsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
