import torch
import torch.nn as nn


def channel(channel_type='AWGN', snr=20):
    def AWGN_channel(z_hat: torch.Tensor):
        if z_hat.dim() == 4:
            # k = np.prod(z_hat.size()[1:])
            k = torch.prod(torch.tensor(z_hat.size()[1:]))
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        elif z_hat.dim() == 3:
            # k = np.prod(z_hat.size())
            k = torch.prod(torch.tensor(z_hat.size()))
            sig_pwr = torch.sum(torch.abs(z_hat).square()) / k
        noi_pwr = sig_pwr / (10 ** (snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
        return z_hat + noise

    def Rayleigh_channel(z_hat: torch.Tensor):
        z_shape = z_hat.shape
        device = z_hat.device
        if z_hat.dim() == 4:
            batch_size = z_hat.shape[0]
            z_hat = z_hat.reshape((batch_size, -1, 2))
            # k = np.prod(z_hat.size()[1:])
            k = torch.prod(torch.tensor(z_hat.size()[1:], device=device))
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2), keepdim=True) / k
            h = torch.normal(torch.zeros((batch_size, 1, 2), device=device),
                             torch.ones((batch_size, 1, 2), device=device) * torch.sqrt(
                                 torch.tensor(0.5, device=device)))
        elif z_hat.dim() == 3:
            z_hat = z_hat.reshape((-1, 2))
            # k = np.prod(z_hat.size()[1:])
            k = torch.prod(torch.tensor(z_hat.size(), device=device))
            sig_pwr = torch.sum(torch.abs(z_hat).square()) / k
            h = torch.normal(torch.zeros((1, 2), device=device),
                             torch.ones((1, 2), device=device) * torch.sqrt(torch.tensor(0.5, device=device)))
            # print(h)

        else:
            raise Exception("Bad Z shape")

        noi_pwr = sig_pwr / (10 ** (snr / 10))
        n = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
        # print(z_hat.device, k.device, h.device, n.device)
        # print(h)
        z_hat = h * z_hat + n
        z_hat = z_hat.reshape(z_shape)
        return z_hat

    def Rician_channel(z_hat: torch.Tensor, eps=1):
        z_shape = z_hat.shape
        device = z_hat.device
        if z_hat.dim() == 4:
            batch_size = z_hat.shape[0]
            z_hat = z_hat.reshape((batch_size, -1, 2))
            # k = np.prod(z_hat.size()[1:])
            k = torch.prod(torch.tensor(z_hat.size()[1:], device=device))
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2), keepdim=True) / k
            h = torch.normal(torch.zeros((batch_size, 1, 2), device=device),
                             torch.ones((batch_size, 1, 2), device=device) * torch.sqrt(
                                 torch.tensor(0.5, device=device)))
            # print(h)
        elif z_hat.dim() == 3:
            z_hat = z_hat.reshape((-1, 2))
            # k = np.prod(z_hat.size()[1:])
            k = torch.prod(torch.tensor(z_hat.size(), device=device))
            sig_pwr = torch.sum(torch.abs(z_hat).square()) / k
            h = torch.normal(torch.zeros((1, 2), device=device),
                             torch.ones((1, 2), device=device) * torch.sqrt(torch.tensor(0.5, device=device)))
            # print(h)

        else:
            raise Exception("Bad Z shape")

        noi_pwr = sig_pwr / (10 ** (snr / 10))
        n = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
        # print(z_hat.device, k.device, h.device, n.device)
        # print(h)
        # z_hat = h * z_hat + n
        z_hat = torch.sqrt(torch.tensor(1 / (1 + eps), device=device)) * h * z_hat \
                + torch.sqrt(torch.tensor(eps / (1 + eps), device=device)) * z_hat + n
        z_hat = z_hat.reshape(z_shape)
        return z_hat

    def Impulse_channel(z_hat: torch.Tensor, eps=0.5, imp_ratio=0.05):
        z_shape = z_hat.shape
        device = z_hat.device
        batch_size = z_hat.shape[0]
        z_hat = z_hat.reshape((-1,))
        data_len = z_hat.shape[0]
        imp_a = int(data_len * imp_ratio)
        sig_pwr = torch.sum(torch.abs(z_hat).square()) / data_len
        noi_pwr = sig_pwr / (10 ** (snr / 10))
        pwr_g = noi_pwr * (1 - eps)
        pwr_i = noi_pwr * eps / (imp_ratio ** 2)
        k = torch.poisson(torch.Tensor([imp_a])).int()
        noi_idx = torch.randperm(data_len)[:k].int().to(device)
        noise = torch.randn_like(z_hat) * torch.sqrt(pwr_g)
        noise_impulse = torch.randn(noi_idx.shape).to(device) * torch.sqrt(pwr_i)
        noise[noi_idx] += noise_impulse
        z_hat = z_hat + noise
        z_hat = z_hat.reshape(z_shape)
        # print(noise[noise > 2])
        return z_hat

    if channel_type == 'AWGN':
        return AWGN_channel
    elif channel_type == 'Rayleigh':
        return Rayleigh_channel
    elif channel_type == 'Rician':
        return Rician_channel
    elif channel_type == 'Impulse':
        return Impulse_channel
    else:
        raise Exception('Unknown type of channel')
