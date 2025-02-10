import torch

import numpy as np

from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
from resize_right import interp_methods, resize
from pytorch_lightning import LightningDataModule

from utilities.s2_loaders import get_data, get_s2l8_data

from utilities.common import StandardizeChannels, create_conv_kernel, MtMx, imgrad_weights


class S2Dataset(Dataset):
    def __init__(
        self,
        filename,
        datadir,
        transform=None,
        target_transform=None,
        d: list = None,
        q: list = None,
        q_max: float = 30.0,
        sub_size: int = None,
        limsub: int = 2,
        training=False,
        sigma: float = 1.0,
        w_clip: float = 0.5,
        steps_per_epoch: int = 1,
    ):
        if d is None:
            d = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2]
        self.d = torch.Tensor(d)

        self.datadir = datadir
        self.filename = filename
        self.transform = transform
        self.target_transform = target_transform
        self.sub_size = sub_size
        self.training = training

        yim, mtf, xm_im = get_data(filename, datadir=datadir, get_mtf=True)[:3]

        yim = [torch.tensor(im, dtype=torch.float) for im in yim]

        yim, self.ch_means, self.ch_stds = StandardizeChannels(yim)

        output_size = [len(d), yim[1].shape[-2], yim[1].shape[-1]]
        sdf = np.array(d) * np.sqrt(-2 * np.log(np.array(mtf)) / np.pi**2)
        sdf[np.array(d) == 1] = 0
        self.sdf = np.copy(sdf)
        sdf_max = sdf.max()
        N = list(map(lambda x: {1: 0, 2: 12, 6: 18}[x], d))
        self.fft_of_B = create_conv_kernel(
            sdf, output_size[-2], output_size[-1], d=self.d, N=N
        ).unsqueeze(0)

        sdf = np.sqrt(sdf_max**2 - sdf**2)
        N = [15] * len(d)
        N[self.sdf.argmax()] = 0
        d = [6] * len(d)
        d[self.sdf.argmax()] = 1
        self.fft_of_max_B = create_conv_kernel(
            sdf, output_size[-2], output_size[-1], d=d, N=N
        ).unsqueeze(0)
        self.sdf_max = sdf

        y_nn = torch.stack(
            [
                repeat(y, "h w -> (h sh) (w sw)", sh=int(s), sw=int(s))
                for y, s in zip(yim, self.d)
            ]
        ).unsqueeze(0)
        y_b = torch.fft.ifft2(torch.fft.fft2(MtMx(y_nn, self.d)) * self.fft_of_B).real
        y_bic = rearrange(
            [
                resize(
                    im[:, :: int(s), :: int(s)],
                    interp_method=interp_methods.cubic,
                    scale_factors=int(s),
                    pad_mode="symmetric",
                    by_convs=True,
                )
                for im, s in zip(
                    [y_nn[:, idx, :, :] for idx in range(y_nn.shape[-3])], self.d
                )
            ],
            "c b h w -> b c h w",
        )

        y_bb = torch.fft.ifft2(torch.fft.fft2(y_bic) * self.fft_of_B).real

        y_maxblur = torch.fft.ifft2(torch.fft.fft2(y_bic) * self.fft_of_max_B).real
        y2n = rearrange(
            y_maxblur[:, :, limsub:-limsub, limsub:-limsub], "b c h w -> b c (h w)"
        )

        test_tensor = y2n @ y2n.mH / (yim[1].shape[-2] * yim[1].shape[-1])
        test_tensor = test_tensor.cpu()  # SVD on CUDA broken?
        self.F_init_weight = test_tensor
        _U, _S, Vh = torch.linalg.svd(test_tensor)

        if q is None:
            q = torch.cumsum(1 + torch.log(_S[0, 0] / _S), 1).squeeze()
        else:
            if isinstance(q, list):
                q = repeat(torch.tensor(q[: self.sub_size]), "c -> (c)")
            else:
                q = q.squeeze()

        q = q[q < q_max] if self.sub_size is None else q[: self.sub_size]

        self.q = repeat(q, "c -> 1 c 1 1")

        self.sub_size = self.q.shape[1]

        self.U = Vh.mH[:, :, : self.sub_size].type_as(y_b)

        W = imgrad_weights(
            y_nn, bands=(self.d == 1).nonzero().squeeze(), sigma=sigma, clip_min=w_clip
        )

        image = y_nn
        im_in = y_b
        blurry = y_bb
        target = rearrange(torch.tensor(xm_im, dtype=torch.float), "h w c -> 1 c h w")

        self.blurry = blurry
        self.image = image
        self.im_in = im_in
        self.target = target
        self.W = W

        self.num_samples = self.image.shape[0]
        if self.training:
            self.num_samples = self.num_samples * steps_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            return None
        if self.training:
            idx = 0

        image = self.image[idx]
        im_in = self.im_in[idx]
        blurry = self.blurry[idx]
        target = self.target[idx]
        ch_mean = self.ch_means[idx]
        ch_std = self.ch_stds[idx]
        fft_of_B = self.fft_of_B[0]
        U = self.U[0]
        W = self.W[0]
        q = self.q[0]
        r = self.sub_size

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return target, image, im_in, blurry, U, W, fft_of_B, ch_mean, ch_std, q, r


class S2DataModule(LightningDataModule):
    def __init__(
        self,
        datadir: str = "../dataset",
        trainfile: str = "apex",
        transform=None,
        target_transform=None,
        limsub: int = 6,
        q: list = None,
        q_max: float = 30.0,
        sub_size: int = 8,
        sigma: float = 1.0,
        w_clip: float = 0.5,
        valfile: str = "apex",
        testfile: str = "apex",
        batch_size: int = 1,
        steps_per_epoch=1,
        num_workers: int = 24,
    ):
        super().__init__()
        self.datadir = datadir
        self.trainfile = trainfile
        self.testfile = testfile
        self.valfile = valfile
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.transform = transform
        self.target_transform = target_transform
        self.limsub = limsub
        self.sub_size = sub_size
        self.sigma = sigma
        self.w_clip = w_clip
        self.q = q
        self.q_max = q_max

    def setup(self, stage: str = None):
        self.s2_train = S2Dataset(
            self.trainfile,
            self.datadir,
            transform=self.transform,
            target_transform=self.target_transform,
            sub_size=self.sub_size,
            sigma=self.sigma,
            q=self.q,
            q_max=self.q_max,
            w_clip=self.w_clip,
            steps_per_epoch=self.steps_per_epoch,
            training=True,
        )

        self.s2_test = S2Dataset(
            self.testfile,
            self.datadir,
            transform=self.transform,
            q=self.q,
            q_max=self.q_max,
            target_transform=self.target_transform,
            sub_size=self.sub_size,
        )

        self.val_dataset = S2Dataset(
            self.valfile,
            self.datadir,
            transform=self.transform,
            q=self.q,
            q_max=self.q_max,
            target_transform=self.target_transform,
            sub_size=self.sub_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.s2_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            multiprocessing_context="fork",
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            multiprocessing_context="fork",
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.s2_test,
            batch_size=1,
            num_workers=self.num_workers,
            multiprocessing_context="fork",
            persistent_workers=True,
        )


class S2L8Dataset(Dataset):
    def __init__(self, filename, datadir, transform=None, target_transform=None,
                 d: list = None, q: list = None, rr: int = False,
                 sub_size: int = 8, limsub: int = 2, training = False,
                 sigma: float = 1.0, w_clip: float = 0.5, steps_per_epoch: int = 1):

        if d is None:
            d = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2]
        self.d = torch.Tensor(d)

        self.datadir = datadir
        self.filename = filename
        self.transform = transform
        self.target_transform = target_transform
        self.sub_size = sub_size
        self.training = training

        yim, mtf, xm_im = get_s2l8_data(filename, datadir=datadir, get_mtf=True, rr=rr)[:3]

        yim = [torch.tensor(im, dtype=torch.float) for im in yim]

        yim, self.ch_means, self.ch_stds = StandardizeChannels(yim)

        output_size = [len(d), yim[1].shape[-2], yim[1].shape[-1]]
        sdf = np.array(d)*np.sqrt(-2*np.log(np.array(mtf))/np.pi**2)
        sdf[np.array(d) == 1] = 0
        self.sdf = np.copy(sdf)
        sdf_max = sdf.max()
        N = list(map(lambda x: {1: 0, 2: 12, 3:15, 6: 18}[x], d))
        self.fft_of_B = create_conv_kernel(
            sdf, output_size[-2], output_size[-1], d=self.d, N=N).unsqueeze(0)

        sdf = np.sqrt(sdf_max**2 - sdf**2)
        N = [15]*len(d)
        N[self.sdf.argmax()] = 0
        d = [6]*len(d)
        d[self.sdf.argmax()] = 1
        self.fft_of_max_B = create_conv_kernel(
            sdf, output_size[-2], output_size[-1], d=d, N=N).unsqueeze(0)
        self.sdf_max = sdf

        y_nn = torch.stack([repeat(y,
                                   "h w -> (h sh) (w sw)",
                                   sh=int(s), sw=int(s))
                            for y, s in zip(yim, self.d)]).unsqueeze(0)
        y_b = torch.fft.ifft2(torch.fft.fft2(MtMx(y_nn, self.d))
                             * self.fft_of_B).real
        y_bic = rearrange([resize(im[:,::int(s), ::int(s)],
                                  interp_method=interp_methods.cubic,
                                  scale_factors=int(s), pad_mode="symmetric",
                                  by_convs=True)
                                  for im, s in zip([y_nn[:,idx, :, :]
                                  for idx in range(y_nn.shape[-3])],
                                  self.d)], "c b h w -> b c h w")

        y_bb = torch.fft.ifft2(torch.fft.fft2(y_bic)*self.fft_of_B).real

        y_maxblur = torch.fft.ifft2(torch.fft.fft2(
            y_bic)*self.fft_of_max_B).real
        y2n = rearrange(y_maxblur[:, :, limsub:-limsub, limsub:-limsub], "b c h w -> b c (h w)")

        test_tensor = y2n@y2n.mH/(yim[1].shape[-2]*yim[1].shape[-1])
        test_tensor = test_tensor.cpu() # SVD on CUDA broken?
        self.F_init_weight = test_tensor
        _U, _S, Vh = torch.linalg.svd(test_tensor)

        if q is None:
            # evr = torch.cumsum((_S/torch.sum(_S)), 1)
            q = torch.cumsum(1 + torch.log(_S[0,0]/_S),1)
            self.q = q[q < 45]
        else:
            self.q = repeat(torch.tensor(q),
                            "c -> 1 (c)")
        self.sub_size = self.q.numel()

        self.U = Vh.mH[:, :, :self.sub_size].type_as(y_b)

        W = imgrad_weights(y_nn, bands=(self.d == 1).nonzero().squeeze(), sigma=sigma, clip_min=w_clip)

        image = y_nn
        im_in = y_b
        blurry = y_bb
        if rr:
            target = rearrange(torch.tensor(xm_im, dtype=torch.float), "c h w -> 1 c h w")
        else:
            target = rearrange(torch.tensor(xm_im, dtype=torch.float), "h w c -> 1 c h w")

        self.blurry= blurry
        self.image = image
        self.im_in = im_in
        self.target = target
        self.W = W

        self.num_samples = self.image.shape[0]
        if self.training:
            self.num_samples = self.num_samples * steps_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            return None
        if self.training:
            idx = 0

        image = self.image[idx]
        im_in = self.im_in[idx]
        blurry = self.blurry[idx]
        target = self.target[idx]
        ch_mean = self.ch_means[idx]
        ch_std = self.ch_stds[idx]
        fft_of_B = self.fft_of_B[0]
        U = self.U[0]
        W = self.W[0]
        q = self.q
        r = len(q)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return target, image, im_in, blurry, U, W, fft_of_B, ch_mean, ch_std, q, r


class S2L8DataModule(LightningDataModule):
    def __init__(self, datadir: str = "../dataset", trainfile: str = "apex",
                 transform=None, target_transform=None,
                 limsub: int = 6,
                 sub_size: int = 8,
                 rr: int = False,
                 q: list = None,
                 sigma: float = 1.0,
                 w_clip: float = 0.5,
                 valfile: str = "apex",
                 testfile: str = "apex",
                 batch_size: int = 1,
                 steps_per_epoch = 1,
                 num_workers: int = 24):
        super().__init__()
        self.datadir = datadir
        self.trainfile = trainfile
        self.testfile = testfile
        self.valfile = valfile
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.transform = transform
        self.target_transform=target_transform
        self.limsub = limsub
        self.sub_size = sub_size
        self.sigma = sigma
        self.w_clip = w_clip
        self.q = q
        self.rr = rr

    def setup(self, stage: str = None):
        self.s2_train = S2L8Dataset(
            self.trainfile, self.datadir,
            transform=self.transform,
            target_transform=self.target_transform,
            sub_size=self.sub_size,
            sigma=self.sigma,
            q=self.q,
            rr=self.rr,
            w_clip=self.w_clip,
            steps_per_epoch=self.steps_per_epoch,
            training=True)

        self.s2_test = S2L8Dataset(
            self.testfile, self.datadir,
            transform=self.transform,
            target_transform=self.target_transform,
            q=self.q,
            rr=self.rr,
            sub_size=self.sub_size)

        self.val_dataset = S2L8Dataset(
            self.valfile, self.datadir,
            transform=self.transform,
            target_transform=self.target_transform,
            q=self.q,
            rr=self.rr,
            sub_size=self.sub_size)

    def train_dataloader(self):
        return DataLoader(self.s2_train, batch_size=self.batch_size,
                           num_workers=self.num_workers, multiprocessing_context="fork", persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1,
                          num_workers=self.num_workers, multiprocessing_context="fork", persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.s2_test, batch_size=1,
                          num_workers=self.num_workers, multiprocessing_context="fork", persistent_workers=True)
