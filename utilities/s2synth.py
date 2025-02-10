import numpy as np


def gaussian_filter(N=3, sigma=0.5):
    n = (N - 1) / 2.0
    y, x = np.ogrid[-n : n + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma**2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def create_conv_kernel(sdf, nl, nc, d=None, N=18):
    if d is None:
        d = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2]
    L = len(d)
    B = np.zeros([L, nl, nc])
    for i in range(L):
        if d[i] == 1:
            B[i, 0, 0] = 1
        else:
            h = gaussian_filter(N, sdf[i])
            B[
                i,
                int(nl % 2 + 1 + (nl - N - d[i]) // 2) : int(
                    nl % 2 + 1 + (nl + N - d[i]) // 2
                ),
                int(nc % 2 + 1 + (nc - N - d[i]) // 2) : int(
                    nc % 2 + 1 + (nc + N - d[i]) // 2
                ),
            ] = h
            # B[i, int(nl%2 + N%2 + (nl - N)//2):int(nl%2 + N%2 + (nl + N)//2),
            #   int(nc%2 + N%2 + (nc - N)//2):int(nc%2 + N%2 + (nc + N)//2)] = h

            B[i, :, :] = np.fft.fftshift(B[i, :, :])
            B[i, :, :] = np.divide(B[i, :, :], np.sum(B[i, :, :]))
    FBM = np.fft.fft2(B)

    return FBM


def rr_s2_data(Yim, ratio=2, mtf=None, d=None, N=18, trim=True):
    if d is None:
        d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2])
    if mtf is None:
        mtf = [
            0.32,
            0.26,
            0.28,
            0.24,
            0.38,
            0.34,
            0.34,
            0.26,
            0.33,
            0.26,
            0.22,
            0.23,
            0.4896,
            0.4884,
            0.4872,
            0.4843,
            0.4806,
            0.4615,
            0.4544,
            0.1,
            0.4612,
        ]
    Yim_c = np.array(Yim, dtype=object)

    idx1 = np.nonzero(np.array([1])[:, None] == d)[1]
    nl1, nc1 = Yim[idx1[0]].shape

    idx2 = np.nonzero(np.array([2])[:, None] == d)[1]
    nl2, nc2 = Yim[idx2[0]].shape

    idx3 = np.nonzero(np.array([3])[:, None] == d)[1]
    nl3, nc3 = Yim[idx3[0]].shape

    idx6 = np.nonzero(np.array([6])[:, None] == d)[1]
    nl6, nc6 = Yim[idx6[0]].shape

    l_trim1_l = int(6 * (np.floor(nl6 % 6 / 2))) if trim else 0
    l_trim1_r = int(6 * np.ceil(nl6 % 6 / 2)) if trim else 0
    c_trim1_l = int(6 * (np.floor(nc6 % 6 / 2))) if trim else 0
    c_trim1_r = int(6 * np.ceil(nc6 % 6 / 2)) if trim else 0

    l_trim2_l = l_trim1_l // 2
    l_trim2_r = l_trim1_r // 2
    c_trim2_l = c_trim1_l // 2
    c_trim2_r = c_trim1_r // 2
    l_trim3_l = l_trim1_l // 3
    l_trim3_r = l_trim1_r // 3
    c_trim3_l = c_trim1_l // 3
    c_trim3_r = c_trim1_r // 3
    l_trim6_l = l_trim1_l // 6
    l_trim6_r = l_trim1_r // 6
    c_trim6_l = c_trim1_l // 6
    c_trim6_r = c_trim1_r // 6

    sdf = ratio * np.sqrt(-2 * np.log(mtf) / np.pi**2)

    nl1 = int(nl1 - l_trim1_r - l_trim1_l)
    nc1 = int(nc1 - c_trim1_r - c_trim1_l)
    d1 = ratio * d[idx1] / 1
    fbm1 = create_conv_kernel(sdf[idx1], nl1, nc1, d1, N=N)

    nl2 = int(nl2 - l_trim2_r - l_trim2_l)
    nc2 = int(nc2 - c_trim2_r - c_trim2_l)
    d2 = ratio * d[idx2] / 2
    fbm2 = create_conv_kernel(sdf[idx2], nl2, nc2, d2, N=N)

    nl3 = int(nl3 - l_trim3_r - l_trim3_l)
    nc3 = int(nc3 - c_trim3_r - c_trim3_l)
    d3 = ratio * d[idx3] / 3
    fbm3 = create_conv_kernel(sdf[idx3], nl3, nc3, d3, N=N)

    nl6 = int(nl6 - l_trim6_r - l_trim6_l)
    nc6 = int(nc6 - c_trim6_r - c_trim6_l)
    d6 = ratio * d[idx6] / 6
    fbm6 = create_conv_kernel(sdf[idx6], nl6, nc6, d6, N=N)

    Yim_rr = np.empty((len(mtf),))
    Yim_rr1 = np.real(
        np.fft.ifft2(
            np.fft.fft2(
                np.stack(Yim_c[idx1])[
                    :,
                    l_trim1_l : None if l_trim1_r == 0 else -l_trim1_r,
                    c_trim1_l : None if c_trim1_r == 0 else -c_trim1_r,
                ]
            )
            * fbm1
        )
    )
    Yim_rr2 = np.real(
        np.fft.ifft2(
            np.fft.fft2(
                np.stack(Yim_c[idx2])[
                    :,
                    l_trim2_l : None if l_trim2_r == 0 else -l_trim2_r,
                    c_trim2_l : None if c_trim2_r == 0 else -c_trim2_r,
                ]
            )
            * fbm2
        )
    )
    Yim_rr3 = np.real(
        np.fft.ifft2(
            np.fft.fft2(
                np.stack(Yim_c[idx3])[
                    :,
                    l_trim3_l : None if l_trim3_r == 0 else -l_trim3_r,
                    c_trim3_l : None if c_trim3_r == 0 else -c_trim3_r,
                ]
            )
            * fbm3
        )
    )
    Yim_rr6 = np.real(
        np.fft.ifft2(
            np.fft.fft2(
                np.stack(Yim_c[idx6])[
                    :,
                    l_trim6_l : None if l_trim6_r == 0 else -l_trim6_r,
                    c_trim6_l : None if c_trim6_r == 0 else -c_trim6_r,
                ]
            )
            * fbm6
        )
    )

    Yim_rr = {}
    for i in range(len(d)):
        if i in idx1:
            Yim_rr[i] = Yim_rr1[np.where(idx1 == i), ::ratio, ::ratio]
        if i in idx2:
            Yim_rr[i] = Yim_rr2[np.where(idx2 == i), ::ratio, ::ratio]
        if i in idx3:
            Yim_rr[i] = Yim_rr3[np.where(idx3 == i), ::ratio, ::ratio]
        if i in idx6:
            Yim_rr[i] = Yim_rr6[np.where(idx6 == i), ::ratio, ::ratio]

    return [np.squeeze(Yim_rr[key]) for key in Yim_rr]


def mod_6_crop_s2_data(Yim, d=None):
    if d is None:
        d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2])
    Yim_c = np.array(Yim, dtype=object)
    nl6, nc6 = Yim[d.tolist().index(6)].shape

    l_trim1_l = int(6 * (np.floor(nl6 % 6 / 2)))
    l_trim1_r = int(6 * np.ceil(nl6 % 6 / 2))
    c_trim1_l = int(6 * (np.floor(nc6 % 6 / 2)))
    c_trim1_r = int(6 * np.ceil(nc6 % 6 / 2))
    l_trim2_l = l_trim1_l // 2
    l_trim2_r = l_trim1_r // 2
    c_trim2_l = c_trim1_l // 2
    c_trim2_r = c_trim1_r // 2
    l_trim3_l = l_trim1_l // 3
    l_trim3_r = l_trim1_r // 3
    c_trim3_l = c_trim1_l // 3
    c_trim3_r = c_trim1_r // 3
    l_trim6_l = l_trim1_l // 6
    l_trim6_r = l_trim1_r // 6
    c_trim6_l = c_trim1_l // 6
    c_trim6_r = c_trim1_r // 6

    Yim_cc = {}
    for i in range(len(d)):
        if d[i] == 1:
            Yim_cc[i] = Yim_c[i][
                l_trim1_l : None if l_trim1_r == 0 else -l_trim1_r,
                c_trim1_l : None if c_trim1_r == 0 else -c_trim1_r,
            ]
        elif d[i] == 2:
            Yim_cc[i] = Yim_c[i][
                l_trim2_l : None if l_trim2_r == 0 else -l_trim2_r,
                c_trim2_l : None if c_trim2_r == 0 else -c_trim2_r,
            ]
        elif d[i] == 3:
            Yim_cc[i] = Yim_c[i][
                l_trim3_l : None if l_trim3_r == 0 else -l_trim3_r,
                c_trim3_l : None if c_trim3_r == 0 else -c_trim3_r,
            ]
        elif d[i] == 6:
            Yim_cc[i] = Yim_c[i][
                l_trim6_l : None if l_trim6_r == 0 else -l_trim6_r,
                c_trim6_l : None if c_trim6_r == 0 else -c_trim6_r,
            ]

    return [np.squeeze(Yim_cc[key]) for key in Yim_cc]


def pad_to_size(Yim, pad_size=(432, 432), band_scales=None):
    if band_scales is None:
        band_scales = np.array(
            [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2]
        )
    padding = np.maximum(np.subtract(pad_size, Yim[1].shape), 0) // 6 * 3
    Ypad = np.copy(Yim)

    if any(padding > 0):
        for idx in range(len(band_scales)):
            Ypad[idx] = np.pad(
                Ypad[idx],
                (
                    (padding[0], padding[0]) // band_scales[idx],
                    (padding[1], padding[1]) // band_scales[idx],
                ),
                "reflect",
            )

    return Ypad, padding


def unpad_from_size(Xpad, padding=(0, 0)):
    Xim = np.copy(Xpad)

    if padding[0] > 0:
        Xim = Xim[padding[0] : -padding[0], :, :]
    if padding[1] > 0:
        Xim = Xim[:, padding[1] : -padding[1], :]

    return Xim
