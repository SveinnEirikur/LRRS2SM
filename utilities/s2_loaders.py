import numpy as np
import scipy.io as spio
import h5py
import os


# Helper function to load Matlab cell arrays
def load_matlab_cell_array(filepath, var_name, verbose=False):
    v = None
    try:
        # Newer mat files use h5py
        with h5py.File(filepath, "r") as matfile:
            v = matfile[var_name][()][0]
            v = [np.transpose(matfile[r][()]) for r in v]
            if verbose:
                print("Loaded hdf5 file")
    except OSError:
        # Older mat files use Scipy
        mat = spio.loadmat(filepath)
        v = mat[var_name]
        v = [r[0] for r in v]
        if verbose:
            print("Loaded mat file")
    finally:
        assert v is not None
        return v


# Helper function to load Matlab arrays
def load_matlab_array(filepath, var_name, verbose=False):
    v = None
    try:
        # Newer mat files use h5py
        with h5py.File(filepath, "r") as matfile:
            # Will load 2D or 3D matrixes
            try:
                v = np.transpose(matfile[var_name][()], (2, 1, 0))
            except ValueError:
                v = np.transpose(matfile[var_name][()], (1, 0))
            if verbose:
                print("Loaded hdf5 file")
    except OSError:
        # Older mat files use Scipy
        mat = spio.loadmat(filepath)
        v = mat[var_name]
        if verbose:
            print("Loaded mat file")
    finally:
        assert v is not None
        return v


# Helper function to load S2 images from mat-files
def get_data(dataset_name, datadir="../dataset", verbose=False, rr=False, get_mtf=False):
    Yim = None
    eval_bands = [2, 6]
    mtf = [0.32, 0.26, 0.28, 0.24, 0.38, 0.34, 0.34, 0.26, 0.33, 0.26, 0.22, 0.23]
    # [.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23]

    if dataset_name == "apex":
        Yim = load_matlab_cell_array(os.path.join(datadir, "apex.mat"), "Yim", verbose)
        Xm_im = load_matlab_array(os.path.join(datadir, "apex.mat"), "Xm_im", verbose)
    elif dataset_name == "aviris":
        Yim = load_matlab_cell_array(os.path.join(datadir, "aviris.mat"), "Yim", verbose)
        Xm_im = load_matlab_array(os.path.join(datadir, "aviris.mat"), "imGT", verbose)
    elif dataset_name == "crops":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "avirisLowCrops.mat"), "Yim", verbose
        )
        Xm_im = load_matlab_array(
            os.path.join(datadir, "avirisLowCrops.mat"), "Xm_im", verbose
        )
    elif dataset_name == "coast":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "avirisLowCoast.mat"), "Yim", verbose
        )
        Xm_im = load_matlab_array(
            os.path.join(datadir, "avirisLowCoast.mat"), "Xm_im", verbose
        )
    elif dataset_name == "escondido":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "escondido.mat"), "Yim", verbose
        )
        Xm_im = load_matlab_array(
            os.path.join(datadir, "escondido.mat"), "Xm_im", verbose
        )
    elif dataset_name == "escondido_s2":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "escondido.mat"), "S2", verbose
        )
        Xm_im = np.zeros((Yim[1].shape[0], Yim[1].shape[1], len(Yim))) * np.nan
        eval_bands = None
    elif dataset_name == "mountain":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "avirisLowMontain.mat"), "Yim", verbose
        )
        Xm_im = load_matlab_array(
            os.path.join(datadir, "avirisLowMontain.mat"), "Xm_im", verbose
        )
    elif dataset_name == "rkvik":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "rkvik_crop.mat"), "Yim", verbose
        )
        Xm_im = np.zeros((Yim[1].shape[0], Yim[1].shape[1], len(Yim))) * np.nan
        eval_bands = None
    elif dataset_name == "rkvik_rr_2":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "rkvik_crop.mat"), "Yim_rr_2", verbose
        )
        Xm_im = load_matlab_array(
            os.path.join(datadir, "rkvik_crop.mat"), "Xm_im", verbose
        )
        eval_bands = [2]
    elif dataset_name == "rkvik_rr_6":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "rkvik_crop.mat"), "Yim_rr_6", verbose
        )
        Xm_im = load_matlab_array(
            os.path.join(datadir, "rkvik_crop.mat"), "Xm_im", verbose
        )
        eval_bands = [6]
    elif dataset_name == "urban":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "avirisLowCity.mat"), "Yim", verbose
        )
        Xm_im = load_matlab_array(
            os.path.join(datadir, "avirisLowCity.mat"), "Xm_im", verbose
        )

    if Yim is None:
        raise Exception("Unknown dataset: " + dataset_name)

    if rr:
        from s2synth import rr_s2_data

        Xm_im = Yim
        Yim = rr_s2_data(Xm_im, ratio=rr)
        eval_bands = [rr]

    if get_mtf:
        return (Yim, mtf, Xm_im, eval_bands)

    return (Yim, Xm_im, eval_bands)


def get_s2l8_data(
    dataset_name, datadir="../dataset/jakob_s2_l8", verbose=False, rr=False, get_mtf=False
):
    Yim = None
    eval_bands = [2, 6]
    d = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2]
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
    # [.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23]

    if dataset_name == "escondido":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "escondido_l8.mat"), "Yim", verbose
        )
        Xm_im = load_matlab_array(
            os.path.join(datadir, "escondido_l8.mat"), "Xm_im", verbose
        )
    elif dataset_name == "page":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "page_s2_l8.mat"), "Yim", verbose
        )
        Xm_im = np.zeros((Yim[1].shape[0], Yim[1].shape[1], len(Yim))) * np.nan
        eval_bands = None
    elif dataset_name == "rkvik":
        Yim = load_matlab_cell_array(
            os.path.join(datadir, "rkvik_s2_l8.mat"), "Yim", verbose
        )
        Xm_im = np.zeros((Yim[1].shape[0], Yim[1].shape[1], len(Yim))) * np.nan
        eval_bands = None

    if Yim is None:
        raise Exception("Unknown dataset: " + dataset_name)

    if rr:
        from utilities.s2synth import rr_s2_data

        N = {2: 12, 3: 15, 6: 18}
        Xm_im = Yim
        Yim = rr_s2_data(Xm_im, ratio=rr, N=N[rr])
        Xm_im = [x if s == rr else np.nan * Yim[1] for x, s in zip(Xm_im, d)]
        Xm_im = np.stack(Xm_im)
        eval_bands = [rr]

    if get_mtf:
        return (Yim, mtf, Xm_im, eval_bands)

    return (Yim, Xm_im, eval_bands)
