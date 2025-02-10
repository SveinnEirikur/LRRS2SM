"""
Implementation of the Learned Reduced-Rank Sharpening neural network models.

This module implements the core neural network architectures described in:
Armannsson, S.E., Ulfarsson, M.O., Sigurdsson, J. (2025). A Learned Reduced-Rank
Sharpening Method for Multiresolution Satellite Imagery. Remote Sensing, 17(3), 432.

Key components:
- LRRSM_S2_net: Main model for Sentinel-2 image sharpening
- LRRSM_fusion_net: Extended model for Sentinel-2/Landsat-8 fusion
- Custom loss functions combining reconstruction fidelity and regularization
- Specialized training procedures with adaptive parameter initialization

The models use a customized U-Net architecture with instance normalization and PReLU
activation, combined with a learned linear transformation in a reduced-rank subspace.
Training is performed in an unsupervised manner using only the input image.
"""

import torch
import wandb
import os
import json

from pytorch_lightning import LightningModule
from einops import asnumpy, rearrange, reduce, repeat
from torch import nn
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from numpy import savez
from datetime import datetime, timezone


from torch.nn import L1Loss, MSELoss, Parameter
from torch.nn import functional as F

from torchmetrics import MeanAbsoluteError, MetricCollection
from torchmetrics.image import (
    ErrorRelativeGlobalDimensionlessSynthesis,
    StructuralSimilarityIndexMeasure,
)

from models.unet import UNet_s2, UNet_l8
from models.OrthLayers import OrthogonalConv2d
from utilities.common import (
    ConvCM,
    MtMx,
    StandardizeChannels,
    CalcGradientNorm,
    NumpyJsonEncoder,
)
from utilities.custom_metrics import (
    MeanAbsoluteGradientError,
    SignalReconstructionError,
    SignalReconstructionError_channel_mean,
    FStepCost,
    GStepCost,
    OldStepCost,
)
from utilities.fsreval import EvaluatePerformance, EvaluateFusionPerformance


class LRRSM_S2_net(LightningModule):
    """
    Neural network model for Sentinel-2 image sharpening using learned reduced-rank representation.

    The model combines a U-Net architecture for generating low-rank spatial representations
    with a learned linear transformation to produce the final sharpened image. Training is
    performed using an unsupervised loss that combines reconstruction fidelity with
    regularization in the low-rank subspace.

    Key Features:
        - Unsupervised training using only input image
        - PReLU activation with learnable parameters
        - Instance normalization for improved training stability
        - Resolution-matched pooling layers
        - Adaptive regularization based on singular values

    Args:
        sub_size (int): Dimensionality of the low-rank subspace
        output_size (tuple): Target spatial dimensions (height, width)
        q (list): Regularization strength progression factors
        alpha (torch.Tensor): Log of regularization parameters
        d (list): Downsampling factors for each band
        limsub (int): Boundary crop size for evaluation
        lr (float): Base learning rate for optimization
        lamb (float): Global regularization strength
        init_slope (float): Initial slope for PReLU activation
        learn_F (bool): Whether to optimize the linear transformation
        initial_F (Parameter): Initial weights for linear transformation
        ortho_constrain (str): Type of orthogonality constraint
        save_test_preds (bool): Whether to save test predictions
        eval_metrics (bool): Whether to compute evaluation metrics
        test_metrics (bool): Whether to compute test metrics
        log_to_cloud (bool): Whether to use WandB logging
    """

    def __init__(
        self,
        sub_size: int = 7,
        output_size: tuple = (198, 198),
        q: list = None,
        alpha: torch.Tensor = None,
        d: list = None,
        limsub: int = 6,
        lr: float = 1e-2,
        lamb: float = 0.002,
        init_slope: float = 0.75,
        learn_F: bool = False,
        initial_F: Parameter = None,
        ortho_constrain=None,
        save_test_preds: bool = False,
        eval_metrics: bool = True,
        test_metrics: bool = True,
        log_to_cloud: bool = False,
    ):
        if d is None:
            d = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2]
        super(LRRSM_S2_net, self).__init__()

        self.output_size = output_size
        self.sub_size = sub_size
        self.limsub = limsub
        self.lr = lr
        self.lamb = lamb
        self.d = torch.Tensor(d)
        self.learn_F = learn_F
        self.save_test_preds = save_test_preds
        self.eval_metrics = eval_metrics
        self.test_metrics = test_metrics
        self.log_to_cloud = log_to_cloud

        self.automatic_optimization = False

        self.training_bands_10 = (self.d == 1).nonzero().squeeze()
        self.training_bands_20 = (self.d == 2).nonzero().squeeze()
        self.training_bands_60 = (self.d == 6).nonzero().squeeze()
        self.training_bands = ((self.d == 6) + (self.d == 2)).nonzero().squeeze()

        if isinstance(q, list):
            self.q = repeat(torch.tensor(q[: self.sub_size]), "c -> 1 (c) 1 1")
        else:
            self.q = q

        if alpha is None and self.q is not None:
            self.register_buffer("alpha", torch.log(self.q))
        else:
            self.alpha = alpha

        self.init_slope = init_slope

        # Initialize U-Net with custom architecture for Sentinel-2
        self.G = UNet_s2(
            in_channels=len(d),
            out_channels=self.sub_size,
            init_features=32,
            init_slope=self.init_slope
        )

        # Apply Kaiming initialization with PReLU adjustment
        for name, param in self.G.named_parameters():
            if "conv" in name:
                if "enc" in name:
                    # Fan-in mode for contracting path
                    nn.init.kaiming_uniform_(param, a=self.init_slope, mode="fan_in")
                if "dec" in name:
                    # Fan-out mode for expanding path
                    nn.init.kaiming_uniform_(param, a=self.init_slope, mode="fan_out")

        self.ortho_constrain = ortho_constrain

        # Initialize linear transformation with orthogonality constraint
        self.F = OrthogonalConv2d(
            in_channels=self.sub_size,
            out_channels=len(d),
            kernel_size=1,
            padding=0,
            bias=False,
            ortho_constrain=ortho_constrain,
        )

        #
        self.PCA_initialized = True
        if initial_F is not None:
            self.F.weight = Parameter(
                rearrange(initial_F, "1 L r -> L r 1 1"), requires_grad=self.learn_F
            )

        if self.ortho_constrain:
            self.Ft = lambda x: F.conv2d(
                x, rearrange(self.F.get_constrained_weights(), "L r ... -> r L ...")
            )
            self.Ft_detached = lambda x: F.conv2d(
                x,
                rearrange(
                    self.F.get_constrained_weights().detach(), "L r ... -> r L ..."
                ),
            )
            self.F_detached = lambda x: F.conv2d(
                x, self.F.get_constrained_weights().detach()
            )
        else:
            self.Ft = lambda x: F.conv2d(
                x, rearrange(self.F.weight, "L r ... -> r L ...")
            )
            self.Ft_detached = lambda x: F.conv2d(
                x, rearrange(self.F.weight.detach(), "L r ... -> r L ...")
            )
            self.F_detached = lambda x: F.conv2d(x, self.F.weight.detach())

        self.G_cost_function = GStepCost()
        self.F_cost_function = FStepCost(self.d)
        self.S2_cost_function = OldStepCost(self.d)
        self.U_loss = L1Loss()
        self.mse_loss = MSELoss()
        self.mse_loss_20 = MSELoss()
        self.mse_loss_60 = MSELoss()
        self.reg_loss = MSELoss()
        self.z_loss = L1Loss()

        # Define horizontal and vertical derivative operators for regularization
        D_h = torch.zeros(self.output_size)
        D_h[0, 0] = 1
        D_h[0, -1] = -1
        D_v = torch.zeros(self.output_size)
        D_v[0, 0] = 1
        D_v[-1, 0] = -1

        self.register_buffer("fft_of_Dh", repeat(torch.fft.fft2(D_h), "w h -> 1 1 w h"))
        self.register_buffer("fft_of_Dv", repeat(torch.fft.fft2(D_v), "w h -> 1 1 w h"))

        self.sweep_loss = MetricCollection(
            {
                "sweep_loss": SignalReconstructionError(),
            }
        )

        self.sre_loss = SignalReconstructionError_channel_mean(eps=1e-8)
        self.sre_ds_loss = SignalReconstructionError_channel_mean(eps=1e-8)

        if self.eval_metrics:
            self.metrics_20 = MetricCollection(
                {
                    "l1_loss_20": MeanAbsoluteError(),
                    "gradient_loss_20": MeanAbsoluteGradientError(),
                    "ergas_loss_20": ErrorRelativeGlobalDimensionlessSynthesis(2),
                    "sre_loss_20": SignalReconstructionError_channel_mean(n_channels=6),
                    "ssim_loss_20": StructuralSimilarityIndexMeasure(),
                }
            )

            self.metrics_60 = MetricCollection(
                {
                    "l1_loss_60": MeanAbsoluteError(),
                    "gradient_loss_60": MeanAbsoluteGradientError(),
                    "ergas_loss_60": ErrorRelativeGlobalDimensionlessSynthesis(6),
                    "sre_loss_60": SignalReconstructionError_channel_mean(n_channels=2),
                    "ssim_loss_60": StructuralSimilarityIndexMeasure(),
                }
            )

        self.val_metrics_20 = MetricCollection(
            {
                "val_l1_loss_20": MeanAbsoluteError(),
                "val_gradient_loss_20": MeanAbsoluteGradientError(),
                "val_ergas_loss_20": ErrorRelativeGlobalDimensionlessSynthesis(2),
                "val_sre_loss_20": SignalReconstructionError_channel_mean(n_channels=6),
                "val_ssim_loss_20": StructuralSimilarityIndexMeasure(),
            }
        )

        self.val_metrics_60 = MetricCollection(
            {
                "val_l1_loss_60": MeanAbsoluteError(),
                "val_gradient_loss_60": MeanAbsoluteGradientError(),
                "val_ergas_loss_60": ErrorRelativeGlobalDimensionlessSynthesis(6),
                "val_sre_loss_60": SignalReconstructionError_channel_mean(n_channels=2),
                "val_ssim_loss_60": StructuralSimilarityIndexMeasure(),
            }
        )

    def forward(self, Y) -> Any:
        """
        Forward pass of the model.

        Args:
            Y: Input image tensor of shape (B, C, H, W)

        Returns:
            tuple: (X, G) where:
                - X: Sharpened image of shape (B, C, H, W)
                - G: Low-rank representation of shape (B, sub_size, H, W)
        """

        G = self.G(Y)
        X = self.F(G)

        return X, G

    def pred_step(self, Y, ch_mean, ch_std) -> None:
        """
        Generates sharpened predictions and handles denormalization and band reordering.

        This method performs the full prediction pipeline:
        1. Generates sharpened predictions using the trained model
        2. Denormalizes outputs using provided channel statistics
        3. Recomposes the final image by combining sharpened bands with
        original high-resolution bands

        Args:
            Y (torch.Tensor): Input image tensor of shape (B, C, H, W)
            ch_mean (torch.Tensor): Channel-wise means used for standardization
            ch_std (torch.Tensor): Channel-wise standard deviations used for standardization

        Returns:
            torch.Tensor: Recomposed image tensor of shape (B, C, H, W) containing:
                - Sharpened versions of low-resolution bands
                - Original high-resolution bands (B02, B03, B04, B08)
                - All bands denormalized to original data range
        """

        pred, _ = self.forward(Y)

        pred = StandardizeChannels(pred, un_mean=ch_mean, un_std=ch_std)
        Y = StandardizeChannels(Y, un_mean=ch_mean, un_std=ch_std)

        X = rearrange(
            [
                pred[:, 0, :, :],
                Y[:, 1, :, :],
                Y[:, 2, :, :],
                Y[:, 3, :, :],
                pred[:, 4, :, :],
                pred[:, 5, :, :],
                pred[:, 6, :, :],
                Y[:, 7, :, :],
                pred[:, 8, :, :],
                pred[:, 9, :, :],
                pred[:, 10, :, :],
                pred[:, 11, :, :],
            ],
            "c b h w -> b c h w",
        )

        return X

    def loss_step(self, Y, fft_of_B):
        pred, G = self.forward(Y)

        BX = ConvCM(pred, fft_of_B)

        MtMBX = MtMx(BX, self.d)

        step_loss = self.S2S_cost_func(
            G, MtMBX, Y, self.fft_of_Dh, self.fft_of_Dv, self.lamb, self.alpha
        )
        return step_loss

    def S2S_cost_func(
        self,
        Z: torch.Tensor,
        MtMBX: torch.Tensor,
        Y: torch.Tensor,
        fft_of_Dh: torch.Tensor,
        fft_of_Dv: torch.Tensor,
        tau: torch.Tensor,
        alpha: torch.Tensor,
    ):
        """
        Computes the Sentinel-2 sharpening loss function.

        Combines reconstruction fidelity term with regularization in low-rank subspace:
        L = 0.5||Y - MBX||² + τ∑(exp(α_j)φ(g_j))

        Args:
            Z: Low-rank representation
            MtMBX: Blurred and downsampled prediction
            Y: Input image
            fft_of_Dh: Fourier transform of horizontal derivative
            fft_of_Dv: Fourier transform of vertical derivative
            tau: Global regularization strength
            alpha: Log of per-dimension regularization strengths

        Returns:
            torch.Tensor: Scalar loss value
        """

        ZhW = ConvCM(Z, fft_of_Dh.conj())
        ZvW = ConvCM(Z, fft_of_Dv.conj())
        ZhW = ConvCM(ZhW, fft_of_Dh)
        ZvW = ConvCM(ZvW, fft_of_Dv)
        grad_pen = ZhW + ZvW

        fid_term = (
            reduce(
                0.5
                * torch.linalg.vector_norm(
                    MtMx(Y, self.d) - MtMBX, dim=(-2, -1)
                ).square(),
                "... -> ",
                "mean",
            )
            / 28750
        )
        pen_term = reduce(tau * alpha.exp() * Z * grad_pen, "... -> ", "mean")

        J = fid_term + pen_term
        return J

    def training_step(self, batch, batchidx) -> STEP_OUTPUT:
        """
        Performs a single training step.

        Executes forward pass, loss computation, and optimization updates.

        Args:
            batch: Training batch containing image and metadata
            batchidx: Index of current batch

        Returns:
            STEP_OUTPUT: Loss value for logging
        """

        target, image, _im_in, blurr_x, _F, W, fft_of_B, ch_mean, ch_std, q, _r = batch

        if self.q is None:
            self.q = q
            self.alpha = torch.log(self.q)

        G_opt = self.optimizers()

        G_opt.zero_grad()

        pred, G = self.forward(image)

        BX = ConvCM(pred, fft_of_B)

        MtMBX = MtMx(BX, self.d)

        step_loss = self.S2S_cost_func(
            G, MtMBX, image, self.fft_of_Dh, self.fft_of_Dv, self.lamb, self.alpha
        )

        self.manual_backward(step_loss)
        if self.eval_metrics:
            G_grad_norm = CalcGradientNorm(self.G)
            F_grad_norm = CalcGradientNorm(self.F)
        G_opt.step()

        self.log_dict({"step_loss": step_loss})

        if self.eval_metrics:
            step_mse = self.mse_loss(BX, blurr_x)

            self.sre_ds_loss(
                MtMBX[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
                MtMx(image, self.d)[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
            )

            self.S2_cost_function(
                G, MtMBX, image, W, self.fft_of_Dh, self.fft_of_Dv, self.lamb, self.alpha
            )

            pred_n = StandardizeChannels(pred, un_mean=ch_mean, un_std=ch_std)
            self.sre_loss(
                pred_n[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
                target[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
            )

            self.metrics_20(
                pred_n[
                    :,
                    self.training_bands_20,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
                target[
                    :,
                    self.training_bands_20,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
            )

            self.metrics_60(
                pred_n[
                    :,
                    self.training_bands_60,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
                target[
                    :,
                    self.training_bands_60,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
            )

            if self.log_to_cloud:
                self.log_dict(self.metrics_20, on_step=False, on_epoch=True)
                self.log_dict(self.metrics_60, on_step=False, on_epoch=True)

                self.log_dict(
                    {
                        "step_F_loss": self.F_cost_function,
                        "step_G_loss": self.G_cost_function,
                        "step_G_grad_norm": G_grad_norm,
                        "step_F_grad_norm": F_grad_norm,
                        "step_loss_mse": step_mse,
                        "step_loss_sre": self.sre_loss,
                        "step_loss_sre_ds": self.sre_ds_loss,
                    }
                )

        return step_loss

    def test_step(self, batch, batchidx) -> None:
        """
        Performs model evaluation during the testing phase.

        Executes the full evaluation pipeline as described in the paper:
        1. Generates sharpened predictions
        2. Denormalizes outputs and reconstructs full image
        3. Computes comprehensive quality metrics when test_metrics=True:
            - SSIM (Structural Similarity Index)
            - SRE (Signal-to-Reconstruction Error)
            - RMSE (Root Mean Square Error)
            - ERGAS (Relative Dimensionless Global Error in Synthesis)
            - SAM (Spectral Angle Mapper)
            - UIQI (Universal Image Quality Index)
        4. Optionally saves predictions and visualizations to wandb or local storage

        Args:
            batch: Tuple containing:
                - target: Ground truth image if available
                - image: Input image tensor
                - Various metadata tensors for preprocessing
                - ch_mean: Channel means for denormalization
                - ch_std: Channel standard deviations for denormalization
            batchidx: Index of the current test batch

        Notes:
            - Evaluation metrics are computed within a boundary-cropped region
            (excluding limsub pixels from edges) to avoid border effects
            - When test_metrics=True, results are saved to JSON files
            - When log_to_cloud=True, predictions and metrics are logged to wandb
            - When save_test_preds=True, predictions are saved as NPZ files
        """

        target, image, _im_in, _blurr_x, _F, _W, _fft_of_B, ch_mean, ch_std, _q, _r = (
            batch
        )

        pred, _G = self.forward(image)

        pred = StandardizeChannels(pred, un_mean=ch_mean, un_std=ch_std)
        y = StandardizeChannels(image, un_mean=ch_mean, un_std=ch_std)

        x = torch.stack(
            [
                pred[:, 0, :, :],
                y[:, 1, :, :],
                y[:, 2, :, :],
                y[:, 3, :, :],
                pred[:, 4, :, :],
                pred[:, 5, :, :],
                pred[:, 6, :, :],
                y[:, 7, :, :],
                pred[:, 8, :, :],
                pred[:, 9, :, :],
                pred[:, 10, :, :],
                pred[:, 11, :, :],
            ]
        )

        if self.save_test_preds:
            if wandb.run is not None:
                run_name = wandb.run.name
            else:
                import secrets

                run_name = "run_result_" + secrets.token_hex(8)
            print("Saving results to file: " + run_name + ".npz")
            savez(run_name + ".npz", x_hat=asnumpy(x))

        if self.test_metrics:
            [ssim, sre, rmse, ergas, sam, uiqi] = EvaluatePerformance(
                asnumpy(rearrange(target.squeeze().clamp(0, 10000), "c h w -> h w c")),
                asnumpy(rearrange(x.squeeze().clamp(0, 10000), "c h w -> h w c")),
                data_range=10000.0,
                limsub=self.limsub,
            )

            self.sweep_loss(
                pred[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
                target[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 10000),
            )

            if self.log_to_cloud:
                self.log_dict(self.sweep_loss)

                self.logger.log_metrics(
                    {
                        "SSIM": ssim,
                        "SRE": sre,
                        "RMSE": rmse,
                        "ERGAS": ergas,
                        "SAM": sam,
                        "UIQI": uiqi,
                    }
                )

                with open(os.path.join(wandb.run.dir, "results.json"), "w") as f:
                    json.dump(
                        {
                            "SSIM": ssim,
                            "SRE": sre,
                            "RMSE": rmse,
                            "ERGAS": ergas,
                            "SAM": sam,
                            "UIQI": uiqi,
                        },
                        f,
                        cls=NumpyJsonEncoder,
                    )
            else:
                utc_datetime = datetime.now(timezone.utc)
                utc_timestamp = utc_datetime.timestamp()
                timestring = datetime.fromtimestamp(utc_timestamp).strftime(
                    "%Y-%m-%dT%H%M%SZ"
                )
                with open(
                    os.path.join(
                        "./results/",
                        timestring + "_SRE_" + str(sre["All"]) + "_results.json",
                    ),
                    "w",
                ) as f:
                    json.dump(
                        {
                            "SSIM": ssim,
                            "SRE": sre,
                            "RMSE": rmse,
                            "ERGAS": ergas,
                            "SAM": sam,
                            "UIQI": uiqi,
                        },
                        f,
                        cls=NumpyJsonEncoder,
                    )

        if self.log_to_cloud:
            pred = x.squeeze()

            caption = caption = [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B11",
                "B12",
            ]
            self.logger.log_image(
                key="Predicted",
                images=[
                    pred[0].clamp(0, 10000) / 10000,
                    pred[1].clamp(0, 10000) / 10000,
                    pred[2].clamp(0, 10000) / 10000,
                    pred[3].clamp(0, 10000) / 10000,
                    pred[4].clamp(0, 10000) / 10000,
                    pred[5].clamp(0, 10000) / 10000,
                    pred[6].clamp(0, 10000) / 10000,
                    pred[7].clamp(0, 10000) / 10000,
                    pred[8].clamp(0, 10000) / 10000,
                    pred[9].clamp(0, 10000) / 10000,
                    pred[10].clamp(0, 10000) / 10000,
                    pred[11].clamp(0, 10000) / 10000,
                ],
                caption=caption,
            )
            if self.eval_metrics:
                target = target.squeeze()
                self.logger.log_image(
                    key="Target",
                    images=[
                        target[0].clamp(0, 10000) / 10000,
                        target[1].clamp(0, 10000) / 10000,
                        target[2].clamp(0, 10000) / 10000,
                        target[3].clamp(0, 10000) / 10000,
                        target[4].clamp(0, 10000) / 10000,
                        target[5].clamp(0, 10000) / 10000,
                        target[6].clamp(0, 10000) / 10000,
                        target[7].clamp(0, 10000) / 10000,
                        target[8].clamp(0, 10000) / 10000,
                        target[9].clamp(0, 10000) / 10000,
                        target[10].clamp(0, 10000) / 10000,
                        target[11].clamp(0, 10000) / 10000,
                    ],
                    caption=caption,
                )

    def validation_step(self, batch, batchidx) -> None:
        target, image, _im_in, _blurr_x, _F, _W, _fft_of_B, ch_mean, ch_std, _q, _r = (
            batch
        )

        pred, _G = self.forward(image)

        pred = StandardizeChannels(pred, un_mean=ch_mean, un_std=ch_std)

        if self.log_to_cloud:
            self.val_metrics_20(
                pred[
                    :,
                    self.training_bands_20,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ],
                target[
                    :,
                    self.training_bands_20,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ],
            )

            self.val_metrics_60(
                pred[
                    :,
                    self.training_bands_60,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ],
                target[
                    :,
                    self.training_bands_60,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ],
            )

            self.log_dict(self.val_metrics_20)
            self.log_dict(self.val_metrics_60)

    def configure_optimizers(self):
        opt_params_G = [{"params": [*self.G.parameters(), *self.F.parameters()]}]
        opt_G = torch.optim.NAdam(opt_params_G, lr=self.lr)
        return opt_G


class LRRSM_fusion_net(LightningModule):
    """
    Extended model for joint Sentinel-2 and Landsat-8 image fusion.

    Builds upon LRRSM_S2_net with modifications for handling Landsat-8 bands:
    - Additional input channels for Landsat-8 bands
    - Modified pooling ratios for 30m resolution bands
    - Adjusted initialization for larger feature space
    - Modified loss function weighting for cross-sensor consistency

    The model maintains the core reduced-rank learning approach while adapting
    to the different spatial resolutions and spectral characteristics of
    Landsat-8 data.

    Args:
        sub_size (int): Dimensionality of the low-rank subspace
        output_size (tuple): Target spatial dimensions (height, width)
        q (list): Regularization strength progression factors
        alpha (torch.Tensor): Log of regularization parameters
        d (list): Downsampling factors for each band
        limsub (int): Boundary crop size for evaluation
        lr (float): Base learning rate for optimization
        lamb (float): Global regularization strength
        init_slope (float): Initial slope for PReLU activation
        learn_F (bool): Whether to optimize the linear transformation
        initial_F (Parameter): Initial weights for linear transformation
        ortho_constrain (str): Type of orthogonality constraint
        save_test_preds (bool): Whether to save test predictions
        eval_metrics (bool): Whether to compute evaluation metrics
        test_metrics (bool): Whether to compute test metrics
        log_to_cloud (bool): Whether to use WandB logging
        rr (int): Reference resolution for evaluation
    """

    def __init__(
        self,
        sub_size: int = 7,
        output_size: tuple = (198, 198),
        q: list = None,
        alpha: torch.Tensor = None,
        d: list = None,
        limsub: int = 6,
        lr: float = 1e-2,
        lamb: float = 0.002,
        init_slope: float = 0.75,
        learn_F: bool = False,
        initial_F: Parameter = None,
        ortho_constrain=None,
        save_test_preds: bool = False,
        eval_metrics: bool = True,
        test_metrics: bool = True,
        log_to_cloud: bool = False,
        rr: int = False,
    ):
        if d is None:
            d = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2, 3, 3, 3, 3, 3, 3, 3, 1, 2]
        super(LRRSM_fusion_net, self).__init__()

        self.output_size = output_size
        self.sub_size = sub_size
        self.limsub = limsub
        self.lr_G = lr
        self.lamb = lamb
        self.d = torch.Tensor(d)
        self.learn_F = learn_F
        self.save_test_preds = save_test_preds
        self.eval_metrics = eval_metrics
        self.test_metrics = test_metrics
        self.log_to_cloud = log_to_cloud
        self.rr = rr

        self.automatic_optimization = False

        self.training_bands_10 = (self.d == 1).nonzero().squeeze()
        self.training_bands_20 = (self.d == 2).nonzero().squeeze()
        self.training_bands_30 = (self.d == 3).nonzero().squeeze()
        self.training_bands_60 = (self.d == 6).nonzero().squeeze()
        self.training_bands = (
            ((self.d == 6) + (self.d == 3) + (self.d == 2)).nonzero().squeeze()
        )

        if isinstance(q, list):
            self.q = repeat(torch.tensor(q[: self.sub_size]), "c -> 1 (c) 1 1")
        else:
            self.q = q

        if alpha is None and q is not None:
            self.register_buffer("alpha", torch.log(self.q))
        else:
            self.alpha = alpha

        self.init_slope = init_slope
        # Initialize U-Net with custom architecture for Sentinel-2
        self.G = UNet_l8(
            in_channels=len(d),
            out_channels=self.sub_size,
            init_features=32,
            init_slope=self.init_slope,
        )

        # Apply Kaiming initialization with PReLU adjustment
        for name, param in self.G.named_parameters():
            if "conv" in name:
                if "enc" in name:
                    # Fan-in mode for contracting path
                    nn.init.kaiming_uniform_(param, a=self.init_slope, mode="fan_in")
                if "dec" in name:
                    # Fan-out mode for expanding path
                    nn.init.kaiming_uniform_(param, a=self.init_slope, mode="fan_out")

        self.ortho_constrain = ortho_constrain

        # Initialize linear transformation with orthogonality constraint
        self.F = OrthogonalConv2d(
            in_channels=self.sub_size,
            out_channels=len(d),
            kernel_size=1,
            padding=0,
            bias=False,
            ortho_constrain=ortho_constrain,
        )

        #
        self.PCA_initialized = True
        if initial_F is not None:
            self.F.weight = Parameter(
                rearrange(initial_F, "1 L r -> L r 1 1"), requires_grad=self.learn_F
            )

        if self.ortho_constrain:
            self.Ft = lambda x: F.conv2d(
                x, rearrange(self.F.get_constrained_weights(), "L r ... -> r L ...")
            )
            self.Ft_detached = lambda x: F.conv2d(
                x,
                rearrange(
                    self.F.get_constrained_weights().detach(), "L r ... -> r L ..."
                ),
            )
            self.F_detached = lambda x: F.conv2d(
                x, self.F.get_constrained_weights().detach()
            )
        else:
            self.Ft = lambda x: F.conv2d(
                x, rearrange(self.F.weight, "L r ... -> r L ...")
            )
            self.Ft_detached = lambda x: F.conv2d(
                x, rearrange(self.F.weight.detach(), "L r ... -> r L ...")
            )
            self.F_detached = lambda x: F.conv2d(x, self.F.weight.detach())

        self.G_cost_function = GStepCost()
        self.F_cost_function = FStepCost(self.d)
        self.S2_cost_function = OldStepCost(self.d)
        self.U_loss = L1Loss()
        self.mse_loss = MSELoss()
        self.mse_loss_20 = MSELoss()
        self.mse_loss_60 = MSELoss()
        self.reg_loss = MSELoss()
        self.z_loss = L1Loss()

        # Define horizontal and vertical derivative operators for regularization
        D_h = torch.zeros(self.output_size)
        D_h[0, 0] = 1
        D_h[0, -1] = -1
        D_v = torch.zeros(self.output_size)
        D_v[0, 0] = 1
        D_v[-1, 0] = -1

        self.register_buffer("fft_of_Dh", repeat(torch.fft.fft2(D_h), "w h -> 1 1 w h"))
        self.register_buffer("fft_of_Dv", repeat(torch.fft.fft2(D_v), "w h -> 1 1 w h"))

        self.sweep_loss = MetricCollection(
            {
                "sweep_loss": SignalReconstructionError(),
            }
        )

        self.sre_loss = SignalReconstructionError_channel_mean(eps=1e-8)
        self.sre_ds_loss = SignalReconstructionError_channel_mean(eps=1e-8)

        if self.eval_metrics:
            self.metrics_20 = MetricCollection(
                {
                    "l1_loss_20": MeanAbsoluteError(),
                    "gradient_loss_20": MeanAbsoluteGradientError(),
                    "ergas_loss_20": ErrorRelativeGlobalDimensionlessSynthesis(2),
                    "sre_loss_20": SignalReconstructionError_channel_mean(n_channels=6),
                    "ssim_loss_20": StructuralSimilarityIndexMeasure(),
                }
            )

            self.metrics_60 = MetricCollection(
                {
                    "l1_loss_60": MeanAbsoluteError(),
                    "gradient_loss_60": MeanAbsoluteGradientError(),
                    "ergas_loss_60": ErrorRelativeGlobalDimensionlessSynthesis(6),
                    "sre_loss_60": SignalReconstructionError_channel_mean(n_channels=2),
                    "ssim_loss_60": StructuralSimilarityIndexMeasure(),
                }
            )

        self.val_metrics_20 = MetricCollection(
            {
                "val_l1_loss_20": MeanAbsoluteError(),
                "val_gradient_loss_20": MeanAbsoluteGradientError(),
                "val_ergas_loss_20": ErrorRelativeGlobalDimensionlessSynthesis(2),
                "val_sre_loss_20": SignalReconstructionError_channel_mean(n_channels=6),
                "val_ssim_loss_20": StructuralSimilarityIndexMeasure(),
            }
        )

        self.val_metrics_60 = MetricCollection(
            {
                "val_l1_loss_60": MeanAbsoluteError(),
                "val_gradient_loss_60": MeanAbsoluteGradientError(),
                "val_ergas_loss_60": ErrorRelativeGlobalDimensionlessSynthesis(6),
                "val_sre_loss_60": SignalReconstructionError_channel_mean(n_channels=2),
                "val_ssim_loss_60": StructuralSimilarityIndexMeasure(),
            }
        )

    def forward(self, Y) -> Any:
        """
        Forward pass of the model.

        Args:
            Y: Input image tensor of shape (B, C, H, W)

        Returns:
            tuple: (X, G) where:
                - X: Sharpened image of shape (B, C, H, W)
                - G: Low-rank representation of shape (B, sub_size, H, W)
        """

        G = self.G(Y)
        X = self.F(G)

        return X, G

    def pred_step(self, Y, ch_mean, ch_std) -> None:
        """
        Generates sharpened predictions and handles denormalization and band reordering.

        This method performs the full prediction pipeline:
        1. Generates sharpened predictions using the trained model
        2. Denormalizes outputs using provided channel statistics
        3. Recomposes the final image by combining sharpened bands with
        original high-resolution bands

        Args:
            Y (torch.Tensor): Input image tensor of shape (B, C, H, W)
            ch_mean (torch.Tensor): Channel-wise means used for standardization
            ch_std (torch.Tensor): Channel-wise standard deviations used for standardization

        Returns:
            torch.Tensor: Recomposed image tensor of shape (B, C, H, W) containing:
                - Sharpened versions of low-resolution bands
                - Original high-resolution bands (B02, B03, B04, B08)
                - All bands denormalized to original data range
        """

        pred, _ = self.forward(Y)

        pred = StandardizeChannels(pred, un_mean=ch_mean, un_std=ch_std)
        Y = StandardizeChannels(Y, un_mean=ch_mean, un_std=ch_std)

        X = rearrange(
            [
                pred[:, 0, :, :],
                Y[:, 1, :, :],
                Y[:, 2, :, :],
                Y[:, 3, :, :],
                pred[:, 4, :, :],
                pred[:, 5, :, :],
                pred[:, 6, :, :],
                Y[:, 7, :, :],
                pred[:, 8, :, :],
                pred[:, 9, :, :],
                pred[:, 10, :, :],
                pred[:, 11, :, :],
                pred[:, 12, :, :],
                pred[:, 13, :, :],
                pred[:, 14, :, :],
                pred[:, 15, :, :],
                pred[:, 16, :, :],
                pred[:, 17, :, :],
                pred[:, 18, :, :],
                pred[:, 19, :, :],
                pred[:, 20, :, :],
            ],
            "c b h w -> b c h w",
        )

        return X

    def loss_step(self, Y, fft_of_B):
        pred, G = self.forward(Y)

        BX = ConvCM(pred, fft_of_B)

        MtMBX = MtMx(BX, self.d)

        step_loss = self.S2S_cost_func(
            G, MtMBX, Y, self.fft_of_Dh, self.fft_of_Dv, self.lamb, self.alpha
        )
        return step_loss

    def S2S_cost_func(
        self,
        Z: torch.Tensor,
        MtMBX: torch.Tensor,
        Y: torch.Tensor,
        fft_of_Dh: torch.Tensor,
        fft_of_Dv: torch.Tensor,
        tau: torch.Tensor,
        alpha: torch.Tensor,
    ):
        """
        Computes the sharpening loss function.

        Combines reconstruction fidelity term with regularization in low-rank subspace:
        L = 0.5||Y - MBX||² + τ∑(exp(α_j)φ(g_j))

        Args:
            Z: Low-rank representation
            MtMBX: Blurred and downsampled prediction
            Y: Input image
            fft_of_Dh: Fourier transform of horizontal derivative
            fft_of_Dv: Fourier transform of vertical derivative
            tau: Global regularization strength
            alpha: Log of per-dimension regularization strengths

        Returns:
            torch.Tensor: Scalar loss value
        """
        ZhW = ConvCM(Z, fft_of_Dh.conj())
        ZvW = ConvCM(Z, fft_of_Dv.conj())
        ZhW = ConvCM(ZhW, fft_of_Dh)
        ZvW = ConvCM(ZvW, fft_of_Dv)
        grad_pen = ZhW + ZvW

        fid_term = (
            reduce(
                0.5
                * torch.linalg.vector_norm(
                    MtMx(Y, self.d) - MtMBX, dim=(-2, -1)
                ).square(),
                "... -> ",
                "mean",
            )
            / 28750
        )
        pen_term = reduce(tau * alpha.exp() * Z * grad_pen, "... -> ", "mean")

        J = fid_term + pen_term
        return J

    def training_step(self, batch, batchidx) -> STEP_OUTPUT:
        """
        Performs a single training step.

        Executes forward pass, loss computation, and optimization updates.

        Args:
            batch: Training batch containing image and metadata
            batchidx: Index of current batch

        Returns:
            STEP_OUTPUT: Loss value for logging
        """

        target, image, _im_in, blurr_x, _F, W, fft_of_B, ch_mean, ch_std, q, r = batch

        if self.q is None:
            self.q = repeat(q, "1 c -> 1 (c) 1 1")
            self.alpha = torch.log(self.q)
            self.sub_size = r

        G_opt = self.optimizers()
        G_opt.zero_grad()

        pred, G = self.forward(image)

        BX = ConvCM(pred, fft_of_B)

        MtMBX = MtMx(BX, self.d)

        step_loss = self.S2S_cost_func(
            G, MtMBX, image, self.fft_of_Dh, self.fft_of_Dv, self.lamb, self.alpha
        )

        self.manual_backward(step_loss)
        if self.eval_metrics:
            G_grad_norm = CalcGradientNorm(self.G)
            F_grad_norm = CalcGradientNorm(self.F)
        G_opt.step()

        self.log_dict({"step_loss": step_loss})

        if self.eval_metrics:
            step_mse = self.mse_loss(BX, blurr_x)

            self.sre_ds_loss(
                MtMBX[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0000),
                MtMx(image, self.d)[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0000),
            )

            self.S2_cost_function(
                G, MtMBX, image, W, self.fft_of_Dh, self.fft_of_Dv, self.lamb, self.alpha
            )

            pred_n = StandardizeChannels(pred, un_mean=ch_mean, un_std=ch_std)
            self.sre_loss(
                pred_n[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0),
                target[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0),
            )

            self.metrics_20(
                pred_n[
                    :,
                    self.training_bands_20,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0),
                target[
                    :,
                    self.training_bands_20,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0),
            )

            self.metrics_60(
                pred_n[
                    :,
                    self.training_bands_60,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0),
                target[
                    :,
                    self.training_bands_60,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0),
            )

            if self.log_to_cloud:
                self.log_dict(self.metrics_20, on_step=False, on_epoch=True)
                self.log_dict(self.metrics_60, on_step=False, on_epoch=True)

                self.log_dict(
                    {
                        "step_F_loss": self.F_cost_function,
                        "step_G_loss": self.G_cost_function,
                        "step_G_grad_norm": G_grad_norm,
                        "step_F_grad_norm": F_grad_norm,
                        "step_loss_mse": step_mse,
                        "step_loss_sre": self.sre_loss,
                        "step_loss_sre_ds": self.sre_ds_loss,
                    }
                )

        return step_loss

    def test_step(self, batch, batchidx) -> None:
        """
        Performs model evaluation during the testing phase.

        Executes the full evaluation pipeline as described in the paper:
        1. Generates sharpened predictions
        2. Denormalizes outputs and reconstructs full image
        3. Computes comprehensive quality metrics when test_metrics=True:
            - SSIM (Structural Similarity Index)
            - SRE (Signal-to-Reconstruction Error)
            - RMSE (Root Mean Square Error)
            - ERGAS (Relative Dimensionless Global Error in Synthesis)
            - SAM (Spectral Angle Mapper)
            - UIQI (Universal Image Quality Index)
        4. Optionally saves predictions and visualizations to wandb or local storage

        Args:
            batch: Tuple containing:
                - target: Ground truth image if available
                - image: Input image tensor
                - Various metadata tensors for preprocessing
                - ch_mean: Channel means for denormalization
                - ch_std: Channel standard deviations for denormalization
            batchidx: Index of the current test batch

        Notes:
            - Evaluation metrics are computed within a boundary-cropped region
            (excluding limsub pixels from edges) to avoid border effects
            - When test_metrics=True, results are saved to JSON files
            - When log_to_cloud=True, predictions and metrics are logged to wandb
            - When save_test_preds=True, predictions are saved as NPZ files
        """

        target, image, _im_in, _blurr_x, _F, _W, _fft_of_B, ch_mean, ch_std, _q, _r = (
            batch
        )

        pred, _G = self.forward(image)

        pred = StandardizeChannels(pred, un_mean=ch_mean, un_std=ch_std)
        y = StandardizeChannels(image, un_mean=ch_mean, un_std=ch_std)

        x = torch.stack(
            [
                pred[:, 0, :, :],
                y[:, 1, :, :],
                y[:, 2, :, :],
                y[:, 3, :, :],
                pred[:, 4, :, :],
                pred[:, 5, :, :],
                pred[:, 6, :, :],
                y[:, 7, :, :],
                pred[:, 8, :, :],
                pred[:, 9, :, :],
                pred[:, 10, :, :],
                pred[:, 11, :, :],
                pred[:, 12, :, :],
                pred[:, 13, :, :],
                pred[:, 14, :, :],
                pred[:, 15, :, :],
                pred[:, 16, :, :],
                pred[:, 17, :, :],
                pred[:, 18, :, :],
                pred[:, 19, :, :],
                pred[:, 20, :, :],
            ]
        )

        if self.save_test_preds:
            if wandb.run is not None:
                run_name = wandb.run.name
            else:
                import secrets

                run_name = "run_result_" + secrets.token_hex(8)
            print("Saving results to file: " + run_name + ".npz")
            savez(run_name + ".npz", x_hat=asnumpy(x.clamp(0, 1.0)))

        if self.test_metrics:
            [ssim, sre, rmse, ergas, sam, uiqi] = EvaluateFusionPerformance(
                asnumpy(rearrange(target.squeeze().clamp(0, 1.0), "c h w -> h w c")),
                asnumpy(rearrange(x.squeeze().clamp(0, 1.0), "c h w -> h w c")),
                data_range=1.0,
                limsub=self.limsub,
                bands=[self.rr],
            )

            self.sweep_loss(
                pred[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0),
                target[
                    :,
                    self.training_bands,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ].clamp(0, 1.0),
            )

            if self.log_to_cloud:
                self.log_dict(self.sweep_loss)

                self.logger.log_metrics(
                    {
                        "SSIM": ssim,
                        "SRE": sre,
                        "RMSE": rmse,
                        "ERGAS": ergas,
                        "SAM": sam,
                        "UIQI": uiqi,
                    }
                )

                with open(os.path.join(wandb.run.dir, "results.json"), "w") as f:
                    json.dump(
                        {
                            "SSIM": ssim,
                            "SRE": sre,
                            "RMSE": rmse,
                            "ERGAS": ergas,
                            "SAM": sam,
                            "UIQI": uiqi,
                        },
                        f,
                        cls=NumpyJsonEncoder,
                    )
            else:
                utc_datetime = datetime.now(timezone.utc)
                utc_timestamp = utc_datetime.timestamp()
                timestring = datetime.fromtimestamp(utc_timestamp).strftime(
                    "%Y-%m-%dT%H%M%SZ"
                )
                with open(
                    os.path.join(
                        "./", timestring + "_SRE_" + str(sre["All"]) + "_results.json"
                    ),
                    "w",
                ) as f:
                    json.dump(
                        {
                            "SSIM": ssim,
                            "SRE": sre,
                            "RMSE": rmse,
                            "ERGAS": ergas,
                            "SAM": sam,
                            "UIQI": uiqi,
                        },
                        f,
                        cls=NumpyJsonEncoder,
                    )

        if self.log_to_cloud:
            pred = x.squeeze()

            caption = caption = [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B11",
                "B12",
                "L01",
                "L02",
                "L03",
                "L04",
                "L05",
                "L06",
                "L07",
                "LPAN10",
                "LPAN20",
            ]
            self.logger.log_image(
                key="Predicted", images=[p for p in pred.clamp(0, 1.0)], caption=caption
            )
            if self.eval_metrics:
                target = target.squeeze()
                self.logger.log_image(
                    key="Target",
                    images=[t for t in target.clamp(0, 1.0)],
                    caption=caption,
                )

    def validation_step(self, batch, batchidx) -> None:
        target, image, _im_in, _blurr_x, _F, _W, _fft_of_B, ch_mean, ch_std, _q, _r = (
            batch
        )

        pred, _G = self.forward(image)

        pred = StandardizeChannels(pred, un_mean=ch_mean, un_std=ch_std)

        if self.log_to_cloud:
            self.val_metrics_20(
                pred[
                    :,
                    self.training_bands_20,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ],
                target[
                    :,
                    self.training_bands_20,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ],
            )

            self.val_metrics_60(
                pred[
                    :,
                    self.training_bands_60,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ],
                target[
                    :,
                    self.training_bands_60,
                    self.limsub : -self.limsub,
                    self.limsub : -self.limsub,
                ],
            )

            self.log_dict(self.val_metrics_20)
            self.log_dict(self.val_metrics_60)

    def configure_optimizers(self):
        opt_params_G = [{"params": [*self.G.parameters(), *self.F.parameters()]}]
        opt_G = torch.optim.NAdam(opt_params_G, lr=self.lr_G)
        return opt_G
