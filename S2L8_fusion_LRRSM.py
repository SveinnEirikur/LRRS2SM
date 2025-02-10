"""
Main training script for the fusion part Learned Reduced-Rank Sharpening Method for
Multiresolution Satellite Imagery.

This module implements the training pipeline for the sharpening method described in:
Armannsson, S.E., Ulfarsson, M.O., Sigurdsson, J. (2025). A Learned Reduced-Rank
Sharpening Method for Multiresolution Satellite Imagery. Remote Sensing, 17(3), 432.

The script handles:
- Dataset loading and preprocessing for Sentinel-2 imagery
- Model initialization with configurable hyperparameters
- Training loop execution with optional logging
- Model evaluation and testing
- Checkpoint management

The model combines traditional model-based methods with neural network optimization
through a customized U-Net architecture and specialized loss function for
unsupervised single-image sharpening.
"""

import os
import wandb

from argparse import ArgumentParser
from datetime import datetime, timezone

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utilities.datasets import S2L8DataModule
from models.LRRSM_net import LRRSM_fusion_net


def main(hparams):
    """
    Main training pipeline for the reduced-rank sharpening method.

    Handles multiple training runs with configurable parameters, including:
    - Data loading and preprocessing
    - Model initialization with PCA/SVD-based weight initialization
    - Training execution with optional WandB logging
    - Model evaluation and checkpoint management

    Args:
        hparams: Namespace object containing hyperparameters including:
            - dataset: Name of the dataset to process
            - datadir: Directory containing the dataset
            - lamb: Regularization strength parameter
            - lr: Learning rate for U-Net weights
            - epochs: Number of training epochs
            - log_to_cloud: Whether to log metrics to WandB
            - Other training and model configuration parameters

    Returns:
        tuple: (trained_model, dataset, trainer) containing:
            - trained_model: The trained model
            - dataset: The DataModule used for training
            - trainer: The PyTorch Lightning trainer object
    """
    # Set random seed for reproducibility
    seed_everything(42)

    if hparams.rr:
        hparams.eval_metrics = False
    for _i in range(hparams.num_runs):
        if hparams.testdata is None:
            hparams.testdata = hparams.dataset

        # Configure logging - either to WandB cloud or local
        if hparams.log_to_cloud:
            wandb_logger = WandbLogger(project=hparams.projectname, tags=hparams.tags)
        else:
            wandb_logger = False
            hparams.log_every_n_steps = 0

        # Initialize data module with configured preprocessing steps
        s2_data = S2L8DataModule(
            hparams.datadir,
            hparams.dataset,
            valfile=hparams.dataset,
            testfile=hparams.testdata,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            sub_size=hparams.sub_size,
            sigma=hparams.sigma,
            w_clip=hparams.w_clip,
            rr=hparams.rr,
            steps_per_epoch=hparams.steps_per_epoch,
            q=hparams.default_q,
        )

        s2_data.setup("fit")

        # Initialize data module with configured preprocessing steps
        if hparams.ortho_constrain == "svd":
            initial_F = s2_data.s2_train.F_init_weight
        else:
            initial_F = s2_data.s2_train.U

        output_size = s2_data.s2_test.image.shape[-2:]

        hparams.sub_size = len(s2_data.s2_train.q)

        if hparams.log_to_cloud:
            wandb_logger.log_hyperparams(hparams)

        model = LRRSM_fusion_net(
            lr=hparams.lr,
            lamb=hparams.lamb,
            sub_size=hparams.sub_size,
            initial_F=initial_F,
            ortho_constrain=hparams.ortho_constrain,
            save_test_preds=hparams.save_test_preds,
            rr=hparams.rr,
            init_slope=hparams.init_slope,
            q=hparams.default_q,
            learn_F=hparams.learn_F,
            output_size=output_size,
            log_to_cloud=hparams.log_to_cloud,
            limsub=hparams.limsub,
            eval_metrics=hparams.eval_metrics,
        )

        # Configure model checkpoint callback
        if hparams.log_to_cloud:
            checkpoint_callback = ModelCheckpoint(
                save_top_k=2,
                every_n_train_steps=hparams.checkpoint_interval,
                monitor="step_loss",
                mode="min",
            )
        else:
            # Local checkpoint saving with timestamp-based directory
            utc_datetime = datetime.now(timezone.utc)
            utc_timestamp = utc_datetime.timestamp()
            timestring = datetime.fromtimestamp(utc_timestamp).strftime(
                "%Y-%m-%dT%H%M%SZ"
            )
            dirpath = os.path.join("./checkpoints/", timestring + hparams.dataset)
            checkpoint_callback = ModelCheckpoint(
                save_top_k=2,
                every_n_train_steps=hparams.checkpoint_interval,
                monitor="step_loss",
                mode="min",
                dirpath=dirpath,
            )

        # Initialize trainer with configured accelerator and logging options
        trainer = Trainer(
            enable_model_summary=False,
            enable_progress_bar=True,
            max_epochs=hparams.epochs,
            limit_val_batches=hparams.val_batches,
            logger=wandb_logger,
            accelerator=hparams.accelerator,
            devices=hparams.devices,
            gradient_clip_val=hparams.g_clip,
            callbacks=[checkpoint_callback],
            log_every_n_steps=hparams.log_every_n_steps,
        )

        # Execute training if requested
        if hparams.train:
            trainer.fit(model, s2_data)

        # Run evaluation on test set if specified
        if hparams.test and hparams.testdata is not None:
            try:
                trainer.test(model, s2_data, "best")
            except KeyError as err:
                print("KeyError:", err)

        if hparams.log_to_cloud:
            wandb_logger.finalize("success")
            wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    # Training configuration
    parser.add_argument(
        "--accelerator",
        default="gpu",
        type=str,
        help="Accelerator type for training (gpu/cpu)",
    )
    parser.add_argument("--devices", default=[0], nargs="*", type=int)
    parser.add_argument(
        "--dataset", default="apex", type=str, help="Name of dataset for training"
    )
    parser.add_argument("--testdata", type=str, help="Name of dataset for testing")
    parser.add_argument(
        "--datadir",
        default="../dataset/jakob_s2_l8",
        type=str,
        help="Directory containing input datasets",
    )
    # Logging configuration
    parser.add_argument("--projectname", default="unet-sweep", type=str)
    parser.add_argument("--regularizer", default="default", type=str)
    parser.add_argument("--tags", nargs="*", default="sweep", type=str)

    parser.add_argument("--sub_size", default=10, type=int)
    parser.add_argument("--lamb", default=0.5, type=float)
    parser.add_argument("--init_slope", default=0.75, type=float)
    parser.add_argument("--w_clip", default=0.5, type=float)
    parser.add_argument("--sigma", default=1.0, type=float)
    parser.add_argument("--lr", default=8e-4, type=float)
    parser.add_argument("--g_clip", default=None, type=float)
    parser.add_argument("--ortho_constrain", default=None, type=str)

    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--steps_per_epoch", default=200, type=int)
    parser.add_argument("--log_every_n_steps", default=None, type=int)
    parser.add_argument("--checkpoint_interval", default=None, type=int)
    parser.add_argument("--val_batches", default=0, type=int)
    parser.add_argument("--patch_repeat", default=0, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--num_runs", default=1, type=int)
    parser.add_argument("--limsub", default=6, type=int)
    parser.add_argument("--rr", default=False, type=int)

    parser.add_argument("--save_test_preds", dest="save_test_preds", action="store_true")
    parser.set_defaults(save_test_preds=False)

    parser.add_argument("--fix_F", dest="learn_F", action="store_false")
    parser.set_defaults(learn_F=True)

    parser.add_argument(
        "--default_q",
        dest="default_q",
        action="store_const",
        const=[1, 1.5, 4, 8, 15, 15, 20],
    )
    parser.set_defaults(default_q=None)

    parser.add_argument("--train", dest="train", action="store_true")
    parser.set_defaults(train=False)

    parser.add_argument("--eval_metrics", dest="eval_metrics", action="store_true")
    parser.set_defaults(eval_metrics=False)

    parser.add_argument("--test_metrics", dest="test_metrics", action="store_true")
    parser.set_defaults(test_metrics=False)

    parser.add_argument("--log_to_cloud", dest="log_to_cloud", action="store_true")
    parser.set_defaults(log_to_cloud=False)

    parser.add_argument("--test", dest="test", action="store_true")
    parser.set_defaults(test=False)

    args = parser.parse_args()

    main(args)
