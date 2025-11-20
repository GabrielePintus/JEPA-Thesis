import torch
from lightning.pytorch.cli import LightningCLI
from src.data.datamodule import PointMazeSequencesDataModule
from src.isometry import Isometry
import os

SEED = 0
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == "__main__":
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Lightning CLI
    LightningCLI(
        model_class=Isometry,
        datamodule_class=PointMazeSequencesDataModule,
        seed_everything_default=SEED,
        save_config_kwargs={"overwrite": True},
    )
