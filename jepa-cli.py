import torch
from lightning.pytorch.cli import LightningCLI
from src.jepa import JEPA
from src.data.datamodule import PointMazeDataModule

SEED = 0


if __name__ == "__main__":
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Lightning CLI
    LightningCLI(
        model_class=JEPA,
        datamodule_class=PointMazeDataModule,
        seed_everything_default=SEED,
        save_config_kwargs={"overwrite": True},
    )