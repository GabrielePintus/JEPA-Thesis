import torch
from lightning.pytorch.cli import LightningCLI
from src.encoder import VisualEncoder
from src.data.datamodule import PointMazeVICRegDataModule

SEED = 0


if __name__ == "__main__":
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Lightning CLI
    LightningCLI(
        model_class=VisualEncoder,
        datamodule_class=PointMazeVICRegDataModule,
        seed_everything_default=SEED,
        save_config_kwargs={"overwrite": True},
    )
