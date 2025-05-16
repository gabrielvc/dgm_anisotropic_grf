import click
import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datasources.h5 import H5Dataset
from torchvision.models import vgg16, resnet18, resnet34, resnet50, resnet101
import os
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchmetrics
from torchmetrics.classification import BinaryConfusionMatrix, BinaryROC


class BinarizedModule(torch.nn.Module):
    def __init__(self, base_module):
        super(BinarizedModule, self).__init__()
        layers = list(base_module(weights=False).children())
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(inplace=True),
            *layers[:-1],
        )
        self.final_layer = torch.nn.Linear(layers[-1].in_features, 1)

    def forward(self, x):
        embedding = self.net(x)
        return self.final_layer(embedding.squeeze(-1).squeeze(-1))


class GenH5Dataset(H5Dataset):

    def __init__(self, std_added: float = 0.0, **kwargs):
        super().__init__(**kwargs, data_axis=0, data_name="images")
        self.std_added = std_added
    def format_element(self, index):
        data_sample = torch.from_numpy(self._file["images"][index]).float()
        return {
            "data_sample":  data_sample + torch.randn_like(data_sample) * self.std_added,
            "label": torch.zeros((1,)),
        }


class RefH5Dataset(H5Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, data_axis=0, data_name="data")


    def format_element(self, index):
        data_sample = torch.from_numpy(self._file["data"][index]).float()
        return {
            "data_sample": data_sample,
            "label": torch.ones((1,)),
        }

class Ref2H5Dataset(H5Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, data_axis=0, data_name="data")


    def format_element(self, index):
        data_sample = torch.from_numpy(self._file["data"][index]).float()
        return {
            "data_sample": data_sample,
            "label": torch.zeros((1,)),
        }

class TrueFakeDataModule(LightningDataModule):
    def __init__(
        self,
        positive_data_dir: str = "path/to/dir",
        negative_data_dir: str = "path/to/dir",
        n_procs: int = os.cpu_count(),
        val_ptg: float = 0.05,
        batch_size: int = 32,
        seed: int = 42,
        std_added: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_procs = n_procs
        self.positive_data_dir = positive_data_dir
        self.negative_data_dir = negative_data_dir
        self.batch_size = batch_size
        self.val_ptg = val_ptg
        self.seed = seed
        self.std_added = std_added

    def setup(self, stage: str):
        if stage == "fit":

            datasets = []
            for dataset_path in os.listdir(self.positive_data_dir):
                datasets.append(
                    RefH5Dataset(
                        path=os.path.join(self.positive_data_dir, dataset_path))
                )
            ref_dataset = ConcatDataset(datasets)
            
            gen_dataset = GenH5Dataset(path=self.negative_data_dir, std_added=self.std_added)
            self.train_pos, self.val_pos = random_split(
                ref_dataset,
                lengths=[1 - self.val_ptg, self.val_ptg],
                generator=torch.Generator().manual_seed(self.seed),
            )
            self.train_neg, self.val_neg = random_split(
                gen_dataset,
                lengths=[1 - self.val_ptg, self.val_ptg],
                generator=torch.Generator().manual_seed(self.seed + 1),
            )

    def train_dataloader(self):
        return DataLoader(
            ConcatDataset([self.train_pos, self.train_neg]),
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.n_procs,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            ConcatDataset([self.val_pos, self.val_neg]),
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.n_procs,
            shuffle=True,
        )


class Classifier(LightningModule):
    def __init__(self, model, effective_batch_size):
        super().__init__()
        self.model = model
        self.train_loss = torch.nn.BCEWithLogitsLoss(reduce="sum")
        self.test_loss = torch.nn.BCEWithLogitsLoss(reduce="mean")
        self.effective_batch_size = effective_batch_size
        self.val_outputs = {"prediction": [], "label": []}

    def training_step(self, batch, batch_idx):

        outs = self.model(batch["data_sample"])
        loss = self.train_loss(outs.flatten(), batch["label"].flatten())

        self.log(
            "train_loss", loss.item() / batch["data_sample"].shape[0], prog_bar=True
        )
        return loss / self.effective_batch_size

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        logit_outs = self.model(batch["data_sample"])
        probs = torch.nn.functional.sigmoid(logit_outs)
        loss = self.test_loss(logit_outs.flatten(), batch["label"].flatten())
        acc = torchmetrics.functional.accuracy(
            probs.flatten(), batch["label"].flatten(), task="binary"
        )
        auc = torchmetrics.functional.auroc(
            probs.flatten(), batch["label"].flatten().long(), task="binary"
        )
        f1 = torchmetrics.functional.f1_score(
            probs.flatten(), batch["label"].flatten(), task="binary"
        )
        dev_05 = probs - 0.5

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("acc", acc, on_epoch=True, sync_dist=True)
        self.log("auc", auc, on_epoch=True, sync_dist=True)
        self.log("f1", f1, on_epoch=True, sync_dist=True)
        self.log("mse_05", (dev_05**2).mean(), on_epoch=True, sync_dist=True)
        self.log("abs_05", torch.abs(dev_05).mean(), on_epoch=True, sync_dist=True)
        self.val_outputs["prediction"].append(probs.flatten())
        self.val_outputs["label"].append(batch["label"].flatten().long())
    

    def on_validation_end(self):
        predictions = torch.cat(self.val_outputs["prediction"])
        label = torch.cat(self.val_outputs["label"])
        self.val_outputs["prediction"].clear()
        self.val_outputs["label"].clear()
        #Binary confusion matrix
        bcm = BinaryConfusionMatrix().to(self.device)
        bcm.update(predictions, label)
        self.logger.experiment.add_figure("confusion_matrix", bcm.plot(labels=["generated", "true"])[0], self.current_epoch)

        #ROC curve
        roc = BinaryROC()
        roc.update(predictions, label)
        self.logger.experiment.add_figure("ROC", roc.plot()[0], self.current_epoch)
	#pvalue
        self.logger.experiment.add_scalar("c2st_p_value", ((predictions > .5) == label).float().mean().item() - 0.5, self.current_epoch)
        # #Precision at Recall
        # pfixed_recall_90 = torchmetrics.functional.precision_at_fixed_recall(preds=predictions, target=label, task="binary", min_recall=.90)[0]
        # pfixed_recall_50 = torchmetrics.functional.precision_at_fixed_recall(preds=predictions, target=label, task="binary", min_recall=.50)[0]
        # self.logger.experiment.add_scalar("precision_at_90_recall", pfixed_recall_90, self.current_epoch)
        # self.logger.experiment.add_scalar("precision_at_50_recall", pfixed_recall_50, self.current_epoch)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=3e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-8, T_max=1000)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "interval": "epoch"}


@click.command()
@click.option(
    "--ref_data_path",
    help="Path to reference dataset",
    metavar="STR",
    type=str,
    required=True,
)
@click.option(
    "--gen_data_path",
    help="Generated data path",
    metavar="DIR",
    type=str,
    required=True,
)
@click.option(
    "--batch_size",
    help="Maximum batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=256,
    show_default=True,
)
@click.option(
    "--model_type",
    help="Model type",
    metavar="str",
    type=str,
    default="resnet18",
    show_default=True,
)
@click.option(
    "--seed",
    help="Seed",
    metavar=int,
    type=int,
    default=42,
    show_default=True,
)
@click.option(
    "--acc_grads",
    help="how many gradients to accumulate",
    metavar=int,
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "--ckpt_path",
    help="Checkpoint path",
    metavar=str,
    type=str,
    default="",
    show_default=True
)
def cmdline(
    ref_data_path,
    gen_data_path,
    batch_size,
    model_type,
    seed,
    acc_grads,
    ckpt_path,
    **opts,
):
    if model_type == "resnet18":
        base_module = resnet18
    elif model_type == "resnet34":
        base_module = resnet34
    elif model_type == "resnet50":
        base_module = resnet50
    elif model_type == "resnet101":
        base_module = resnet101
    else:
        raise NotImplemented
    
    torch.manual_seed(2025)
    model = BinarizedModule(base_module)
    ds = TrueFakeDataModule(positive_data_dir=ref_data_path, negative_data_dir=gen_data_path, seed=seed, val_ptg=0.5, batch_size=batch_size, n_procs=3)
    ds.setup("fit")
    clf = Classifier(model, effective_batch_size=batch_size * acc_grads)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(
            save_dir="lightning_logs/classifiers",
            name=f"{model_type}",
            version=f"{seed}",
        )
    checkpoint_callback = ModelCheckpoint(
            dirpath=f"lightning_models/{model_type}/{seed}/checkpoints"
        )
    trainer = Trainer(accumulate_grad_batches=acc_grads, logger=logger, callbacks=[lr_monitor, checkpoint_callback], max_epochs=1000)
    trainer.fit(clf, train_dataloaders=ds.train_dataloader(), val_dataloaders=ds.val_dataloader(), ckpt_path=ckpt_path if ckpt_path != "" else None)
    
if __name__ == "__main__":
    cmdline()
