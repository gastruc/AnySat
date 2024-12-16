from lightning import LightningModule
import torch
import pandas as pd

class Module(LightningModule):
    def __init__(self, network, loss, train_metrics, val_metrics, test_metrics, scheduler, optimizer):
        super().__init__()
        self.model = network.instance
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            loss.pop("logits")
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            self.val_metrics.update(loss["logits"])
            loss.pop("logits")
        else:
            self.val_metrics.update(pred, batch)
        for metric_name, metric_value in loss.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            self.test_metrics.update(loss["logits"])
            loss.pop("logits")
        else:
            self.test_metrics.update(pred, batch)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        if "results" in metrics.keys():
            pd.DataFrame(metrics['results']).T.to_csv('results.csv')
            print("saving results dict")
            metrics.pop("results")
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
