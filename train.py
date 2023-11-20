from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import wandb
import argparse
import copy
from dataloader import LBSDataset
import math
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger


from utils import get_ssim_loss, get_l1_gradient_loss
from models.unet_attention import ATUNet, ATUNetS, ATUNetM


class TrainModule(pl.LightningModule):
    def __init__(self, model,):
        super(TrainModule, self).__init__()
        self.model = model
        self.loss_mse = torch.nn.MSELoss()
        self.loss_ssim = get_ssim_loss
        self.loss_grad = get_l1_gradient_loss
        self.training_step_output = None
        
        # apply kaiming initialization
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
    
    def _get_loss(self, batch):
        x, y = batch
        out, _ = self.model(x)
        
        # print(out.shape)
        # print(y.shape)
        self.training_step_output = [out[:10][:,:3], y[:10][:,:3]]
        return 1.5*self.loss_mse(out, y) + 0.5*self.loss_grad(out, y)
    
    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train/loss", loss,  prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        # self.traininig_stp_outputs.append(loss)
        return loss
        
    def on_train_epoch_end(self) -> None:
        
        self.logger.experiment.log({
        "samples": [wandb.Image(torch.concatenate([img, img2], dim=2), caption=f"{i}th pred | gt")
        for i, (img, img2) in enumerate(zip(self.training_step_output[0], self.training_step_output[1]))]
        })
        
        # api = wandb.Api()
        # run = api.run("LBS_2D/LBS")
        # for artifact in run.logged_artifacts():
        #     artifact.delete()
            #     print(f"train_loss: {avg_loss}")
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        return loss
    
    # def on_validation_epoch_end(self, outputs: STEP_OUTPUT) -> None:
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     self.log('val_loss', avg_loss)
        
    #     print(f"val_loss: {avg_loss}")
        
    def configure_optimizers(self) :
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)   
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch : 0.98 ** epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch : 0.98 ** (epoch//8))
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler, }
        
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
    

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.data_root = "./demo_data/RP_demo"
    args.run_name = "LBS_2D_new6"
    args.epochs = 500
    args.batch_size = 4
    args.image_size = 64
    args.device = "cuda"
    args.lr = 3e-4
    
    
    logdir = "./logs"
    logger = WandbLogger(name=args.run_name, project="LBS_new", save_dir=logdir,)
    checkpoint_callback = ModelCheckpoint(monitor='train/loss', save_top_k=3, save_last=True, mode = 'min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
                         max_epochs=args.epochs, 
                         precision=32,
                         accelerator='gpu', 
                         logger=logger,
                         log_every_n_steps=2,
                         callbacks=[checkpoint_callback, lr_monitor])
    
    model = ATUNet(in_ch=6, out_ch=6, split_last=True)
    # model = ATUNetS(in_ch=6, out_ch2=6)
    train_module = TrainModule(model)
    
    dataset = LBSDataset(args.data_root, mode="compressed")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    trainer.fit(train_module, train_dataloaders=dataloader)
    
if __name__ == "__main__":
    main()