import os
from datetime import datetime
from datetime import timedelta
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch import autograd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance

from ffhqdataset import get_ffhq, unnormalize
from styleNAT import Discriminator


def CR_DiffAug(x, flip=True, translation=True, color=True, cutout=True):
    if flip:
        x = random_flip(x, 0.5)
    if translation:
        x = rand_translation(x, 1 / 8)
    if color:
        aug_list = [rand_brightness, rand_saturation, rand_contrast]
        for func in aug_list:
            x = func(x)
    if cutout:
        x = rand_cutout(x)
    if flip or translation:
        x = x.contiguous()
    return x


def random_flip(x, p):
    x_out = x.clone()
    n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    flip_prob = torch.FloatTensor(n, 1).uniform_(0.0, 1.0)
    flip_mask = flip_prob < p
    flip_mask = flip_mask.type(torch.bool).view(n, 1, 1, 1).repeat(1, c, h, w).to(x.device)
    x_out[flip_mask] = torch.flip(x[flip_mask].view(-1, c, h, w), [3]).view(-1)
    return x_out


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def d_logistic_loss(real_pred, fake_pred):
    assert type(real_pred) == type(fake_pred), "real_pred must be the same type as fake_pred"
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img, target=0):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = (grad_real - target).pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


class GradientAdversarial(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = Discriminator(scale=0.25,
                                   size=256
                                   )

        self.fid_metric = FrechetInceptionDistance()

        self.fixed_noise = None
        self.last_fake = None

        self.bcr = True
        self.r1 = True
        self.max_steps = 100

    @torch.set_grad_enabled(True)
    def generate(self, max_steps, noise):

        sample = torch.nn.Parameter(noise.detach().clone())
        optimizer = AdamW([sample], lr=1e10000)
        scheduler = OneCycleLR(optimizer, max_lr=1e0, total_steps=max_steps)

        model = self.model
        model.requires_grad_(False)
        grad_norms = []
        losses = []
        n_steps = max_steps

        samples = []

        for i in range(n_steps):
            samples.append(sample.detach().clone())
            optimizer.zero_grad()
            score = model(sample)
            loss = F.softplus(-score).mean()
            loss.backward()
            grad_norms.append(sample.grad.view(sample.shape[0], -1).norm(dim=1).mean().detach()
                              )
            losses.append(loss.detach())
            optimizer.step()

            scheduler.step()

        print(f"step {i} loss {loss}")
        model.requires_grad_(True)

        return {"sample": sample.detach(), "losses": losses, "grad_norms": grad_norms, "used_steps": i,
                "samples": samples}

    @torch.enable_grad()
    def shared_step(self, batch, tag):
        self.model.train()

        real = batch
        if self.fixed_noise is None:
            self.fixed_noise = torch.randn_like(real)
        if self.last_fake is None:
            self.last_fake = torch.randn_like(real)

        generated = self.generate(max_steps=self.max_steps,
                                  noise=self.fixed_noise if tag == "val" else
                                  torch.randn_like(self.fixed_noise)
                                  )
        fake = generated["sample"]
        fake.requires_grad = True
        real.requires_grad = True

        score_real = self.model(real)
        score_fake = self.model(fake)
        loss_real = F.softplus(-score_real).mean()
        loss_fake = F.softplus(score_fake).mean()
        loss = loss_real + loss_fake

        self.log(name=f"{tag}_disc_loss", value=loss.detach())

        if self.bcr:
            real_img_cr_aug = CR_DiffAug(real.detach())
            fake_img_cr_aug = CR_DiffAug(fake.detach())
            fake_pred_aug = self.model(fake_img_cr_aug)
            real_pred_aug = self.model(real_img_cr_aug)

            bcr_weight = 10
            bcr_loss = bcr_weight * (fake_pred_aug - score_fake).square().mean() + \
                       bcr_weight * (real_pred_aug - score_real).square().mean()
            self.log(name=f"{tag}_bcr_loss", value=bcr_loss)

            loss = loss + bcr_loss
        if self.r1:
            r1_loss = d_r1_loss(score_real, real)
            self.log(name=f"{tag}_r1_loss", value=r1_loss)

            loss = loss + (10 / 2 * r1_loss)

        std_real = torch.sqrt(score_real.var(dim=0) + 0.0001)
        std_fake = torch.sqrt(score_fake.var(dim=0) + 0.0001)
        std_comb = torch.sqrt(torch.cat([score_real, score_fake]).flatten().var() + 0.0001)

        self.log(name=f"{tag}_std_real", value=std_real)
        self.log(name=f"{tag}_std_comb", value=std_comb)
        self.log(name=f"{tag}_std_fake", value=std_fake)

        self.log(name=f"{tag}_loss", value=loss)
        self.log(name=f"{tag}_loss_real", value=loss_real)
        self.log(name=f"{tag}_loss_fake", value=loss_fake)

        self.log(name=f"{tag}_real_mean", value=real.mean())
        self.log(name=f"{tag}_fake_mean", value=fake.mean())
        self.log(name=f"{tag}_real_std_batch", value=real.std(0).mean())
        self.log(name=f"{tag}_fake_std_batch", value=fake.std(0).mean())
        self.log(name=f"{tag}_gen_loss_final", value=generated["losses"][-1])
        self.log(name=f"{tag}_gen_grad_norm_final", value=generated["grad_norms"][-1])
        self.log(name=f"{tag}_gen_grad_norm_0", value=generated["grad_norms"][0])
        self.log(name=f"{tag}_gen_grad_norm_1", value=generated["grad_norms"][1])
        self.log(name=f"{tag}_gen_grad_norm_0.5", value=generated["grad_norms"][int(len(generated["grad_norms"]) // 2)])
        self.log(name=f"{tag}_gen_grad_norm_-2", value=generated["grad_norms"][-2])
        self.log(name=f"{tag}_used_steps", value=generated["used_steps"])

        self.log(name=f"{tag}_gen_grad_loss_final", value=generated["losses"][-1])
        self.log(name=f"{tag}_gen_grad_loss_0", value=generated["losses"][0])
        self.log(name=f"{tag}_gen_grad_loss_1", value=generated["losses"][1])
        mid = int(len(generated["losses"]) // 2)
        self.log(name=f"{tag}_gen_grad_loss_0.5", value=generated["losses"][mid])
        self.log(name=f"{tag}_gen_grad_loss_-2", value=generated["losses"][-2])
        self.log(name=f"{tag}_gen_grad_loss_last_speed", value=(generated["losses"][-1] - generated["losses"][-2]))
        self.log(name=f"{tag}_gen_grad_loss_0.5_speed", value=(generated["losses"][mid] - generated["losses"][mid - 2]))

        self.log(name=f"{tag}_used_steps", value=generated["used_steps"])

        if tag == "val":
            with torch.no_grad():
                self.fid_metric.update(unnormalize(real * 255).to(torch.uint8), real=True)
                self.fid_metric.update(unnormalize(fake * 255).to(torch.uint8), real=False)
            self.logger.experiment.add_image(f'f{tag}_real',
                                             torchvision.utils.make_grid(real[:8].detach(), normalize=True),
                                             global_step=self.global_step)
            self.logger.experiment.add_image(f'f{tag}_fake',
                                             torchvision.utils.make_grid(fake[:8].detach(), normalize=True),
                                             global_step=self.global_step)
            for i, sample in list(enumerate(generated["samples"]))[::5]:
                self.logger.experiment.add_image(f'f{tag}_fake_{i}',
                                                 torchvision.utils.make_grid(sample[:8].detach(), normalize=True),
                                                 global_step=self.global_step)

            fake_from_real = self.generate(max_steps=self.max_steps,
                                           noise=real.detach()
                                           )["sample"]
            self.logger.experiment.add_image(f'f{tag}_fake_from_real',
                                             torchvision.utils.make_grid(fake_from_real[:8].detach(),
                                                                         normalize=True),
                                             global_step=self.global_step)
            fake_from_noisy_real = self.generate(max_steps=self.max_steps,
                                                 noise=real.detach() + self.fixed_noise * 0.5
                                                 )["sample"]
            self.logger.experiment.add_image(f'f{tag}_fake_from_noisy_real',
                                             torchvision.utils.make_grid(fake_from_noisy_real[:8].detach(),
                                                                         normalize=True),
                                             global_step=self.global_step)
            fake_from_overlayed_real = self.generate(max_steps=self.max_steps,
                                                     noise=0.5 * (real + real.roll(1, 0))
                                                     )["sample"]
            self.logger.experiment.add_image(f'f{tag}_fake_from_overlayed_real',
                                             torchvision.utils.make_grid(fake_from_overlayed_real[:8].detach(),
                                                                         normalize=True),
                                             global_step=self.global_step)
            self.logger.experiment.add_image(f'f{tag}_fake_from_last_fake',
                                             torchvision.utils.make_grid(self.last_fake[:8].detach(),
                                                                         normalize=True),
                                             global_step=self.global_step)
        else:
            self.last_fake = fake.detach()

        return loss

    def on_validation_epoch_start(self):
        self.fid_metric.reset()

    def on_validation_epoch_end(self):
        try:
            fid = self.fid_metric.compute()
            self.log(
                name="validation_FID",
                value=fid,
                sync_dist=True,
            )
        except Exception as e:
            print(f"FID exception: {e}")

    def training_step(self, batch, *args, **kwargs):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, *args, **kwargs):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=2e-4,
                          betas=(0., 0.9)
                          )
        return optimizer


def main():
    seed_everything(7, workers=True)
    n_gpus = torch.cuda.device_count()
    print(f"#gpus available: {n_gpus}")
    assert n_gpus >= 1

    exp_name = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    save_path = "./logs/"
    save_dir = os.path.join(save_path, exp_name + "/")

    os.makedirs(save_dir, exist_ok=True)
    print(f"save dir: {save_dir}")
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=exp_name)
    tb_logger.experiment.add_text("save dir", save_dir)

    trainer = pl.Trainer(
        logger=tb_logger,
        max_time=timedelta(hours=6 * 24),
        max_epochs=100,
        accelerator="gpu",
        auto_select_gpus=True,
        profiler="advanced",
        log_every_n_steps=50,
        limit_val_batches=1,
        gradient_clip_val=5,
    )
    model = GradientAdversarial()

    train_data = get_ffhq(evaluation=False)
    val_data = get_ffhq(evaluation=True)
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  num_workers=5, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                num_workers=5,
                                drop_last=False, shuffle=False)
    trainer.fit(model=model,
                train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
                )

    tb_logger.save()
    print("done")


if __name__ == "__main__":
    main()
