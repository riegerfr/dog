import os
from datetime import datetime
from datetime import timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset


def get_data():
    seed_everything(0)
    toy_data = []
    means = [-1.5, -0.75, 0, 0.75, 1.5]
    for x in means:
        for y in means:
            toy_data.append(0.01 * torch.randn(2, 2000) + torch.tensor([[x], [y]]))
    toy_data = torch.cat(toy_data, 1).T
    return toy_data


class GradientAdversarial(pl.LightningModule):
    def __init__(self, plot_data, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential((nn.Linear(2, 128)), nn.LeakyReLU(),
                                   (nn.Linear(128, 128)), nn.LeakyReLU(),
                                   (nn.Linear(128, 1))) 
        self.fixed_noise = None

        self.max_steps = 100
        self.loss_history = []
        self.toy_data = plot_data

    @torch.set_grad_enabled(True)
    def generate(self, max_steps, noise):

        sample = torch.nn.Parameter(noise.detach().clone())
        optimizer = AdamW([sample], lr=1e1000)
        scheduler = OneCycleLR(optimizer, max_lr=1e-1, total_steps=max_steps)
        model = self.model
        model.requires_grad_(False)
        grad_norms = []
        losses = []
        n_steps = max_steps
        samples = []
        for i in range(n_steps):
            optimizer.zero_grad()
            score = model(sample)
            loss = -score.mean()
            loss.backward()
            grad_norms.append(sample.grad.norm().detach())
            losses.append(-score.detach())
            samples.append(sample.detach().clone())
            optimizer.step()
            scheduler.step()
        samples.append(sample.detach().clone())

        print(f"step {i} loss {loss}")
        self.model.requires_grad_(True)

        return {"sample": sample.detach(), "losses": losses, "grad_norms": grad_norms, "used_steps": i,
                "samples": samples}

    @torch.enable_grad()
    def shared_step(self, batch, tag):
        self.model.train()

        real = batch
        if self.fixed_noise is None:
            self.fixed_noise = torch.randn_like(real)
        generated = self.generate(max_steps=self.max_steps,
                                  noise=self.fixed_noise if tag == "val" else
                                  torch.randn_like(self.fixed_noise)
                                  )

        fake = generated["sample"]

        fake.requires_grad = True
        real.requires_grad = True

        score_real = self.model(real)
        score_fake = self.model(fake)
        loss_real = (-score_real).mean()
        loss_fake = (score_fake).mean()

        loss = loss_real + loss_fake
        if tag == "train":
            self.loss_history.append(loss.item())
        self.log(name=f"{tag}_disc_loss", value=loss.detach())

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
        self.log(name=f"{tag}_gen_loss_final", value=generated["losses"][-1].mean())
        self.log(name=f"{tag}_gen_grad_norm_final", value=generated["grad_norms"][-1])
        self.log(name=f"{tag}_gen_grad_norm_0", value=generated["grad_norms"][0])
        self.log(name=f"{tag}_gen_grad_norm_1", value=generated["grad_norms"][1])
        self.log(name=f"{tag}_gen_grad_norm_0.5", value=generated["grad_norms"][int(len(generated["grad_norms"]) // 2)])
        self.log(name=f"{tag}_gen_grad_norm_-2", value=generated["grad_norms"][-2])
        self.log(name=f"{tag}_used_steps", value=generated["used_steps"])

        self.log(name=f"{tag}_gen_grad_loss_final", value=generated["losses"][-1].mean())
        self.log(name=f"{tag}_gen_grad_loss_0", value=generated["losses"][0].mean())
        self.log(name=f"{tag}_gen_grad_loss_1", value=generated["losses"][1].mean())
        mid = int(len(generated["losses"]) // 2)
        self.log(name=f"{tag}_gen_grad_loss_0.5", value=generated["losses"][mid].mean())
        self.log(name=f"{tag}_gen_grad_loss_-2", value=generated["losses"][-2].mean())
        self.log(name=f"{tag}_gen_grad_loss_last_speed",
                 value=(generated["losses"][-1].mean() - generated["losses"][-2].mean()))
        self.log(name=f"{tag}_gen_grad_loss_0.5_speed",
                 value=(generated["losses"][mid].mean() - generated["losses"][mid - 2].mean()))

        self.log(name=f"{tag}_used_steps", value=generated["used_steps"])

        if tag == "val":
            self.logger.experiment.add_image(f'f{tag}_real',
                                             torchvision.utils.make_grid(real[:8].detach(), normalize=True),
                                             global_step=self.global_step)
            self.logger.experiment.add_image(f'f{tag}_fake',
                                             torchvision.utils.make_grid(fake[:8].detach(), normalize=True),
                                             global_step=self.global_step)
            fake_from_real = self.generate(max_steps=self.max_steps,
                                           noise=real.detach()
                                           )["sample"]
            self.logger.experiment.add_image(f'f{tag}_fake_from_real',
                                             torchvision.utils.make_grid(fake_from_real[:8].detach(),
                                                                         normalize=True),
                                             global_step=self.global_step)

            coords = torch.linspace(-2, 2, 1000, device=self.device)
            X, Y = torch.meshgrid([coords, coords])
            inps = torch.stack((X, Y)).reshape(2, -1)
            d_scores = self.model(inps.T)

            d_scores_reshaped = d_scores.reshape(1000, 1000)
            d_scores_reshaped = d_scores_reshaped / d_scores_reshaped.std()
            symmetry_score = (d_scores_reshaped - d_scores_reshaped.flip(0, 1)).square().mean()
            self.log(name=f"{tag}_symmetry_score", value=symmetry_score)

            all_generated = []
            for i in range(10):
                generated = self.generate(max_steps=self.max_steps,
                                          noise=self.fixed_noise if i == 0 else torch.randn_like(self.fixed_noise)
                                          )
                all_generated.append(generated)
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 10)

            markers = []
            ax.contourf(X.detach().cpu(), Y.detach().cpu(), d_scores.detach().cpu().reshape(1000, 1000), 1000,
                        cmap=plt.cm.plasma,
                        )
            markers.append(
                ax.plot(self.toy_data[:, 0], self.toy_data[:, 1], marker="x", color="black", alpha=0.5,
                        linestyle='none')[0])

            lines = []
            for i in range(10):
                if i == 0:
                    paths = torch.stack(all_generated[i]['samples']).detach().cpu()[:]
                    for j in range(8):  # for 8 samples of the first generated batch plot paths
                        line, = ax.plot(paths[:, j, 0], paths[:, j, 1], '-', linewidth=4., alpha=0.8,
                                        # path_effects=[pe.Stroke(linewidth=5, foreground='g'), pe.Normal()]
                                        )
                        lines.append(line)
                fake = all_generated[i]["sample"].detach().cpu()
                for j in range(fake.shape[0]):
                    marks, = ax.plot(fake[j, 0], fake[j, 1], marker="x", color="white", linestyle='none'
                                     )
                    markers.append(marks)
            plt.axis('off')

            fig.tight_layout(pad=0)
            fig.canvas.draw()
            plt.savefig(f'val_{self.current_epoch}.png')

            for mark in markers:
                mark.set_markersize(10)
            for line in lines:
                line.set_linewidth(15)

            ax.set_xlim([-0.05, 0.05])
            ax.set_ylim([-0.05, 0.05])
            fig.tight_layout(pad=0)

            plt.savefig(f'val_zoom_center_{self.current_epoch}.png')

            ax.set_xlim([1.45, 1.55])
            ax.set_ylim([0.70, 0.80])
            fig.tight_layout(pad=0)

            plt.savefig(f'val_zoom_right_second_top_{self.current_epoch}.png')

            plt.close()

            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.size": 50,
            })

            fig, ax = plt.subplots()
            fig.set_size_inches(10, 5)

            losses = torch.stack(all_generated[0]['losses']).detach().cpu()[:]
            for j in range(8):
                ax.plot(losses[:, j], '-', linewidth=4., alpha=0.5)
            ax.set_xlabel("$T$")
            ax.set_ylabel("$L_G$")
            fig.tight_layout(pad=0.1)
            fig.canvas.draw()
            plt.savefig(f'gen_loss_{self.current_epoch}.svg')

            plt.close()

            fig, ax = plt.subplots()
            fig.set_size_inches(10, 5)

            ax.plot(self.loss_history, '-', linewidth=4., alpha=1)
            ax.set_xlabel("Train steps")
            ax.set_ylabel("$L_D$")
            fig.tight_layout(pad=0.1)
            fig.canvas.draw()
            plt.savefig(f'train_loss_{self.current_epoch}.svg')

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = np.transpose(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)), (2, 0, 1))

            self.logger.experiment.add_image(f'f{tag}_path_plot',
                                             data,
                                             global_step=self.global_step)
            plt.close()

        return loss

    def training_step(self, batch, *args, **kwargs):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, *args, **kwargs):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=1e-5,
                     betas=(0., 0.9)
                     )


class ToyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


def main():
    data = get_data()
    seed_everything(7, workers=True)
    n_gpus = torch.cuda.device_count()
    print(f"#gpus available: {n_gpus}")
    assert n_gpus >= 1

    exp_name = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    save_path = ("./gradient_ascent/")
    save_dir = os.path.join(save_path, exp_name + "/")
    os.makedirs(save_dir, exist_ok=True)
    print(f"save dir: {save_dir}")
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=exp_name)
    tb_logger.experiment.add_text("save dir", save_dir)
    n_nodes = 1
    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=50,
        accelerator="gpu",
        auto_select_gpus=True,
        profiler="advanced",
        log_every_n_steps=50,
        limit_val_batches=1,
        gradient_clip_val=5,
        num_nodes=n_nodes,
        check_val_every_n_epoch=10,
    )
    model = GradientAdversarial(plot_data=data
                                )

    train_data = ToyDataset(data)
    train_dataloader = DataLoader(train_data, batch_size=128,
                                  num_workers=0, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(train_data, batch_size=128,
                                num_workers=0,
                                drop_last=False, shuffle=False)
    trainer.fit(model=model,
                train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
                )

    tb_logger.save()
    print("done")


if __name__ == "__main__":
    main()

