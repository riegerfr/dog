# DOG on FFHQ

1. Setup: like https://github.com/SHI-Labs/StyleNAT + `pip install pytorch_lightning torch-fidelity`
2. Download data (https://github.com/NVlabs/ffhq-dataset)
3. Prepare data to ffhq.lmdb (e.g.
   using https://github.com/rosinality/stylegan2-pytorch/blob/bef283a1c24087da704d16c30abc8e36e63efa0e/prepare_data.py)
4. Run `python StyleGradientAdversarial.py`
5. Look at tensorboard