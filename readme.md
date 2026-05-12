# Evaluating GAN and VAE Approaches for Image Captioning
This repository contains files of project *Evaluating GAN and VAE Approaches for Image Captioning*. 
Here, we store all the training experiments and results.

# Files Navigation
├─ [_**code**_](code) contains models' architectures and additional scripts<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`utils.py`](code/utils.py) additional scripts<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [**architectures**](code/architectures) models' architectures and trainers<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`adapter.py`](code/architectures/adapter.py) general adapter architecture<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`gan.py`](code/architectures/gan.py) GAN discriminator architecture<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`vae.py`](code/architectures/vae.py) VAE encoder architecture<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [**lm-decoding**](code/architectures/lm-decoding) trainers that involve LLM decoder<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`baseline_trainer.py`](code/architectures/lm-decoding/baseline_trainer.py) baseline trainer<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`gan_trainer.py`](code/architectures/lm-decoding/gan_trainer.py) GAN trainer<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`vae_trainer.py`](code/architectures/lm-decoding/vae_trainer.py) VAE trainer<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`soft_prompt_trainer.py`](code/architectures/lm-decoding/soft_prompt_trainer.py) soft prompts trainer<br>
├─ [_**notebooks**_](notebooks) contains notebooks for inference and training<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [**training**](notebooks/training) training notebooks<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`baseline.ipynb`](notebooks/training/baseline.ipynb) baseline training<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`gan.ipynb`](notebooks/training/gan.ipynb) GAN training<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`vae.ipynb`](notebooks/training/vae.ipynb) VAE training<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [**evaluation**](notebooks/evaluation) inference notebooks<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`baseline.ipynb`](notebooks/evaluation/baseline.ipynb) baseline inference<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`gan.ipynb`](notebooks/evaluation/gan.ipynb) GAN inference<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`vae.ipynb`](notebooks/evaluation/vae.ipynb) VAE inference<br>
├─ [_**data**_](data) contains JSON files with data splits and inference results<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [`ds_captions.json`](data/ds_captions.json) data splits<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ [**results**](data/results) inference results<br>

# Training Logs
Training logs and artifacts (including trained adapters, encoders and discriminators) can be accessed via Weights&Biases:<br>
|  | S (6.9m) | M (18.3m) | L (36.7m) |
| ------ | ----- | ----- | ----- |
| *baseline* | [🔗](https://wandb.ai/pburub/gan-vae-image-captioning/runs/zgltu203) | [🔗](https://wandb.ai/pburub/gan-vae-image-captioning/runs/xsmk39od) | [🔗](https://wandb.ai/pburub/gan-vae-image-captioning/runs/oppsndpv) |
| *VAE* | [🔗](https://wandb.ai/pburub/gan-vae-image-captioning/runs/dj95dmc6) | [🔗](https://wandb.ai/pburub/gan-vae-image-captioning/runs/hrvn2ww8) | [🔗](https://wandb.ai/pburub/gan-vae-image-captioning/runs/lmh6eyr3) |
| *GAN* | [🔗](https://wandb.ai/pburub/gan-vae-image-captioning/runs/7pvtpczb) | [🔗](https://wandb.ai/pburub/gan-vae-image-captioning/runs/og188rrw) | [🔗](https://wandb.ai/pburub/gan-vae-image-captioning/runs/l5gpazuq) |


