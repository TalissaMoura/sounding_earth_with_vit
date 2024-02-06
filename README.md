# Self-supervised learning for Audiovisual Aerial Scene Classification


This repo reunites code for the project "Using Self-Supervised Learning to classify aerial scenes audiovisuals with remote sensing data". We use the vision transformers paradigm
to generate embeddings that approximate audio and image from SoundingEarth dataset. Then,
we check its ability to classify audiovisual scenes from the ADVANCE data. Most of the code
was based from this other [repo](https://github.com/khdlr/SoundingEarth).

## How to run this code?

1. First, you can run this locally and on colab notebooks (check the notebooks folder). 
2. Download the datasets from [SoundingEarth](https://zenodo.org/records/5600379) and ADVANCE (for images use this [link](https://zenodo.org/records/3828124) and spectrograms this [link](https://github.com/khdlr/SoundingEarth/releases/tag/spectrograms)).
3. Create an account on [wandb](https://wandb.ai/). We're using this platform to log our experiments.
4. If you run this locally:
   - Look at the config.py file and adjust the DataRoot, also, check the dataloaders.py to ensure that the patchs to data folders are correct.
   - [Configure wandb into your local setup](https://docs.wandb.ai/quickstart).
   - Use the `environment_v4.yml` to configure your local virtual environment to download 
   the necessary libraries.
   - To run the train script use: `python train.py` and to run the classifier use: `python advance.py`
5. If you run this on colab:
   - Load the datasets into your google drive or other place that you can access on colab.
   - The train notebooks is `01_embeddings_soundingearth_with_vit_base.ipynb` and the advance classifier is `02_advance_classifier.ipynb` and the EDA is on `03_evaluate_results_for_vit_models_in_advance.ipynb`.
6. In lib is where you find the models (`lib/models`) and loss_functions (`lib/loss_functions.py`).

## Releases

You can download the models weights from the [Releases](https://github.com/TalissaMoura/sounding_earth_with_vit/releases) tab.
