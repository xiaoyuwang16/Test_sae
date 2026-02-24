
**Note:** This repository is a modification based on the original work [Rep_SAEs_PLMs](https://github.com/onkarsg10/Rep_SAEs_PLMs) by onkarsg10.


## Objectives

The primary goal of this project is to disentangle and isolate features specific to protein-protein binding sites and interfaces using Sparse Autoencoders (SAEs) trained on Protein Language Model (ESM) embeddings.


## Modifications from Original Repo 

1.  **WandB Configuration:** 
    - The `wandb` entity and project settings have been updated to log experiments to my personal workspace for independent tracking.
    - Added support for custom API keys via command line arguments.

## Usage

To run the training script, you need to set up your WandB API key first.

The following is necessary for all the SAEs and transcoders:

1. swissprot.tsv needs to be present in your current working directory, or the path to it must be provided in the swissprot_filtered_uniref_dataset.py script. This is simply the Swissprot dataset downloadable as a TSV file (https://www.uniprot.org/uniprotkb?query=*&facets=reviewed%3Atrue) in order to exclude its entries from SAE training. This is because the TSV later gets used for analysis. Remenber to include "Sequence" column. 

2. Topk_weights folder must exist in your current working directory. This folder is where the trained model weights will get saved.

3. The path to the uniref_file (uniref50.fasta.gz) needs to be provided as an argument. It can be downloaded from: https://www.uniprot.org/help/downloads as a .fasta.gz file.


#### Training Logs
Check out the training progress and metrics on Weights & Biases:
👉 **[Click here to view WandB Dashboard](https://wandb.ai/xiaoyuwang-bnu-university-of-danmark/protein-training-test?nw=nwuserxiaoyuwangbnu)** 

According to our needs, we'll only train SAE on AA level with this usage:

```python
import os

# Replace with YOUR OWN WandB API key
my_wandb_key = "YOUR_WANDB_API_KEY_HERE"

os.environ["WANDB_MODE"] = "online"

!python Flatten_instead_of_Pool/main_script.py \
    --uniref_file uniref50.fasta.gz \
    --batch_size 32 \
    --learning_rate 0.0004 \
    --k 32 \
    --hidden_dim 15360 \
    --epochs 1 \
    --encoder_decoder_init 1 \
    --max_seq_len 1024 \
    --esm_model esm2_t12_35M_UR50D \
    --inactive_threshold 50 \
    --val_check_interval 1000 \
    --limit_val_batches 10 \
    --max_samples 2000000 \
    --max_steps 20000 \
    --esm_layer 12 \
    --cuda_device 0 \
    --wandb_api_key {my_wandb_key}
```


## Dependencies

Main dependencies along with versions that were used for SAE/transcoder training and automated interpretation experiments:

- torch: 2.4.1+cu121
- pytorch-lightning: 2.2.0
- wandb: 0.17.7
- numpy: 1.24.4
- pandas: 2.0.1
- scipy: 1.10.1
- tqdm: 4.66.5
- h5py: 3.11.0
- biopython: 1.83
- anthropic: 0.39.0
- fair-esm: 2.0.1
- Python 3.8.18

Dependencies used for GO analysis:

- numpy: 1.24.1
- pandas: 2.0.3
- plotly: 5.24.1
- goatools: 1.4.12
- networkx: 3.0
- rich: 13.9.4
- scikit-learn: 1.3.2
- h5py: 3.11.0
- umap-learn: 0.5.7
- ipywidgets: 7.7.1
- notebook: 7.2.2
- statsmodels: 0.14.1
