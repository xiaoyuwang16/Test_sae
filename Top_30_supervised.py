import torch
import esm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from torch import nn
from matplotlib.patches import Patch


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, k=None):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.k = k

    def forward_encoder(self, x):
        pre_activations = self.encoder(x)
        if self.k is not None:
            topk_values, topk_indices = torch.topk(pre_activations, k=self.k, dim=-1)
            mask = torch.zeros_like(pre_activations).scatter_(-1, topk_indices, 1.0)
            encoded = pre_activations * mask
            encoded = torch.relu(encoded)
        else:
            encoded = torch.relu(pre_activations)
        return encoded

def get_esm_embeddings(model, alphabet, sequence, layer_idx, device):
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer_idx], return_contacts=False)
    return results["representations"][layer_idx], batch_tokens

def plot_aggregate_interface_response(
    target_seq,
    binder_seq_1,
    binder_seq_2,
    pdb_offset=0,
    sae_ckpt_path="",
    esm_model_name="esm2_t12_35M_UR50D",
    esm_layer=12,
    hidden_dim=15360,
    k=32,
    top_k=30,  # Picking the number of top features you want”
    output_file="aggregate_interface_response.png",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"--- Calculating bingding interface controlling features (Top {top_k} Features) ---")

    # PDB no. of interface AA
    interface_residues_pdb = [10, 12, 15, 83, 86, 87, 89, 121, 17, 19, 20, 21, 23, 24, 49, 114, 115, 116, 117, 118, 11, 13, 16, 119, 120]
    # Align with sequence index
    interface_indices_0 = [r - 1 - pdb_offset for r in interface_residues_pdb]
    valid_interface_indices = [i for i in interface_indices_0 if 0 <= i < len(target_seq)]
    valid_interface_indices = sorted(list(set(valid_interface_indices))) 

    print(f"Valid Interface Indices (0-indexed): {valid_interface_indices}")

    print("Loading Models...")
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
    esm_model = esm_model.to(device).eval()

    sae = SparseAutoencoder(input_dim=esm_model.embed_dim, hidden_dim=hidden_dim, k=k).to(device)
    
    if os.path.exists(sae_ckpt_path):
        checkpoint = torch.load(sae_ckpt_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        sae.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {sae_ckpt_path}, using random weights.")
    
    sae.eval()

    linker = "G" * 25

    # Single State
    emb_single_full, _ = get_esm_embeddings(esm_model, alphabet, target_seq, esm_layer, device)
    emb_single = emb_single_full[0, 1 : len(target_seq) + 1]
    with torch.no_grad():
        acts_single = sae.forward_encoder(emb_single) 

    # Complex State
    seq_complex = target_seq + linker + binder_seq_1 + linker + binder_seq_2
    emb_complex_full, _ = get_esm_embeddings(esm_model, alphabet, seq_complex, esm_layer, device)
    emb_complex_sliced = emb_complex_full[0, 1 : len(target_seq) + 1]
    with torch.no_grad():
        acts_complex = sae.forward_encoder(emb_complex_sliced) 

    print("Calculating Differential Activation...")

    delta_matrix = (acts_complex - acts_single)
    abs_delta_matrix = delta_matrix.abs()

    if len(valid_interface_indices) > 0:
        delta_at_interface = abs_delta_matrix[valid_interface_indices, :]
        
        feature_scores = delta_at_interface.sum(dim=0) 

        top_scores, top_indices = torch.topk(feature_scores, k=top_k)
        top_indices = top_indices.cpu()
        print(f"Top {top_k} Interface-Sensitive Feature Indices: {top_indices.numpy()}")
    else:
        print("Error: No valid interface indices found.")
        return

    print("Aggregating Signal...")

    selected_features_delta = abs_delta_matrix[:, top_indices]

    aggregate_signal = selected_features_delta.sum(dim=1).cpu().numpy()

    print("Plotting...")
    sns.set_style("whitegrid") 
    plt.figure(figsize=(12, 5)) 

    x_axis = np.arange(len(target_seq))

    plt.plot(x_axis, aggregate_signal, color='#1f77b4', linewidth=2.5, label='Aggregated Feature Response')
    plt.fill_between(x_axis, aggregate_signal, color='#1f77b4', alpha=0.15)

    for idx in valid_interface_indices:
        plt.axvspan(idx - 0.5, idx + 0.5, color='#f1c40f', alpha=0.4, lw=0)

    plt.xlim(0, len(target_seq))
    plt.xlabel("Residue Position", fontsize=12, fontweight='bold')
    plt.ylabel("Sum of Feature Shifts (Top K)", fontsize=12, fontweight='bold')
    plt.title(f"Collective Response of Top {top_k} Interface-Sensitive Features", fontsize=14, fontweight='bold', pad=15)

    legend_elements = [
        plt.Line2D([0], [0], color='#1f77b4', lw=2.5, label=f'SAE Response (Top {top_k} Features)'),
        Patch(facecolor='#f1c40f', edgecolor=None, alpha=0.4, label='Ground Truth Interface')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)

    tick_step = 10
    ticks = np.arange(0, len(target_seq), tick_step)
    tick_labels = [f"{i+1+pdb_offset}\n{target_seq[i]}" for i in ticks]
    plt.xticks(ticks, tick_labels, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"Plot saved to: {output_file}")

if __name__ == "__main__":
    SEQ_A = "PSTQPWEHVNAIQEARRLLNLSRDTAAEMNETVEVISEMFDLQEPTCLQTRLELYKQGLRGSLTKLKGPLTMMASHYKQHCPPTPETSCATQIITFESFKENLKDFLLVIPFDCW"
    SEQ_H = "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMNWVRQAPGKGLEWVSGISYSGSETYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGFGTDFWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPK"
    SEQ_L = "DIELTQPPSVSVAPGQTARISCSGDSIGKKYAYWYQQKPGQAPVLVIYKKRPSGIPERFSGSNSGNTATLTISGTQAEDEADYYCSSWDSTGLVFGGGTKLTVLGQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPT"

    CKPT_PATH = "/content/drive/MyDrive/Rep_SAEs_PLMs/Topk_weights/Regular_esmLayer12_MeanPooled_sae_20260223_215130_esmt12_k32_hd15360_lr0.0004_ep1.ckpt"

    plot_aggregate_interface_response(
        target_seq=SEQ_A,
        binder_seq_1=SEQ_H,
        binder_seq_2=SEQ_L,
        pdb_offset=7, 
        sae_ckpt_path=CKPT_PATH,
        top_k=30, # Picking the number of top features you want”
        output_file="aggregate_interface_plot.png"
    )