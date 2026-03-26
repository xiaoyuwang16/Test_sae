import os
import random
import numpy as np
import torch
from torch import nn
import esm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import umap

# ==========================================
# 1. Model Definitions
# ==========================================
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

# ==========================================
# 2. Global Discovery & Aggregation Plot
# ==========================================
def plot_global_delta_response(
    target_seq,
    binder_seq_1,
    binder_seq_2,
    pdb_offset=0,
    sae_ckpt_path="",
    esm_model_name="esm2_t12_35M_UR50D",
    esm_layer=12,
    hidden_dim=15360,
    k=32,
    top_k=1,
    output_file="global_delta_response.png",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("--- Calculating Global Binding Response (Unsupervised Discovery) ---")

    interface_residues_pdb = [10, 12, 15, 83, 86, 87, 89, 121, 17, 19, 20, 21, 23, 24, 49, 114, 115, 116, 117, 118, 11, 13, 16, 119, 120]
    interface_indices_0 = [r - 1 - pdb_offset for r in interface_residues_pdb]
    valid_interface_indices = [i for i in interface_indices_0 if 0 <= i < len(target_seq)]
    valid_interface_indices = sorted(list(set(valid_interface_indices)))

    print("Loading Models...")
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
    esm_model = esm_model.to(device).eval()

    sae = SparseAutoencoder(input_dim=esm_model.embed_dim, hidden_dim=hidden_dim, k=k).to(device)

    if not os.path.exists(sae_ckpt_path):
        raise FileNotFoundError(f"Fatal Error: SAE checkpoint not found at {sae_ckpt_path}")

    checkpoint = torch.load(sae_ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {key.replace('model.', ''): v for key, v in state_dict.items()}
    sae.load_state_dict(state_dict, strict=False)
    sae.eval()

    linker = "G" * 25

    emb_single_full, _ = get_esm_embeddings(esm_model, alphabet, target_seq, esm_layer, device)
    emb_single = emb_single_full[0, 1 : len(target_seq) + 1]
    with torch.no_grad():
        acts_single = sae.forward_encoder(emb_single)

    seq_complex = target_seq + linker + binder_seq_1 + linker + binder_seq_2
    emb_complex_full, _ = get_esm_embeddings(esm_model, alphabet, seq_complex, esm_layer, device)
    emb_complex_sliced = emb_complex_full[0, 1 : len(target_seq) + 1]
    with torch.no_grad():
        acts_complex = sae.forward_encoder(emb_complex_sliced)

    print("Calculating Global Differential Activation...")
    delta_matrix = (acts_complex - acts_single).abs().cpu()

    feature_scores = delta_matrix.sum(dim=0) 
    top_scores, top_indices = torch.topk(feature_scores, k=top_k)

    print(f"Top {top_k} Features with Highest Global Change: {top_indices.numpy()}")
    print(f"Max Score: {top_scores[0]:.4f}, Min Score (in top k): {top_scores[-1]:.4f}")

    print("Aggregating Signal...")
    selected_features_delta = delta_matrix[:, top_indices]
    aggregate_signal = selected_features_delta.sum(dim=1).numpy() 

    print("Plotting...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 6))

    x_axis = np.arange(len(target_seq))
    plt.plot(x_axis, aggregate_signal, color='#e74c3c', linewidth=2.5, label='Global Delta Response (Top Features)')
    plt.fill_between(x_axis, aggregate_signal, color='#e74c3c', alpha=0.15)

    for idx in valid_interface_indices:
        plt.axvspan(idx - 0.5, idx + 0.5, color='#f1c40f', alpha=0.4, lw=0)

    plt.xlim(0, len(target_seq))
    plt.xlabel("Residue Position", fontsize=12, fontweight='bold')
    plt.ylabel("Aggregate Feature Shift (L1 Norm)", fontsize=12, fontweight='bold')
    plt.title(f"Unsupervised Interface Discovery: Top {top_k} Most Changed Features", fontsize=15, fontweight='bold', pad=15)

    legend_elements = [
        plt.Line2D([0], [0], color='#e74c3c', lw=2.5, label=f'Top {top_k} Changed Features (Model Discovery)'),
        Patch(facecolor='#f1c40f', edgecolor=None, alpha=0.4, label='Ground Truth Interface (Validation)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)

    tick_step = 10
    ticks = np.arange(0, len(target_seq), tick_step)
    tick_labels = [f"{i+1+pdb_offset}\n{target_seq[i]}" for i in ticks if i < len(target_seq)]
    plt.xticks(ticks[:len(tick_labels)], tick_labels, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"Plot saved to: {output_file}")

    return aggregate_signal, top_indices.numpy().tolist()

# ==========================================
# 3. UMAP Feature Visualization
# ==========================================
def print_and_plot_features(ckpt_path, top_feature_indices, d_sae=15360):
    print(f"\nLoading SAE checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    decoder_weight_key = None
    for k, v in state_dict.items():
        if len(v.shape) == 2 and d_sae in v.shape:
            decoder_weight_key = k

    if decoder_weight_key is None:
        print(f"\nError: Could not find a 2D weight matrix containing dimension {d_sae}!")
        return

    print(f"Successfully located decoder weights: {decoder_weight_key}")
    W_dec = state_dict[decoder_weight_key]

    is_row_major = W_dec.shape[0] == d_sae
    if not is_row_major:
        W_dec = W_dec.t() 

    print("\nExtracting Top 10 feature vectors...")
    top_10_features = top_feature_indices[:10]

    for feat_idx in top_10_features:
        feature_vector = W_dec[feat_idx, :]
        print(f"--- Feature ID: {feat_idx} | L2 Norm: {torch.norm(feature_vector, p=2).item():.4f} ---")

    print(" Running UMAP dimensionality reduction on 15360 features (this may take 1-2 minutes)...")
    W_dec_np = W_dec.detach().cpu().numpy()

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(W_dec_np)

    print("UMAP reduction complete. Plotting...")

    plt.figure(figsize=(12, 10))

    plt.scatter(embedding[:, 0], embedding[:, 1],
                c='lightgray', s=5, alpha=0.5, label='All SAE Features (15360)')

    top_10_embeddings = embedding[top_10_features]

    plt.scatter(top_10_embeddings[:, 0], top_10_embeddings[:, 1],
                c='red', s=200, marker='*', edgecolor='darkred', zorder=5, label='Top 10 Binding Features')

    for i, feat_idx in enumerate(top_10_features):
        plt.annotate(str(feat_idx),
                     (top_10_embeddings[i, 0], top_10_embeddings[i, 1]),
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=11, color='black', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    plt.title("UMAP Projection of SAE Decoder Features (Highlighting Top Binding Features)", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("umap_features_projection.png", dpi=300)
    plt.show()

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    SEQ_A = "PSTQPWEHVNAIQEARRLLNLSRDTAAEMNETVEVISEMFDLQEPTCLQTRLELYKQGLRGSLTKLKGPLTMMASHYKQHCPPTPETSCATQIITFESFKENLKDFLLVIPFDCW"
    SEQ_H = "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMNWVRQAPGKGLEWVSGISYSGSETYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGFGTDFWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPK"
    SEQ_L = "DIELTQPPSVSVAPGQTARISCSGDSIGKKYAYWYQQKPGQAPVLVIYKKRPSGIPERFSGSNSGNTATLTISGTQAEDEADYYCSSWDSTGLVFGGGTKLTVLGQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPT"

    # NOTE: Update this path if you are running locally instead of Google Colab
    CKPT_PATH = "/content/drive/MyDrive/Rep_SAEs_PLMs/Topk_weights/Regular_esmLayer12_MeanPooled_sae_20260223_215130_esmt12_k32_hd15360_lr0.0004_ep1.ckpt"

    # Step 1: Run global delta response and get the top features automatically
    aggregate_signal, top_features_list = plot_global_delta_response(
        target_seq=SEQ_A,
        binder_seq_1=SEQ_H,
        binder_seq_2=SEQ_L,
        pdb_offset=7,
        sae_ckpt_path=CKPT_PATH,
        top_k=30, 
        output_file="global_delta_discovery.png"
    )

    # Step 2: Pass the dynamically generated top features to the UMAP plotting function
    print_and_plot_features(
        ckpt_path=CKPT_PATH, 
        top_feature_indices=top_features_list
    )