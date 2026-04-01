import streamlit as st
import pandas as pd
import numpy as np
import os
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import json

def scatter_mean(src, index, dim_size, dim=0):
    """
    src: tensor of shape [num_edges, hidden_dim]
    index: tensor of shape [num_edges] (destination node indices)
    dim_size: number of destination nodes
    dim: ignored (kept for compatibility)
    Returns: tensor of shape [dim_size, hidden_dim] where each node gets the mean of messages from its neighbors
    """
    # Sum messages per destination
    sum_ = torch.zeros(dim_size, src.size(1), device=src.device)
    sum_ = sum_.scatter_add(0, index.unsqueeze(1).expand_as(src), src)
    # Count number of messages per destination
    count = torch.zeros(dim_size, device=src.device)
    count = count.scatter_add(0, index, torch.ones_like(index, dtype=src.dtype))
    # Avoid division by zero
    count = count.clamp(min=1)
    return sum_ / count.unsqueeze(1)

# -------------------------------------------------------------------
# Corrected NeoDTI model definition
# -------------------------------------------------------------------
class CorrectedNeoDTI(nn.Module):
    def __init__(self, num_nodes_dict, hidden_dim, relations, num_layers=2, use_layer_norm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relations = relations
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.rel_to_key = {rel: "__".join(rel) for rel in relations}
        self.emb = nn.ModuleDict()
        for ntype, num_nodes in num_nodes_dict.items():
            self.emb[ntype] = nn.Embedding(num_nodes, hidden_dim)
        self.W_rel = nn.ModuleDict()
        for rel in relations:
            key = self.rel_to_key[rel]
            self.W_rel[key] = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if use_layer_norm:
            self.layer_norm = nn.ModuleDict()
            for ntype in num_nodes_dict:
                self.layer_norm[ntype] = nn.LayerNorm(hidden_dim)
        self.W_final = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, edge_index_dict, drug_ids, cell_ids):
        h = {ntype: self.emb[ntype].weight for ntype in self.emb}
        for _ in range(self.num_layers):
            messages = {ntype: torch.zeros_like(h[ntype]) for ntype in h}
            for (src_type, rel, dst_type), edge_idx in edge_index_dict.items():
                src, dst = edge_idx
                key = self.rel_to_key[(src_type, rel, dst_type)]
                src_emb = h[src_type][src]
                msg = self.W_rel[key](src_emb)
                msg_mean = scatter_mean(msg, dst, dim=0, dim_size=h[dst_type].size(0))
                messages[dst_type] += msg_mean
            for ntype in h:
                updated = h[ntype] + self.dropout(messages[ntype])
                if self.use_layer_norm:
                    updated = self.layer_norm[ntype](updated)
                h[ntype] = F.relu(updated)
        drug_emb = h['drug'][drug_ids]
        cell_emb = h['cell'][cell_ids]
        combined = torch.cat([drug_emb, cell_emb], dim=1)
        logit = self.W_final(combined).squeeze()
        return torch.sigmoid(logit)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
import os
base_path = os.path.dirname(__file__)
st.set_page_config(page_title="Breast Cancer Drug Prioritization Tool", layout="wide")
# Add watermark as a styled badge
st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 15px;
        right: 15px;
        background-color: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(4px);
        padding: 6px 12px;
        border-radius: 30px;
        font-family: 'Segoe UI', sans-serif;
        font-size: 12px;
        font-weight: 500;
        color: #f0f0f0;
        z-index: 1000;
        pointer-events: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        letter-spacing: 0.5px;
    }
    .watermark emoji {
        margin-right: 6px;
        font-size: 14px;
    }
    </style>
    <div class="watermark">
        <span class="emoji">🧬</span> Bioinformatics Lab, NIT Warangal
    </div>
    """,
    unsafe_allow_html=True
)
st.title("🔬 Breast Cancer Subtype‑Specific Drug Prioritization Tool")
st.markdown("Powered by corrected NeoDTI model. For research use only.")

# -------------------------------------------------------------------
# Cached data loading
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    all_scores = pd.read_csv(os.path.join(base_path, "all_drug_scores.csv"))
    drug_gene = pd.read_csv(os.path.join(base_path, "drug_gene_map.csv"))
    target_map = pd.read_csv(os.path.join(base_path, "drug_target_class.csv"))
    gene_pathway = pd.read_csv(os.path.join(base_path, "gene_pathway_edges.csv"))
    return all_scores, drug_gene, target_map, gene_pathway

all_scores, drug_gene, target_map, gene_pathway = load_data()

# -------------------------------------------------------------------
# Cached model loading and node embeddings computation
# -------------------------------------------------------------------
@st.cache_resource
def load_model_and_embeddings():
    # Load graph data and node mappings
    data = torch.load(os.path.join(base_path, "hetero_data_corrected.pt"), map_location='cpu', weights_only=False)
    with open(os.path.join(base_path, "node_id_mappings.json"), 'r') as f:
        mappings = json.load(f)
    drug_to_id = mappings['drug_to_id']
    gene_to_id = mappings['gene_to_id']
    pathway_to_id = mappings['pathway_to_id']
    cell_to_id = mappings['cell_to_id']
    # Node counts
    num_nodes_dict = {
        'drug': len(drug_to_id),
        'gene': len(gene_to_id),
        'pathway': len(pathway_to_id),
        'cell': len(cell_to_id)
    }
    relations = list(data.edge_index_dict.keys())
    hidden_dim = 64
    num_layers = 2
    model = CorrectedNeoDTI(num_nodes_dict, hidden_dim, relations, num_layers, use_layer_norm=True)
    model.load_state_dict(torch.load(os.path.join(base_path, "neodti_final_model.pt"), map_location='cpu'))
    model.eval()
    edge_index_dict = data.edge_index_dict

    # Pre‑compute node embeddings after message passing
    x_dict = {
        'drug': torch.arange(num_nodes_dict['drug']),
        'gene': torch.arange(num_nodes_dict['gene']),
        'pathway': torch.arange(num_nodes_dict['pathway']),
        'cell': torch.arange(num_nodes_dict['cell'])
    }
    with torch.no_grad():
        h = {ntype: model.emb[ntype].weight for ntype in model.emb}
        for _ in range(model.num_layers):
            messages = {ntype: torch.zeros_like(h[ntype]) for ntype in h}
            for (src_type, rel, dst_type), edge_idx in edge_index_dict.items():
                src, dst = edge_idx
                key = model.rel_to_key[(src_type, rel, dst_type)]
                src_emb = h[src_type][src]
                msg = model.W_rel[key](src_emb)
                msg_mean = scatter_mean(msg, dst, dim=0, dim_size=h[dst_type].size(0))
                messages[dst_type] += msg_mean
            for ntype in h:
                updated = h[ntype] + model.dropout(messages[ntype])
                if model.use_layer_norm:
                    updated = model.layer_norm[ntype](updated)
                h[ntype] = F.relu(updated)
    gene_embeddings = h['gene']
    drug_embeddings = h['drug']
    # Create mapping from drug name to local ID
    drug_name_to_id = {name: idx for idx, name in enumerate(drug_to_id.keys())}
    return model, drug_embeddings, gene_embeddings, drug_name_to_id, gene_to_id, edge_index_dict

model, drug_embeddings, gene_embeddings, drug_name_to_id, gene_to_id, edge_index_dict = load_model_and_embeddings()

# -------------------------------------------------------------------
# Helper: predict for a new cell line
# -------------------------------------------------------------------
def predict_for_new_cell(overexpression_genes, amplification_genes, mutation_genes):
    """
    Input: sets of gene symbols for each edge type.
    Returns DataFrame with drug_name and raw predicted score (ranking is most important).
    """
    # Filter to genes that exist in our graph
    def filter_genes(genes):
        valid = []
        missing = []
        for g in genes:
            if g in gene_to_id:
                valid.append(gene_to_id[g])
            else:
                missing.append(g)
        return valid, missing

    overexpr_ids, missing_over = filter_genes(overexpression_genes)
    ampl_ids, missing_ampl = filter_genes(amplification_genes)
    mut_ids, missing_mut = filter_genes(mutation_genes)
    
    all_gene_ids = set(overexpr_ids + ampl_ids + mut_ids)
    
    if missing_over or missing_ampl or missing_mut:
        warn_msg = ""
        if missing_over:
            warn_msg += f"Overexpression missing: {missing_over[:5]}; "
        if missing_ampl:
            warn_msg += f"Amplification missing: {missing_ampl[:5]}; "
        if missing_mut:
            warn_msg += f"Mutation missing: {missing_mut[:5]}"
        st.warning(f"Some genes not found in graph. {warn_msg}")
    
    if not all_gene_ids:
        cell_embedding = torch.zeros(drug_embeddings.shape[1])
    else:
        max_id = len(gene_embeddings) - 1
        valid_ids = [i for i in all_gene_ids if 0 <= i <= max_id]
        if not valid_ids:
            cell_embedding = torch.zeros(drug_embeddings.shape[1])
        else:
            gene_embs = gene_embeddings[valid_ids]
            cell_embedding = gene_embs.mean(dim=0)
    
    with torch.no_grad():
        cell_emb_expanded = cell_embedding.unsqueeze(0).expand(len(drug_embeddings), -1)
        combined = torch.cat([drug_embeddings, cell_emb_expanded], dim=1)
        logits = model.W_final(combined).squeeze()
        scores = torch.sigmoid(logits).numpy()
    
    # Return raw scores (ranking is most important)
    drug_names = list(drug_name_to_id.keys())
    result_df = pd.DataFrame({'drug_name': drug_names, 'predicted_score': scores})
    result_df = result_df.sort_values('predicted_score', ascending=False)
    return result_df
# -------------------------------------------------------------------
# Sidebar mode selection
# -------------------------------------------------------------------
st.sidebar.header("Choose your mode")
with st.sidebar.expander("ℹ️ About this tool"):
    st.markdown("""
    **Model**: Corrected NeoDTI – a graph neural network trained on 126,030 drug‑cell line pairs.

    **Data**:
    - Drug sensitivity: GDSC, PRISM, CTRPv2
    - Multi‑omics: CCLE (expression, CNV, mutation)
    - Drug targets: ChEMBL
    - PPI: STRING
    - Pathways: Reactome

    **Interpretation**:
    - Scores are model outputs; **ranking order** is most important for prioritisation.
    - Explore by Subtype → top drugs for HER2, Luminal, TNBC.
    - Search by Gene → find drugs targeting a gene.
    - Compare Drugs → compare two drugs side by side.
    - Visualize Drug Network → see drug‑target‑pathway connections.
    - Predict for New Cell Line → upload molecular data for a custom cell line.

    **Citation**: If you use this tool, please cite our paper (link will be added upon publication).
    """)
# === INSERT THE NEW USER GUIDE HERE ===
with st.sidebar.expander("📖 User Guide"):
    st.markdown("""
    ### How to use this tool

    **1. Explore by Subtype**
    - Choose a subtype (HER2, Luminal, TNBC)
    - Optionally filter by target class (e.g., "Signal Transduction")
    - Adjust the slider to see top 10–100 drugs
    - Download the full ranking as CSV

    **2. Search by Gene**
    - Type or select a gene symbol
    - View all drugs that target that gene, with predicted scores per subtype
    - Download the list as CSV

    **3. Compare Drugs**
    - Select two drugs from the dropdowns
    - See a bar chart comparing their scores across subtypes
    - View shared target genes and common pathways

    **4. Visualize Drug Network**
    - Choose a drug to see an interactive network of its targets and associated pathways
    - Use the **Download network as PNG** button to save the current view

    **5. Predict for New Cell Line**
    - Download the template CSV, fill in molecular data (expression Z‑score, CNV, mutation)
    - Upload the file to get predicted drug rankings for that cell line
    - The ranking order is the most reliable indicator

    *All data and predictions are from our corrected NeoDTI model.*
    """)

mode = st.sidebar.radio("Select mode", [
    "🏠 Home",
    "Explore by Subtype",
    "Search by Gene",
    "Compare Drugs",
    "Visualize Drug Network",
    "Predict for New Cell Line"
])
# -------------------------------------------------------------------
# Mode: Home
# -------------------------------------------------------------------
if mode == "🏠 Home":
    st.title("🎯 Welcome to the Breast Cancer Drug Prioritization Tool")
    st.markdown("""
    This interactive web application provides **subtype‑specific drug predictions** using a **corrected NeoDTI** graph neural network model.

    ### 🔬 What can you do?
    - **Explore by Subtype** – View top drugs for HER2, Luminal, and TNBC.
    - **Search by Gene** – Find drugs targeting a specific gene.
    - **Compare Drugs** – Side‑by‑side comparison of two drugs.
    - **Visualize Drug Network** – See drug‑target‑pathway networks.
    - **Predict for New Cell Line** – Upload your own molecular data to get predictions.

    ### 📊 Model Performance
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ROC‑AUC", "0.767", delta=None)
    with col2:
        st.metric("Precision@10", "0.9", delta=None)
    with col3:
        st.metric("Drugs", "4,594", delta=None)
    st.markdown("---")
    st.info("Use the sidebar to explore predictions, compare drugs, and even predict for your own cell line.")
    # Optional: add an image (uncomment if you have a logo file)
    # st.image("logo.png", use_container_width=True)

# -------------------------------------------------------------------
# Mode 1: Explore by Subtype
# -------------------------------------------------------------------
if mode == "Explore by Subtype":
    subtype = st.sidebar.selectbox("Select subtype", ["HER2", "Luminal", "TNBC"])
    target_classes = st.sidebar.multiselect(
        "Filter by target class",
        options=sorted(target_map['target_class'].unique())
    )
    st.subheader(f"Top predicted drugs for {subtype}")
    all_ranked = all_scores[["drug_name", subtype]].sort_values(subtype, ascending=False)
    if target_classes:
        filtered_drugs = target_map[target_map['target_class'].isin(target_classes)]['drug_name'].tolist()
        all_ranked = all_ranked[all_ranked['drug_name'].isin(filtered_drugs)]
    top_n = st.slider("Number of top drugs to display", 10, 100, 20)
    top_df = all_ranked.head(top_n)
    st.dataframe(top_df.style.format({subtype: "{:.4f}"}))
    csv = all_ranked.to_csv(index=False)
    st.download_button("Download full ranking (CSV)", csv, file_name=f"{subtype}_full_ranking.csv", mime="text/csv")

# -------------------------------------------------------------------
# Mode 2: Search by Gene
# -------------------------------------------------------------------
elif mode == "Search by Gene":
    st.subheader("🔎 Find drugs that target a specific gene")
    all_genes = sorted(drug_gene["gene"].unique())
    gene = st.selectbox("Select a gene (type to search)", all_genes)
    if gene:
        drugs_targeting = drug_gene[drug_gene["gene"] == gene]["drug_name"].tolist()
        if drugs_targeting:
            results = all_scores[all_scores["drug_name"].isin(drugs_targeting)].copy()
            results = results.merge(drug_gene[drug_gene["gene"] == gene][["drug_name", "gene"]], on="drug_name", how="left")
            st.write(f"Found {len(results)} drugs targeting **{gene}**")
            st.dataframe(results[["drug_name", "HER2", "Luminal", "TNBC", "gene"]].style.format({
                "HER2": "{:.4f}", "Luminal": "{:.4f}", "TNBC": "{:.4f}"
            }))
            csv = results[["drug_name", "HER2", "Luminal", "TNBC"]].to_csv(index=False)
            st.download_button("Download results (CSV)", csv, file_name=f"drugs_targeting_{gene}.csv")
        else:
            st.warning(f"No drugs found targeting {gene} in our database.")

# -------------------------------------------------------------------
# Mode 3: Compare Drugs
# -------------------------------------------------------------------
elif mode == "Compare Drugs":
    st.subheader("📊 Compare Two Drugs Side by Side")
    drug_list = sorted(all_scores['drug_name'].tolist())
    col1, col2 = st.columns(2)
    with col1:
        drug1 = st.selectbox("Select first drug", drug_list, index=0)
    with col2:
        drug2 = st.selectbox("Select second drug", drug_list, index=min(1, len(drug_list)-1))
    
    if drug1 and drug2:
        scores1 = all_scores[all_scores['drug_name'] == drug1][['HER2','Luminal','TNBC']].iloc[0]
        scores2 = all_scores[all_scores['drug_name'] == drug2][['HER2','Luminal','TNBC']].iloc[0]
        df_comp = pd.DataFrame({
            'Drug': [drug1, drug2],
            'HER2': [scores1['HER2'], scores2['HER2']],
            'Luminal': [scores1['Luminal'], scores2['Luminal']],
            'TNBC': [scores1['TNBC'], scores2['TNBC']]
        })
        df_melt = df_comp.melt(id_vars='Drug', var_name='Subtype', value_name='Score')
        fig = px.bar(df_melt, x='Subtype', y='Score', color='Drug', barmode='group',
                     title=f'Predicted Efficacy Comparison: {drug1} vs {drug2}')
        st.plotly_chart(fig, use_container_width=True)
        
        targets1 = set(drug_gene[drug_gene['drug_name'] == drug1]['gene'].tolist())
        targets2 = set(drug_gene[drug_gene['drug_name'] == drug2]['gene'].tolist())
        shared_genes = targets1.intersection(targets2)
        st.subheader("🎯 Shared Target Genes")
        if shared_genes:
            st.write(", ".join(sorted(shared_genes)))
        else:
            st.write("No shared target genes found.")
        
        class1 = target_map[target_map['drug_name'] == drug1]['target_class'].values
        class2 = target_map[target_map['drug_name'] == drug2]['target_class'].values
        st.subheader("🔗 Primary Target Class")
        st.write(f"{drug1}: {class1[0] if len(class1) else 'Unknown'}")
        st.write(f"{drug2}: {class2[0] if len(class2) else 'Unknown'}")
        
        if shared_genes:
            gp = gene_pathway  # already loaded
            gene_to_pathways = gp.groupby('symbol')['pathway_name'].apply(set).to_dict()
            common_pathways = set()
            for gene in shared_genes:
                if gene in gene_to_pathways:
                    common_pathways.update(gene_to_pathways[gene])
            if common_pathways:
                st.subheader("🔬 Pathways Enriched in Shared Target Genes")
                st.write(", ".join(sorted(common_pathways)[:5]))

# -------------------------------------------------------------------
# Mode 4: Visualize Drug Network
# -------------------------------------------------------------------
elif mode == "Visualize Drug Network":
    st.subheader("🌐 Drug Target‑Pathway Network")
    drug_list = sorted(all_scores['drug_name'].tolist())
    selected_drug = st.selectbox("Select a drug", drug_list)
    
    if selected_drug:
        targets = drug_gene[drug_gene['drug_name'] == selected_drug]['gene'].tolist()
        if not targets:
            st.warning("No target genes found for this drug.")
        else:
            G = nx.Graph()
            G.add_node(selected_drug, node_type='drug', label=selected_drug)
            for gene in targets:
                G.add_node(gene, node_type='gene', label=gene)
                G.add_edge(selected_drug, gene, edge_type='target')
                pathways = gene_pathway[gene_pathway['symbol'] == gene]['pathway_name'].tolist()
                for pw in pathways[:3]:
                    G.add_node(pw, node_type='pathway', label=pw)
                    G.add_edge(gene, pw, edge_type='belongs')
            
            pos = nx.spring_layout(G, k=1.5, seed=42)
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace = go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                edge_traces.append(edge_trace)
            
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_type = G.nodes[node]['node_type']
                if node_type == 'drug':
                    node_color.append('red')
                elif node_type == 'gene':
                    node_color.append('lightblue')
                else:
                    node_color.append('lightgreen')
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                hoverinfo='text',
                marker=dict(color=node_color, size=20, line=dict(width=2, color='DarkSlateGrey'))
            )
            
            fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    title=f'Network for {selected_drug}',
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))
            
            # Display figure with download button side by side
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                try:
                    import plotly.io as pio
                    img_bytes = pio.to_image(fig, format='png', scale=2)
                    st.download_button(
                        label="📸 Download network as PNG",
                        data=img_bytes,
                        file_name=f"{selected_drug}_network.png",
                        mime="image/png",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning("Download requires the `kaleido` package. Install it with `pip install kaleido`. For now, you can right‑click on the plot and select 'Save image as'.")
            
            # Expandable sections for target genes and pathways
            with st.expander("Show target genes"):
                st.write(", ".join(targets))
            all_pathways = set()
            for gene in targets:
                pathways = gene_pathway[gene_pathway['symbol'] == gene]['pathway_name'].tolist()
                all_pathways.update(pathways)
            with st.expander("Show associated pathways"):
                st.write(", ".join(sorted(all_pathways)[:20]))


# -------------------------------------------------------------------
# Mode 5: Predict for New Cell Line
# -------------------------------------------------------------------

elif mode == "Predict for New Cell Line":
    st.subheader("🧬 Predict Drug Efficacy for a New Cell Line")
    st.markdown("""
    Upload a CSV file with columns:
    - **gene** (gene symbol)
    - **expression_zscore** (e.g., from RNA‑seq)
    - **cnv_log2ratio** (copy number log2 ratio)
    - **mutation** (0 or 1)

    The thresholds used:
    - Overexpression: Z‑score > 2
    - Amplification: log2 ratio > 0.3
    - Mutation: 1
    """)
    
    template = pd.DataFrame({
        'gene': ['EGFR', 'MTOR', 'TP53'],
        'expression_zscore': [1.5, 2.3, -0.8],
        'cnv_log2ratio': [0.1, 0.5, -0.2],
        'mutation': [0, 0, 1]
    })
    csv_template = template.to_csv(index=False)
    st.download_button("Download template CSV", csv_template, "new_cell_template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload your cell line data (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Validate columns
        required = ['gene', 'expression_zscore', 'cnv_log2ratio', 'mutation']
        if not all(col in df.columns for col in required):
            st.error("CSV must contain columns: " + ", ".join(required))
        else:
            # Build sets
            overexpression = set(df[df['expression_zscore'] > 2]['gene'].tolist())
            amplification = set(df[df['cnv_log2ratio'] > 0.3]['gene'].tolist())
            mutation = set(df[df['mutation'] == 1]['gene'].tolist())
            
            st.write(f"Found {len(overexpression)} overexpressed genes, {len(amplification)} amplified, {len(mutation)} mutated.")
            
            with st.spinner("Running prediction..."):
                result = predict_for_new_cell(overexpression, amplification, mutation)
            
            st.subheader("Top predicted drugs for this cell line")
            top_n = st.slider("Number of top drugs to display", 10, 100, 20, key="inference_top_n")
            st.dataframe(result.head(top_n).style.format({"predicted_score": "{:.4f}"}))
            csv = result.to_csv(index=False)
            st.download_button("Download full ranking (CSV)", csv, "new_cell_predictions.csv", "text/csv")

st.markdown("---")
st.caption("The **ranking order** is the most reliable indicator for drug prioritisation. Predicted scores are raw model outputs; absolute values are influenced by the simplified cell embedding method and should be interpreted relative to each other.")