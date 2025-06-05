"""Compare GIST loss vs MNRL approaches for leadership embedding analysis.

This script:
1. Automatically identifies the best models from each approach (GIST and MNRL)
2. Compares their performance on both IPIP and leadership data
3. Performs statistical significance testing between approaches
4. Generates visualizations and a comprehensive comparison report

Usage:
    python scripts/compare_gist_vs_mnrl.py [--gist_model_dir models/ipip_gist_*] [--mnrl_model_dir models/ipip_mnrl_*]

For help on all options:
    python scripts/compare_gist_vs_mnrl.py --help

Outputs:
    data/visualizations/model_comparison/ - Comparison results and visualizations
"""

import argparse
import logging
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import ttest_ind
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Constants
IPIP_DATA = "data/processed/leadership_ipip_mapping.csv"
LEADERSHIP_DATA = "data/processed/leadership_focused_clean.csv"
OUTPUT_DIR = "data/visualizations/model_comparison"
RANDOM_SEED = 42

# Ensure reproducibility
np.random.seed(RANDOM_SEED)

def find_latest_models(gist_pattern=None, mnrl_pattern=None):
    """Find the latest GIST and MNRL models."""
    if gist_pattern is None:
        gist_pattern = "models/ipip_gist_*"
    if mnrl_pattern is None:
        mnrl_pattern = "models/ipip_mnrl_*"
    
    # Find all matching model directories
    gist_models = sorted(glob.glob(gist_pattern))
    mnrl_models = sorted(glob.glob(mnrl_pattern))
    
    if not gist_models:
        logger.warning(f"No GIST models found matching pattern: {gist_pattern}")
        gist_model = None
    else:
        # Get the most recent model (assume directories are named with timestamps)
        gist_model = gist_models[-1]
        logger.info(f"Latest GIST model: {gist_model}")
    
    if not mnrl_models:
        logger.warning(f"No MNRL models found matching pattern: {mnrl_pattern}")
        mnrl_model = None
    else:
        # Get the most recent model
        mnrl_model = mnrl_models[-1]
        logger.info(f"Latest MNRL model: {mnrl_model}")
    
    return gist_model, mnrl_model

def load_dataset(dataset_name):
    """Load the specified dataset."""
    logger.info(f"Loading {dataset_name} dataset...")
    
    if dataset_name.lower() == "ipip":
        file_path = IPIP_DATA
        text_col = "Text"
        label_col = "StandardConstruct"
    elif dataset_name.lower() == "leadership":
        file_path = LEADERSHIP_DATA
        text_col = "Text"
        label_col = "StandardConstruct"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to load dataset {file_path}: {str(e)}")
        raise
    
    # Basic cleaning
    df = df.dropna(subset=[text_col, label_col])
    
    # Get construct statistics
    constructs = df[label_col].value_counts()
    logger.info(f"Dataset has {len(df)} items across {len(constructs)} constructs")
    
    # Return the dataset and important info
    return {
        "df": df,
        "text_col": text_col,
        "label_col": label_col,
        "n_constructs": len(constructs),
        "construct_counts": constructs
    }

def evaluate_model(model, dataset):
    """Evaluate a model on a dataset."""
    logger.info(f"Evaluating model...")
    
    df = dataset["df"]
    text_col = dataset["text_col"]
    label_col = dataset["label_col"]
    n_constructs = dataset["n_constructs"]
    
    texts = df[text_col].tolist()
    true_labels = df[label_col].tolist()
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Ensure n_clusters doesn't exceed number of samples
    n_clusters = min(n_constructs, len(embeddings)-1)
    
    # Perform clustering
    logger.info(f"Clustering embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init='auto')
    predicted_clusters = kmeans.fit_predict(embeddings)
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, predicted_clusters)
    nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
    
    # Calculate purity
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predicted_clusters)
    purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)
    
    logger.info(f"Metrics - ARI: {ari:.4f}, NMI: {nmi:.4f}, Purity: {purity:.4f}")
    
    # Generate TSNE visualization
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    embedded = tsne.fit_transform(embeddings)
    
    return {
        "embeddings": embeddings,
        "predicted_clusters": predicted_clusters,
        "true_labels": true_labels,
        "tsne": embedded,
        "metrics": {
            "ari": ari,
            "nmi": nmi,
            "purity": purity
        }
    }

def compare_models(gist_model, mnrl_model):
    """Compare GIST and MNRL models on IPIP and leadership data."""
    results = {
        "ipip": {},
        "leadership": {}
    }
    
    # Load datasets
    ipip_dataset = load_dataset("ipip")
    leadership_dataset = load_dataset("leadership")
    
    # Evaluate on IPIP data
    if gist_model:
        logger.info("Evaluating GIST model on IPIP data...")
        results["ipip"]["gist"] = evaluate_model(gist_model, ipip_dataset)
    
    if mnrl_model:
        logger.info("Evaluating MNRL model on IPIP data...")
        results["ipip"]["mnrl"] = evaluate_model(mnrl_model, ipip_dataset)
    
    # Evaluate on leadership data
    if gist_model:
        logger.info("Evaluating GIST model on leadership data...")
        results["leadership"]["gist"] = evaluate_model(gist_model, leadership_dataset)
    
    if mnrl_model:
        logger.info("Evaluating MNRL model on leadership data...")
        results["leadership"]["mnrl"] = evaluate_model(mnrl_model, leadership_dataset)
    
    return results

def create_comparison_visualizations(results, output_dir):
    """Create visualizations comparing the models."""
    logger.info("Creating comparison visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metrics comparison chart
    plt.figure(figsize=(15, 10))
    
    datasets = list(results.keys())
    models = []
    if "gist" in results[datasets[0]]:
        models.append("gist")
    if "mnrl" in results[datasets[0]]:
        models.append("mnrl")
    
    metrics = ["ari", "nmi", "purity"]
    metric_labels = ["Adjusted Rand Index", "Normalized Mutual Info", "Cluster Purity"]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        
        for j, model in enumerate(models):
            values = []
            for dataset in datasets:
                if model in results[dataset]:
                    values.append(results[dataset][model]["metrics"][metric])
                else:
                    values.append(0)
            
            plt.bar(x + (j - len(models)/2 + 0.5) * width, values, width, 
                    label=f"{model.upper()}")
        
        plt.xlabel('Dataset')
        plt.ylabel(metric_labels[i])
        plt.title(f'{metric_labels[i]} Comparison')
        plt.xticks(x, [d.capitalize() for d in datasets])
        plt.ylim(0, 1.0)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300)
    
    # Create a summary CSV file
    df_rows = []
    for dataset in datasets:
        for model in models:
            if model in results[dataset]:
                row = {
                    "dataset": dataset,
                    "model": model.upper(),
                    "ari": results[dataset][model]["metrics"]["ari"],
                    "nmi": results[dataset][model]["metrics"]["nmi"],
                    "purity": results[dataset][model]["metrics"]["purity"]
                }
                df_rows.append(row)
    
    metrics_df = pd.DataFrame(df_rows)
    metrics_df.to_csv(f"{output_dir}/comparison_metrics.csv", index=False)
    
    logger.info(f"Comparison visualizations saved to {output_dir}")

def create_comparison_report(results, gist_model_path, mnrl_model_path, output_dir):
    """Create a comprehensive comparison report."""
    report_path = f"{output_dir}/model_comparison_report.txt"
    
    with open(report_path, "w") as f:
        f.write("GIST vs MNRL Model Comparison Report\n")
        f.write("==================================\n\n")
        
        f.write(f"GIST Model: {os.path.basename(gist_model_path) if gist_model_path else 'Not available'}\n")
        f.write(f"MNRL Model: {os.path.basename(mnrl_model_path) if mnrl_model_path else 'Not available'}\n\n")
        
        f.write("Performance Comparison:\n")
        f.write("-" * 50 + "\n\n")
        
        for dataset in results.keys():
            f.write(f"{dataset.upper()} Dataset:\n")
            f.write("-" * 30 + "\n")
            
            models = []
            if "gist" in results[dataset]:
                models.append("gist")
            if "mnrl" in results[dataset]:
                models.append("mnrl")
            
            # Compare metrics
            metrics = ["ari", "nmi", "purity"]
            metric_names = ["Adjusted Rand Index", "Normalized Mutual Info", "Cluster Purity"]
            
            for metric, name in zip(metrics, metric_names):
                f.write(f"\n{name}:\n")
                
                values = {model: results[dataset][model]["metrics"][metric] for model in models}
                for model in models:
                    f.write(f"  {model.upper()}: {values[model]:.4f}\n")
                
                if len(models) > 1:
                    # Determine better model
                    best_model = max(models, key=lambda m: values[m])
                    diff = abs(values["gist"] - values["mnrl"])
                    f.write(f"  Difference: {diff:.4f} ({best_model.upper()} is better)\n")
            
            f.write("\n")
        
        # Overall analysis
        f.write("\nOverall Analysis:\n")
        f.write("-" * 30 + "\n\n")
        
        if "gist" in results["ipip"] and "mnrl" in results["ipip"]:
            gist_ipip_ari = results["ipip"]["gist"]["metrics"]["ari"]
            mnrl_ipip_ari = results["ipip"]["mnrl"]["metrics"]["ari"]
            
            if gist_ipip_ari > mnrl_ipip_ari:
                f.write(f"GIST approach performs better on IPIP data (ARI: {gist_ipip_ari:.4f} vs {mnrl_ipip_ari:.4f})\n")
            else:
                f.write(f"MNRL approach performs better on IPIP data (ARI: {mnrl_ipip_ari:.4f} vs {gist_ipip_ari:.4f})\n")
        
        if "gist" in results["leadership"] and "mnrl" in results["leadership"]:
            gist_lead_ari = results["leadership"]["gist"]["metrics"]["ari"]
            mnrl_lead_ari = results["leadership"]["mnrl"]["metrics"]["ari"]
            
            if gist_lead_ari > mnrl_lead_ari:
                f.write(f"GIST approach performs better on leadership data (ARI: {gist_lead_ari:.4f} vs {mnrl_lead_ari:.4f})\n")
            else:
                f.write(f"MNRL approach performs better on leadership data (ARI: {mnrl_lead_ari:.4f} vs {gist_lead_ari:.4f})\n")
        
        # Final recommendation
        f.write("\nRecommendation:\n")
        if (("gist" in results["ipip"] and "mnrl" in results["ipip"]) and 
            ("gist" in results["leadership"] and "mnrl" in results["leadership"])):
            
            # Calculate average ARI across datasets
            gist_avg = (results["ipip"]["gist"]["metrics"]["ari"] + 
                         results["leadership"]["gist"]["metrics"]["ari"]) / 2
            
            mnrl_avg = (results["ipip"]["mnrl"]["metrics"]["ari"] + 
                         results["leadership"]["mnrl"]["metrics"]["ari"]) / 2
            
            if gist_avg > mnrl_avg:
                f.write(f"Based on average ARI across datasets, the GIST approach is recommended "
                        f"(avg ARI: {gist_avg:.4f} vs {mnrl_avg:.4f}).\n")
            else:
                f.write(f"Based on average ARI across datasets, the MNRL approach is recommended "
                        f"(avg ARI: {mnrl_avg:.4f} vs {gist_avg:.4f}).\n")
        
        # Leadership construct findings
        f.write("\nLeadership Construct Analysis:\n")
        f.write("-" * 30 + "\n\n")
        
        best_lead_ari = 0
        best_approach = None
        
        if "gist" in results["leadership"]:
            gist_lead_ari = results["leadership"]["gist"]["metrics"]["ari"]
            if gist_lead_ari > best_lead_ari:
                best_lead_ari = gist_lead_ari
                best_approach = "GIST"
        
        if "mnrl" in results["leadership"]:
            mnrl_lead_ari = results["leadership"]["mnrl"]["metrics"]["ari"]
            if mnrl_lead_ari > best_lead_ari:
                best_lead_ari = mnrl_lead_ari
                best_approach = "MNRL"
        
        if best_lead_ari < 0.05:
            f.write("Both approaches indicate that leadership constructs have substantial semantic overlap "
                   f"(best ARI: {best_lead_ari:.4f} with {best_approach}).\n")
            f.write("This supports the hypothesis that leadership constructs, as currently measured, do not "
                   "represent semantically distinct categories.\n")
        elif best_lead_ari < 0.2:
            f.write("The best approach shows modest separation of leadership constructs "
                   f"(ARI: {best_lead_ari:.4f} with {best_approach}).\n")
            f.write("This suggests some separability, but still indicates substantial overlap between constructs.\n")
        else:
            f.write("The best approach shows meaningful separation of leadership constructs "
                   f"(ARI: {best_lead_ari:.4f} with {best_approach}).\n")
            f.write("This suggests leadership constructs may have some semantic distinctiveness.\n")
        
        f.write(f"\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    logger.info(f"Comparison report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare GIST loss vs MNRL approaches")
    
    # Model parameters
    parser.add_argument('--gist_model_dir', type=str, default=None,
                        help="Directory containing the GIST model (default: latest)")
    parser.add_argument('--mnrl_model_dir', type=str, default=None,
                        help="Directory containing the MNRL model (default: latest)")
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help="Directory to save comparison results")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"{args.output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find models
    gist_model_path, mnrl_model_path = find_latest_models(
        gist_pattern=args.gist_model_dir,
        mnrl_pattern=args.mnrl_model_dir
    )
    
    # Load models
    gist_model = None
    mnrl_model = None
    
    if gist_model_path:
        logger.info(f"Loading GIST model from {gist_model_path}...")
        try:
            gist_model = SentenceTransformer(gist_model_path)
        except Exception as e:
            logger.error(f"Failed to load GIST model: {str(e)}")
    
    if mnrl_model_path:
        logger.info(f"Loading MNRL model from {mnrl_model_path}...")
        try:
            mnrl_model = SentenceTransformer(mnrl_model_path)
        except Exception as e:
            logger.error(f"Failed to load MNRL model: {str(e)}")
    
    if not gist_model and not mnrl_model:
        logger.error("No models could be loaded. Exiting.")
        return
    
    # Compare models
    results = compare_models(gist_model, mnrl_model)
    
    # Create visualizations
    create_comparison_visualizations(results, output_dir)
    
    # Create comparison report
    create_comparison_report(results, gist_model_path, mnrl_model_path, output_dir)
    
    logger.info(f"Comparison complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()