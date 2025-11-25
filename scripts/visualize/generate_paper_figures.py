#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Paper

Creates 6 figures for cross-LLM backdoor detection paper:
1. System Architecture
2. Cross-LLM Detection Heatmap (6×6)
3. Feature Stability Box Plot
4. Per-Model Discriminative Features (Radar Charts)
5. Cross-Model Correlation Heatmap
6. Ensemble Approach Comparison

Usage:
    python generate_paper_figures.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12


# =============================================================================
# Figure 2: Cross-LLM Detection Heatmap (6×6)
# =============================================================================

def generate_detection_heatmap():
    """Generate 6×6 detection accuracy heatmap"""
    print("\n📊 Generating Figure 2: Cross-LLM Detection Heatmap...")

    # Load detection matrix
    results_dir = Path("data/cross_llm_results")
    matrix_df = pd.read_csv(results_dir / "detection_matrix_accuracy.csv", index_col=0)

    # Model display names
    model_names = {
        "gpt51": "GPT-5.1",
        "claude45": "Claude 4.5",
        "grok41": "Grok 4.1",
        "llama4": "Llama 4",
        "gptoss": "GPT-OSS",
        "deepseek": "DeepSeek"
    }

    # Rename rows/columns
    matrix_df.index = [model_names[m] for m in matrix_df.index]
    matrix_df.columns = [model_names[m] for m in matrix_df.columns]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt='.1%',
        cmap='RdYlGn',
        vmin=0.4,
        vmax=1.0,
        cbar_kws={'label': 'Detection Accuracy'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    # Highlight diagonal (same-model detection)
    for i in range(6):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    # Labels
    ax.set_xlabel('Test Model', fontweight='bold')
    ax.set_ylabel('Train Model', fontweight='bold')
    ax.set_title('Cross-LLM Backdoor Detection Accuracy Matrix\n(Blue boxes: same-model detection)',
                 fontweight='bold', pad=15)

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Add annotations
    same_model_avg = np.mean([matrix_df.iloc[i, i] for i in range(6)])
    cross_model_vals = [matrix_df.iloc[i, j] for i in range(6) for j in range(6) if i != j]
    cross_model_avg = np.mean(cross_model_vals)

    textstr = f'Same-model: {same_model_avg:.1%}\nCross-model: {cross_model_avg:.1%}\nGap: {same_model_avg - cross_model_avg:.1%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('figures/fig2_detection_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig2_detection_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/fig2_detection_heatmap.pdf")
    plt.close()


# =============================================================================
# Figure 3: Feature Stability Box Plot
# =============================================================================

def generate_feature_stability_boxplot():
    """Generate box plot showing CV distribution across feature categories"""
    print("\n📊 Generating Figure 3: Feature Stability Box Plot...")

    # Load feature matrices
    data_dir = Path("data/analysis")
    models = ["gpt51", "claude45", "grok41", "llama4", "gptoss", "deepseek"]

    # Feature categories (must match actual 51 features in data)
    # Temporal (10), Sequence (15), Action (12), Data-Flow (14) = 51 total
    temporal_features = ['avg_duration', 'max_duration', 'std_duration', 'avg_delay',
                        'max_delay', 'total_duration', 'execution_rate', 'timing_entropy',
                        'has_burst', 'burst_intensity']

    sequence_features = ['unique_bigrams', 'max_bigram_freq', 'bigram_diversity',
                        'unique_trigrams', 'trigram_diversity', 'repetition_ratio',
                        'max_consecutive_repeats', 'sequence_length', 'unique_transitions',
                        'transition_entropy', 'has_loops', 'loop_count', 'dependency_ratio',
                        'total_dependencies', 'rare_tool_ratio']

    action_features = ['tool_count', 'unique_tools', 'tool_diversity', 'most_common_tool_freq',
                      'tool_transition_entropy', 'file_read_count', 'web_fetch_count',
                      'tool_switching_rate', 'unauthorized_tool_access', 'tool_sequence_anomaly_score',
                      'avg_params_length', 'max_params_length']

    dataflow_features = ['avg_input_size', 'max_input_size', 'avg_output_size', 'max_output_size',
                        'avg_io_ratio', 'max_io_ratio', 'std_input_size', 'std_output_size',
                        'sensitive_data_mentions', 'data_flow_complexity', 'input_diversity',
                        'output_diversity', 'io_entropy', 'io_coupling']

    # Compute CV for each feature
    feature_cvs = []

    # Order must match data file: temporal, sequence, action, dataflow
    for feature_list, category in [(temporal_features, 'Temporal'),
                                   (sequence_features, 'Sequence'),
                                   (action_features, 'Action'),
                                   (dataflow_features, 'Data-Flow')]:
        for feature_idx, feature_name in enumerate(feature_list):
            model_means = []

            # Get feature index in the 51-feature vector (order: temporal, sequence, action, dataflow)
            all_features = temporal_features + sequence_features + action_features + dataflow_features
            global_idx = all_features.index(feature_name)

            for model in models:
                benign = np.load(data_dir / f"{model}_benign_features.npy")
                backdoor = np.load(data_dir / f"{model}_backdoor_features.npy")
                all_samples = np.vstack([benign, backdoor])

                model_mean = np.mean(all_samples[:, global_idx])
                model_means.append(model_mean)

            # Compute CV
            if np.mean(model_means) > 0:
                cv = np.std(model_means) / np.mean(model_means)
            else:
                cv = 0.0

            feature_cvs.append({'Category': category, 'Feature': feature_name, 'CV': cv})

    df = pd.DataFrame(feature_cvs)

    # Create box plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Box plot
    sns.boxplot(x='Category', y='CV', data=df, palette='Set2', ax=ax)

    # Add horizontal lines for thresholds
    ax.axhline(y=0.2, color='green', linestyle='--', linewidth=1.5, label='Stable threshold (CV=0.2)')
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, label='Unstable threshold (CV=0.8)')

    # Labels
    ax.set_xlabel('Feature Category', fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (CV)', fontweight='bold')
    ax.set_title('Cross-LLM Feature Stability by Category\n(Lower CV = more stable across models)',
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add stats
    for i, category in enumerate(['Temporal', 'Sequence', 'Action', 'Data-Flow']):
        cat_data = df[df['Category'] == category]['CV']
        median = cat_data.median()
        ax.text(i, median, f'{median:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/fig3_feature_stability.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig3_feature_stability.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/fig3_feature_stability.pdf")
    plt.close()


# =============================================================================
# Figure 4: Per-Model Discriminative Features (Radar Charts)
# =============================================================================

def generate_discriminative_features_radar():
    """Generate 6 radar charts showing top discriminative features per model"""
    print("\n📊 Generating Figure 4: Per-Model Discriminative Features...")

    # Top discriminative features per model (from RQ7 analysis)
    model_features = {
        'GPT-5.1': {'data_flow_complexity': 0.294, 'tool_entropy': 0.22, 'sequence_length': 0.18,
                   'avg_duration': 0.15, 'io_entropy': 0.12, 'burst_intensity': 0.10},
        'Claude 4.5': {'tool_entropy': 0.269, 'data_flow_complexity': 0.21, 'avg_params_length': 0.19,
                      'sequence_length': 0.17, 'total_duration': 0.14, 'io_coupling': 0.11},
        'Llama 4': {'max_io_ratio': 0.325, 'data_flow_complexity': 0.25, 'avg_output_size': 0.22,
                   'tool_diversity': 0.18, 'sequence_length': 0.15, 'avg_delay': 0.12},
        'Grok 4.1': {'avg_duration': 0.222, 'sequence_length': 0.20, 'tool_call_rate': 0.17,
                    'total_duration': 0.15, 'step_duration_cv': 0.13, 'burst_intensity': 0.10},
        'GPT-OSS': {'sequence_length': 0.287, 'data_flow_complexity': 0.23, 'tool_diversity': 0.19,
                   'avg_params_length': 0.16, 'io_entropy': 0.14, 'max_dependencies': 0.11},
        'DeepSeek': {'burst_intensity': 0.241, 'avg_delay': 0.21, 'tool_entropy': 0.18,
                    'max_delay': 0.16, 'sequence_length': 0.14, 'data_flow_complexity': 0.12}
    }

    # Create 2×3 subplot
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    for idx, (model_name, features) in enumerate(model_features.items()):
        ax = axes[idx]

        # Prepare data
        categories = list(features.keys())
        values = list(features.values())

        # Number of variables
        N = len(categories)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label='Effect Size (Cohen\'s d)')
        ax.fill(angles, values, alpha=0.25)

        # Fix axis to go from 0 to 0.35
        ax.set_ylim(0, 0.35)

        # Draw one axe per variable and add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace('_', ' ').title()[:15] for c in categories], size=7)

        # Add title
        ax.set_title(model_name, fontweight='bold', pad=10)

        # Add grid
        ax.grid(True)

        # Highlight top feature
        max_idx = values.index(max(values[:-1]))
        ax.plot([angles[max_idx]], [values[max_idx]], 'r*', markersize=15)

    plt.suptitle('Top Discriminative Features by Model\n(Red star: highest Cohen\'s d)',
                 fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('figures/fig4_discriminative_features.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig4_discriminative_features.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/fig4_discriminative_features.pdf")
    plt.close()


# =============================================================================
# Figure 5: Cross-Model Correlation Heatmap
# =============================================================================

def generate_correlation_heatmap():
    """Generate cross-model feature correlation heatmap"""
    print("\n📊 Generating Figure 5: Cross-Model Correlation Heatmap...")

    # Load feature matrices
    data_dir = Path("data/analysis")
    models = ["gpt51", "claude45", "grok41", "llama4", "gptoss", "deepseek"]
    model_names = {
        "gpt51": "GPT-5.1",
        "claude45": "Claude 4.5",
        "grok41": "Grok 4.1",
        "llama4": "Llama 4",
        "gptoss": "GPT-OSS",
        "deepseek": "DeepSeek"
    }

    # Compute mean feature vectors for each model
    model_vectors = {}
    for model in models:
        benign = np.load(data_dir / f"{model}_benign_features.npy")
        backdoor = np.load(data_dir / f"{model}_backdoor_features.npy")
        all_samples = np.vstack([benign, backdoor])

        # Mean feature vector across all traces
        model_vectors[model] = np.mean(all_samples, axis=0)

    # Compute correlation matrix
    corr_matrix = np.zeros((6, 6))
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            corr_matrix[i, j] = np.corrcoef(model_vectors[model1], model_vectors[model2])[0, 1]

    # Create DataFrame
    corr_df = pd.DataFrame(corr_matrix,
                          index=[model_names[m] for m in models],
                          columns=[model_names[m] for m in models])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr_df,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Pearson Correlation'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    # Labels
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.set_title('Cross-Model Feature Vector Correlation\n(Higher correlation → better transfer expected)',
                 fontweight='bold', pad=15)

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig('figures/fig5_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig5_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/fig5_correlation_heatmap.pdf")
    plt.close()


# =============================================================================
# Figure 6: Ensemble Approach Comparison
# =============================================================================

def generate_ensemble_comparison():
    """Generate bar chart comparing 4 ensemble approaches"""
    print("\n📊 Generating Figure 6: Ensemble Approach Comparison...")

    # Load ensemble results
    results_dir = Path("data/ensemble_results")

    with open(results_dir / "ensemble_summary.json") as f:
        ensemble_data = json.load(f)

    # Prepare data
    approaches = ['Baseline\n(Single-Model)', 'Pooled\nTraining', 'Ensemble\nVoting', 'Model-Aware']
    same_model = [
        ensemble_data['baseline']['same_model_acc'],
        ensemble_data['pooled']['same_model_acc'],
        ensemble_data['ensemble']['same_model_acc'],
        ensemble_data['model_aware']['same_model_acc']
    ]
    cross_model = [
        ensemble_data['baseline']['cross_model_acc'],
        ensemble_data['pooled']['cross_model_acc'],
        ensemble_data['ensemble']['cross_model_acc'],
        ensemble_data['model_aware']['cross_model_acc']
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(approaches))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, same_model, width, label='Same-Model', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, cross_model, width, label='Cross-Model', color='#3498db', edgecolor='black')

    # Add generalization gap line
    for i in range(len(approaches)):
        gap = same_model[i] - cross_model[i]
        if gap > 0.01:  # Show gap for baseline
            ax.plot([i - width/2, i + width/2], [same_model[i], cross_model[i]],
                   'r--', linewidth=2)
            ax.text(i, (same_model[i] + cross_model[i]) / 2, f'Gap:\n{gap:.1%}',
                   ha='center', fontsize=8, color='red', fontweight='bold')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Labels
    ax.set_xlabel('Detection Approach', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Ensemble Approach Comparison\n(Goal: Eliminate generalization gap)',
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    # Add 50% random guessing line
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, label='Random Guessing (50%)')

    # Highlight best approach
    best_idx = 3  # Model-aware
    ax.add_patch(Rectangle((best_idx - 0.5, 0), 1, 1.05, fill=False, edgecolor='gold', lw=3))
    ax.text(best_idx, 0.95, '✓ Best', ha='center', fontsize=10, color='gold', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/fig6_ensemble_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig6_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figures/fig6_ensemble_comparison.pdf")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*70)
    print("Generating Publication-Quality Figures")
    print("="*70)

    # Create figures directory
    Path("figures").mkdir(exist_ok=True)

    # Generate all figures
    generate_detection_heatmap()  # Figure 2
    generate_feature_stability_boxplot()  # Figure 3
    generate_discriminative_features_radar()  # Figure 4
    generate_correlation_heatmap()  # Figure 5
    generate_ensemble_comparison()  # Figure 6

    print("\n" + "="*70)
    print("✅ All Figures Generated Successfully!")
    print("="*70)
    print("\nGenerated Files:")
    print("  - figures/fig2_detection_heatmap.pdf (6×6 detection matrix)")
    print("  - figures/fig3_feature_stability.pdf (box plot by category)")
    print("  - figures/fig4_discriminative_features.pdf (6 radar charts)")
    print("  - figures/fig5_correlation_heatmap.pdf (6×6 correlation)")
    print("  - figures/fig6_ensemble_comparison.pdf (4 approaches bar chart)")
    print("\nNote: Figure 1 (System Architecture) requires manual diagram creation")
    print("      Recommend using draw.io, Visio, or TikZ/LaTeX")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
