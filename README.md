# Cross-LLM Backdoor Detection

**Cross-LLM Generalization of Behavioral Backdoor Detection in AI Agent Supply Chains**

This repository contains the implementation and experimental code for our paper on detecting backdoored AI agents through behavioral trace analysis across 6 production LLMs.

## Overview

We present a comprehensive multi-LLM backdoor detection system that analyzes behavioral traces across diverse language models. Our key contributions:

1. **Cross-LLM Generalization Gap**: Discovered 43.4% accuracy drop when applying single-model detectors across LLMs (49.2% vs 92.7%)
2. **Model-Aware Detection Solution**: Achieved 90.6% universal accuracy across 6 production LLMs using model-aware features
3. **Architectural Analysis**: Identified temporal features as primary source of model-specific behavioral variance (CV > 0.8)
4. **Deployment Guidelines**: Provided evidence-based strategies for single vs multi-LLM environments
5. **Production-Scale Validation**: Evaluated on 1,198 traces from GPT-5.1, Claude 4.5, Grok 4.1, Llama 4, GPT-OSS, DeepSeek

## Installation

### Requirements

- Python 3.8+
- pip (Python package manager)

### Setup

```bash
# Clone repository
git clone https://github.com/arunsanna/cross-llm-backdoor-detection.git
cd cross-llm-backdoor-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Our multi-LLM dataset consists of **1,198 execution traces** (99.8% success rate):

| Model | Provider | Benign | Backdoor | Total |
|-------|----------|--------|----------|-------|
| **GPT-5.1** | OpenAI | 100 | 100 | 200 |
| **Claude Sonnet 4.5** | Anthropic | 100 | 100 | 200 |
| **Grok 4.1 Fast** | XAI | 100 | 100 | 200 |
| **Llama 4 Maverick** | Meta | 100 | 100 | 200 |
| **GPT-OSS 120B** | OpenAI | 100 | 100 | 200 |
| **DeepSeek Chat V3.1** | DeepSeek | 99 | 99 | 198 |
| **TOTAL** | - | **599** | **599** | **1,198** |

**Features**: 51 behavioral features (action, sequence, data-flow, temporal)
**Access**: OpenRouter API (unified multi-LLM access)

See the paper in `paper/` directory for complete documentation.

## Project Structure

```
cross-llm-backdoor-detection/
├── src/                        # Core library code
│   ├── features/              # Feature extraction (51 features)
│   ├── agents/                # Agent wrappers & backdoor implementations
│   ├── models/                # Detector models
│   └── utils/                 # Utility functions
├── scripts/
│   ├── collect/               # Data collection scripts
│   ├── train/                 # Model training scripts
│   ├── evaluate/              # Evaluation & analysis scripts
│   ├── visualize/             # Figure generation scripts
│   └── validate/              # API validation scripts
├── tests/                      # Test files
├── paper/                      # LaTeX paper source
│   ├── sections/              # Paper sections
│   └── figures/               # Paper figures (PDF)
├── config/                     # Model configurations
├── requirements.txt            # Python dependencies
└── .env.example               # Environment template
```

**Note**: Data, models, and generated results are not included in this repository.
Run collection scripts to generate your own dataset.

## Quick Start

### 1. Feature Extraction

```bash
# Extract features from traces
python scripts/evaluate/analyze_multi_llm_features.py
```

### 2. Cross-LLM Detection Matrix

```bash
# Generate 6×6 detection matrix (36 experiments)
python scripts/evaluate/cross_llm_detection_matrix.py
```

### 3. Ensemble Experiments

```bash
# Compare 4 detection approaches
python scripts/train/multi_llm_ensemble_experiments.py
```

### 4. Data Collection (Requires API Key)

```bash
# Set up environment
cp .env.example .env
# Edit .env with your OpenRouter API key

# Collect traces
python scripts/collect/collect_multi_llm_incremental.py --workers 15
```

## Reproducibility

All experiments use fixed random seeds (42) for reproducibility. Results are saved as JSON files.

## Key Results

### Cross-LLM Detection Performance

| Approach | Same-Model | Cross-Model | Overall | Gap |
|----------|------------|-------------|---------|-----|
| **Baseline (Single-Model)** | 92.7% | 49.2% | 56.5% | **43.4%** ⚠️ |
| **Pooled Training** | 89.8% | 89.8% | 89.8% | 0.0% |
| **Model-Aware** | **90.6%** | **90.6%** | **90.6%** | **0.0%** ✅ |
| **Ensemble Voting** | 62.8% | 62.8% | 62.8% | 0.0% ❌ |

### Per-Model Same-Model Accuracy

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| **Llama 4 Maverick** | 100.0% | 100.0% | 1.000 |
| **DeepSeek Chat V3.1** | 99.0% | 99.0% | 0.992 |
| **Claude Sonnet 4.5** | 93.5% | 93.8% | 0.992 |
| **GPT-OSS 120B** | 92.0% | 91.3% | 0.980 |
| **Grok 4.1 Fast** | 89.5% | 90.5% | 0.979 |
| **GPT-5.1** | 82.0% | 78.0% | 0.915 |
| **Average** | **92.7%** | **92.1%** | **0.976** |

### Key Insights

🔴 **Problem**: Single-model detectors fail across LLMs (49.2% = random guessing)
✅ **Solution**: Model-aware detection achieves 90.6% universal accuracy
📊 **Evidence**: 36 experiments (6×6 matrix), 43.4% generalization gap quantified

## Citation

\`\`\`bibtex
@inproceedings{sanna2025crossllm,
  title={Cross-LLM Generalization of Behavioral Backdoor Detection in AI Agent Supply Chains},
  author={Sanna, Arun Chowdary},
  booktitle={Proceedings of the 34th USENIX Security Symposium},
  year={2025},
  note={Comprehensive evaluation across 6 production LLMs with 1,198 traces}
}
\`\`\`

## License

MIT License

## Contact

- **Author**: Arun Chowdary Sanna
- **Affiliation**: Precise Software Solutions
- **Email**: arun.sanna@outlook.com

For questions or issues, please open a GitHub issue.
