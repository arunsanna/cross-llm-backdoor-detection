# Behavioral Anomaly Detection for AI Agent Supply Chain Security

**Research Implementation** | **Status:** Phase 1 - Data Collection

Extending ["Malice in Agentland"](https://arxiv.org/abs/2510.05159) with behavioral anomaly detection defense.

---

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Add your API keys
# OPENAI_API_KEY=sk-...
# Or configure local LLM
```

### 3. Run Data Collection (Phase 1)

```bash
# Collect clean traces
python src/agents/collect_traces.py --mode clean --num-traces 1000

# Implement backdoors and collect poisoned traces
python src/agents/collect_traces.py --mode backdoor --threat-model TM1 --num-traces 200
```

---

## Project Structure

```
behavioral-anomaly-detection/
├── data/                   # Datasets
│   ├── clean_traces/       # 1000+ benign traces
│   ├── backdoor_traces/    # 500+ poisoned (TM1/TM2/TM3)
│   └── mixed_traces/       # Evaluation data
├── src/                    # Source code
│   ├── agents/             # Agent instrumentation & backdoors
│   ├── features/           # Feature extractors (4 dimensions)
│   ├── detectors/          # Detection algorithms
│   ├── evaluation/         # Metrics & experiments
│   └── utils/              # Helpers
├── experiments/            # Experiment scripts & results
├── notebooks/              # Jupyter/Colab notebooks
├── configs/                # YAML configurations
├── tests/                  # Unit tests
└── paper/                  # LaTeX paper
```

---

## Research Questions

- **RQ1:** How does BAD compare to existing defenses?
- **RQ2:** Which behavioral features are most discriminative?
- **RQ3:** Can the system generalize across TM1-TM3?
- **RQ4:** What is the real-time computational overhead?
- **RQ5:** How robust against adaptive backdoors?

---

## Target Metrics

| Metric | Target |
|--------|--------|
| TPR | > 80% |
| FPR | < 5% |
| F1 | > 0.85 |
| Latency | < 500ms |

---

## Phases

- [x] **Phase 0:** Literature Review (31 papers analyzed)
- [ ] **Phase 1:** Data Collection (Weeks 1-2)
- [ ] **Phase 2:** Feature Engineering (Week 3)
- [ ] **Phase 3:** Detection Implementation (Weeks 4-5)
- [ ] **Phase 4:** Evaluation (Week 6)
- [ ] **Phase 5:** Paper Writing (Weeks 7-8)

---

## Google Colab Integration

Notebooks are designed for Google Colab GPU training:

1. **01_data_exploration.ipynb** - Analyze collected traces
2. **02_feature_analysis.ipynb** - Visualize feature distributions
3. **03_detector_training.ipynb** - Train LSTM on GPU
4. **04_results_visualization.ipynb** - Generate paper figures
5. **05_paper_figures.ipynb** - Publication-quality plots

---

## Technology Stack

- **Agent Framework:** LangChain v0.1+
- **ML:** scikit-learn (Isolation Forest), PyTorch (LSTM)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Compute:** Local dev + Google Colab GPUs

---

## Development

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
```

### Type Checking

```bash
mypy src/
```

---

## Citation

If you use this code or approach, please cite:

```bibtex
@inproceedings{sanna2025behavioral,
  title={Behavioral Anomaly Detection for AI Agent Supply Chain Security},
  author={Sanna, Arun},
  booktitle={Proceedings of [Venue]},
  year={2025}
}

@article{base-paper,
  title={Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain},
  author={[Authors]},
  journal={arXiv preprint arXiv:2510.05159},
  year={2024}
}
```

---

## License

MIT License (will be added upon publication)

---

## Contact

Arun Sanna - [email]

**Project Documentation:** See `.claude/` folder for detailed plans and learnings
