# Configuration Guide - Multi-LLM Study

## Setup Instructions

### 1. API Credentials

Edit `llm_credentials.yaml` and replace placeholders:

#### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
```
Or update directly in the YAML file.

#### XAI (Grok)
```bash
export XAI_API_KEY="xai-..."
```

#### Vertex AI (Google Cloud)
```bash
# Option 1: Application Default Credentials (recommended)
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Option 2: Service Account Key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

Update `project_id` in the YAML:
```yaml
vertex_ai:
  project_id: "your-actual-project-id"
```

### 2. Vertex AI Model Access

Enable required models in your GCP project:

```bash
# Check available models
gcloud ai models list --region=us-central1

# Verify specific models are accessible
gcloud ai endpoints list --region=us-central1 | grep -E "gemini|claude|llama|mistral"
```

**Required Vertex AI models:**
- ✅ `gemini-pro` (already collected)
- ✅ `claude-3-opus@20240229` (already collected)
- 🔄 `meta/llama-3-405b-instruct-maas`
- 🔄 `mistralai/mistral-large@2407`
- 🔄 `kimi-chat` (verify exact model ID)
- 🔄 `deepseek-ai/deepseek-coder-33b-instruct`

### 3. Verify Setup

Run the credential verification script:

```python
python scripts/verify_llm_credentials.py
```

Expected output:
```
✅ OpenAI API: Connected (gpt-4, gpt-4o available)
✅ XAI API: Connected (grok-1 available)
✅ Vertex AI: Connected (6 models accessible)
   ✅ gemini-pro
   ✅ claude-3-opus@20240229
   ✅ meta/llama-3-405b-instruct-maas
   ✅ mistralai/mistral-large@2407
   ⚠️  kimi-chat (verify model ID)
   ✅ deepseek-ai/deepseek-coder-33b-instruct
```

## Data Collection Plan

### Phase 1: High-Priority LLMs (Week 1-2)
1. **GPT-4** (OpenAI) - 100 benign + 100 backdoor traces
2. **Grok** (XAI) - 100 benign + 100 backdoor traces
3. **Llama-4** (Vertex AI) - 100 benign + 150 backdoor (includes TM3)

### Phase 2: Specialized Models (Week 2-3)
4. **Mistral Large** (Vertex AI) - 100 benign + 150 backdoor
5. **DeepSeek Coder** (Vertex AI) - 100 benign + 150 backdoor

### Phase 3: Optional Extended Analysis (Week 3-4)
6. **GPT-4o** (OpenAI) - Compare with GPT-4
7. **Kimi Chat** (Vertex AI) - Long-context specialist

### Total Dataset Size

**Conservative (5 LLMs):**
- Benign: 500 traces (5 × 100)
- Backdoor: 650 traces (2 × 100 + 3 × 150)
- **Total: 1,150 traces**

**Full (7 LLMs):**
- Benign: 700 traces (7 × 100)
- Backdoor: 850 traces (4 × 100 + 3 × 150)
- **Total: 1,550 traces**

## Collection Commands

### Start collection for a single LLM:
```bash
python collect_traces.py --llm gpt-4 --num-benign 100 --num-backdoor 100
```

### Batch collection:
```bash
# Collect from all priority LLMs
python collect_traces.py --llms gpt-4,grok-1,llama,mistral,deepseek --batch
```

### Resume interrupted collection:
```bash
python collect_traces.py --resume --llm llama
```

## Experiments

### 1. Cross-LLM Detection Matrix
```bash
python experiments/cross_llm_matrix.py
```

Generates 28-36 pairwise train/test combinations.

### 2. Multi-LLM Training Study
```bash
python experiments/multi_llm_training.py --training-sizes 1,2,3,4,5
```

Tests performance scaling with number of LLMs in training set.

### 3. Feature Invariance Analysis
```bash
python experiments/feature_invariance.py --variance-threshold 0.10
```

Identifies features with <10% coefficient of variation across LLMs.

### 4. Architectural Clustering
```bash
python experiments/architectural_clustering.py --method hierarchical
```

Groups LLMs by behavioral similarity.

## Troubleshooting

### "Quota exceeded" errors
- OpenAI: Check rate limits at https://platform.openai.com/account/rate-limits
- Vertex AI: Request quota increase at https://console.cloud.google.com/iam-admin/quotas

### "Model not found" errors
- Verify model IDs: `gcloud ai models list --region=us-central1`
- Check model access permissions in GCP console

### Vertex AI authentication issues
```bash
# Re-authenticate
gcloud auth application-default login

# Verify project
gcloud config get project

# Check service account permissions
gcloud projects get-iam-policy PROJECT_ID
```

## Cost Estimates

**Per LLM (100 benign + 100 backdoor traces):**

| LLM | Cost/1K tokens | Avg tokens/trace | Est. cost |
|-----|----------------|------------------|-----------|
| GPT-4 | $0.03 | 500 | $3.00 |
| GPT-4o | $0.005 | 500 | $0.50 |
| Grok | $0.01 | 500 | $1.00 |
| Gemini Pro | $0.001 | 500 | $0.10 |
| Claude 3 | $0.015 | 500 | $1.50 |
| Llama-4 | $0.002 | 500 | $0.20 |
| Mistral | $0.002 | 500 | $0.20 |
| DeepSeek | $0.002 | 500 | $0.20 |

**Total for 7 LLMs**: ~$6.70 per full collection round

**For 1,550 traces (full dataset)**: ~$50-100 total

## Timeline

- **Week 1**: Setup + GPT-4, Grok, Llama-4
- **Week 2**: Mistral, DeepSeek
- **Week 3**: Optional GPT-4o, Kimi
- **Week 4**: Cross-LLM experiments
- **Week 5**: Analysis + paper updates

## Security Notes

- **Never commit API keys to git**
- Add `llm_credentials.yaml` to `.gitignore`
- Use environment variables or secret management
- Rotate keys after project completion
- Monitor API usage dashboards regularly
