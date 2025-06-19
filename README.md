# LaRS: Latent Reasoning Skills for Chain-of-Thought Reasoning
The code for our EMNLP 2024 Paper: [LaRS: Latent Reasoning Skills for Chain-of-Thought Reasoning](https://arxiv.org/abs/2312.04684).

## Setup Environment
```
conda create -n rsd python=3.10
conda activate rsd
pip install -r requirements.txt
```

## Download Datasets
Download the preprocessed datasets from [Google Drive](https://drive.google.com/file/d/1fmL3KMoFwGqwO3A9R8fRrB22aBz4ZPHp/view?usp=sharing). The archive contains the following datasets:
- **TableMWP**: Mathematical word problems with tabular data
- **GSM8K**: Grade school math problems
- **COGS**: Compositional generalization tasks

After downloading, extract the archive in the parent directory. The resulting file structure should appear as follows:

```
assets/
└── dataset/
    ├── tablemwp/
    ├── grade-school-math/
    └── COGS/
```

## Setup Endpoint

### Quick Testing with OpenAI
For quick testing, you can use the OpenAI endpoint by setting `--endpoint gpt-3.5-turbo`. This requires setting your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

### Adding Custom Endpoints
To integrate your own LLM endpoint:

1. **Create an endpoint class** similar to `src/llm/openai.py` that inherits from the base class `BaseLLM`
2. **Implement required methods**:
   - `__init__`: Initialize your endpoint connection
   - `predict`: Handle the actual LLM inference
3. **Register your endpoint** by adding it to `src/llm/__init__.py` with your chosen endpoint name

Example usage with a custom endpoint on TableMWP:
```bash
python scripts/eval.py --task tabmwp --split test1k --endpoint <YOUR_ENDPOINT_NAME> --selection random --seed 11 --verbose
```

## Add your own endpoint
1. Create an endpoint class similarly to `src.llm.openai.py` that inherits the base class `BaseLLM`.
2. Implement `__init__` and `predict` methods accordingly.
3. Register the endpoint class with <YOUR_ENDPOINT_NAME> in `src/llm/__init__.py`.

## Experiments

### Training the LaRS Skill Encoder
Train the skill encoder model using the following command:

```bash
python scripts/train_skill_discovery.py --task tabmwp --split train_cand1k --num_epoch 2000
```

The trained model weights will be saved to `results/train_skill_discovery.py/<TIMESTAMP>-<TASK_NAME>/model.pt`. An example of the trained skill embedding can be found [here](results/train_skill_discovery.py/example-tabmwp).

### Evaluating the Skill Encoder
To visualize and evaluate the trained skill encoder:

```bash
python scripts/train_skill_discovery.py --task tabmwp --split train_cand1k --test --load <PATH_TO_MODEL>
```

This will display skill discovery metrics and visualizations for the trained encoder.

### Running LaRS and Baseline Evaluations

#### Random Selection Baseline
```bash
# Run multiple seeds for statistical significance
python scripts/eval.py --task tabmwp --split test1k --endpoint <YOUR_ENDPOINT_NAME> --selection random --seed 11 --verbose
```

#### Question-based Retrieval Baseline
```bash
python scripts/eval.py --task tabmwp --split test1k --endpoint <YOUR_ENDPOINT_NAME> --selection retrieval_q --seed 11 --verbose
```

#### LaRS Method (Retrieval with Skill Discovery)
```bash
python scripts/eval.py --task tabmwp --split test1k --endpoint <YOUR_ENDPOINT_NAME> --selection retrieval_rsd --seed 11 --verbose --skill_encoder <PATH_TO_YOUR_SKILL_ENCODER>
```

**Note**: Replace `<PATH_TO_YOUR_SKILL_ENCODER>` with the path to your trained skill encoder model.
