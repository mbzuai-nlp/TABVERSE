# TABVERSE: Benchmarking Cross-Format Table Understanding in LLMs and VLMs

A comprehensive benchmark for evaluating Large Language Models (LLMs) and Vision-Language Models (VLMs) on cross-format table images for Structural Understanding Capability (SUC), Question-Answering (QA), and Structure Reconstruction (SR) tasks.

## 🎯 Overview

TABVERSE evaluates models across multiple table formats (HTML, Markdown, LaTeX) and tasks including:

- **Structured Understanding and Comprehension (SUC)**: Cell value retrieval, row/column retrieval, table summarization
- **Task Prediction**: Classification and reasoning over tabular data
- **Format Generation**: Converting tables between different formats

## 📊 Datasets

TABVERSE provides a **cross-format multimodal dataset** that includes **tables represented in HTML, Markdown, and LaTeX**, along with their **corresponding rendered images**. Each table also includes the **source text files** in these formats to support both LLM and VLM evaluations.

- **Text formats**: `.html`, `.md`, `.tex`
- **Rendered images**: Aligned PNG renderings for each format
- **Coverage**: ~3.5K unique tables and 5K query–table pairs
- **Sources**: FEVEROUS, HybridQA, SQA, TabFact, ToTTo

These multimodal, cross-format representations enable consistent evaluation of models on **text**, **visual**, and **format conversion** tasks.

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/mbzuai-nlp/TABVERSE.git
cd TABVERSE
chmod +x setup.sh
./setup.sh
```

### 2. Configure API Keys

Edit the `.env` file with your actual credentials:

```bash
nano .env
```

Required environment variables:

```bash
# Hugging Face Token for dataset access
HF_TOKEN=your_huggingface_token_here

# OpenRouter API Key for GPT models
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Direct OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Evaluations

#### VLM Models (Vision-Language Models)

```bash
# Qwen VLM 2.5-3B
./src/scripts/run_qwen3b.sh 50 suc

# Qwen VLM 2.5-7B
./src/scripts/run_qwen7b.sh 100 task

# SmolVLM 1.7B
./src/scripts/run_smolvlm.sh 50 generation
```

#### LLM Models (Text-only)

```bash
# Qwen LLM 2.5-3B
./src/scripts/run_llm_qwen_llm_2.5-3B-Instruct.sh

# Qwen LLM 2.5-7B
./src/scripts/run_llm_qwen_llm_2.5-7B-Instruct.sh

# SmolLM2 1.7B
./src/scripts/run_llm_SmolLM2-1.7B-Instruct.sh
```

## 📁 Repository Structure

```
TABVERSE/
├── data/                          # Dataset files
│   ├── 1-raw/                    # Raw dataset files (.jsonl)
│   ├── 2-task/                   # Task-specific data (.json)
│   ├── 3-suc/                    # SUC task data (.json)
│   └── 4-generation_subset/      # Generation subset data
├── src/
│   ├── scripts/                  # Execution scripts
│   ├── utils/                    # Utility functions
│   └── zeroshot/                # Zero-shot evaluation code
│       ├── gpt_*.py             # GPT model evaluations
│       ├── vlm_*.py             # VLM model evaluations
│       ├── llm_*.py             # LLM model evaluations
│       └── generation*.py       # Format generation tasks
├── results/                      # Output results
│   ├── vlmpipeline/             # VLM results
│   ├── llmpipeline/             # LLM results
│   └── vlmpipeline-text/        # VLM text-only results
├── .env.template                # Environment variables template
├── requirements.txt             # Python dependencies
└── setup.sh                    # Setup script
```

## 🔧 Usage Details

### Script Parameters

Most scripts accept the following parameters:

```bash
./src/scripts/script_name.sh [max_samples] [task]
```

- `max_samples`: Number of samples to evaluate (default: 1000)
- `task`: Task type to run
  - `suc`: Structured Understanding and Comprehension
  - `task`: Task-specific prediction
  - `generation`: Format generation
  - `text_only_vlm`: VLM in text-only mode
  - `all`: Run all available tasks

### Available Tasks

#### 1. Structured Understanding and Comprehension (SUC)

- **Cell Value Retrieval**: Extract specific cell values by coordinates
- **Column Retrieval**: Get column names by index
- **Row Retrieval**: Extract entire row data
- **Table Summarization**: Generate concise table summaries

#### 2. Task Prediction

- **Binary Classification**: True/false fact verification
- **Multi-class Classification**: Category prediction
- **Question Answering**: Answer questions about table content

#### 3. Format Generation

- **HTML ↔ Markdown**: Convert between HTML and Markdown formats
- **HTML ↔ LaTeX**: Convert between HTML and LaTeX formats
- **Markdown ↔ LaTeX**: Convert between Markdown and LaTeX formats

## 📈 Results and Evaluation

Results are automatically saved in structured directories:

```
results/
├── vlmpipeline/
│   └── model_name/
│       ├── suc/              # SUC task results
│       ├── task/             # Task prediction results
│       └── generation/       # Generation results
├── llmpipeline/
│   └── model_name/
│       └── ...
└── vlmpipeline-text/
    └── model_name/
        └── ...
```

Each result file contains:

- Model predictions
- Ground truth labels
- Evaluation metrics
- Execution metadata

## 🛠️ Development

### Adding New Models

1. Create evaluation script in `src/zeroshot/`
2. Add model configuration
3. Create runner script in `src/scripts/`
4. Update documentation

### Environment Variables

The codebase uses `python-dotenv` for environment management:

- All API keys are loaded from `.env` file
- Command-line arguments can override environment variables
- Supports multiple API providers (OpenRouter, OpenAI, etc.)

### Dependencies

Key dependencies include:

- `openai`: API client for language models
- `datasets`: Hugging Face datasets
- `python-dotenv`: Environment variable management
- `PIL`: Image processing for VLMs
- `requests`: HTTP client for API calls

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or issues:

- Open an issue on GitHub
- Check existing documentation
- Review the environment setup guide

## 🏆 Citation

If you use TABVERSE in your research, please cite:

```bibtex
@misc{tabverse2024,
  title={TABVERSE: Benchmarking Cross-Format Table Understanding in LLMs and VLMs},
  author={Ahsan, Momina and Ahmad, Sarfraz and Hee, Ming Shan and Lee, Roy Ka-wei and Nakov, Preslav},
  journal={},
  year={2025},
}
```
