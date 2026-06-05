#!/bin/bash

echo "Setting up TABVERSE: Benchmarking Cross-Format Table Understanding in LLMs and VLMs"
echo "==============================================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

echo "✓ Python3 detected"

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "Error: pip is not installed. Please install pip and try again."
    exit 1
fi

echo "✓ pip detected"

# Install required Python packages
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p results/vlmpipeline
mkdir -p results/llmpipeline
mkdir -p results/vlmpipeline-text
mkdir -p logs
mkdir -p models

echo "✓ Directory structure created"

# Set up environment variables
echo ""
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "✓ Created .env file from template"
        echo ""
        echo "⚠️  IMPORTANT: Please edit the .env file and add your actual API keys:"
        echo "   - HF_TOKEN: Your Hugging Face token"
        echo "   - OPENROUTER_API_KEY: Your OpenRouter API key (for GPT models)"
        echo "   - OPENAI_API_KEY: Your OpenAI API key (optional)"
        echo ""
        echo "   You can edit the file with: nano .env"
    else
        echo "✗ .env.template not found. Creating basic .env file..."
        cat > .env << EOF
# Environment Variables for TABVERSE
# Fill in your actual API keys and tokens

# Hugging Face Token for private dataset access
HF_TOKEN=your_huggingface_token_here

# OpenRouter API Key for GPT models
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: OpenAI API Key (if using direct OpenAI API)
OPENAI_API_KEY=your_openai_api_key_here
EOF
        echo "✓ Created basic .env file"
    fi
else
    echo "✓ .env file already exists"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
find src/scripts -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
echo "✓ Scripts are now executable"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "==============================================================================="
echo "NEXT STEPS:"
echo "==============================================================================="
echo ""
echo "1. 📝 Edit your environment variables:"
echo "   nano .env"
echo ""
echo "2. 🚀 Usage examples for different models:"
echo ""
echo "   For VLM models (Vision-Language Models):"
echo "   • SmolVLM 1.7B:    ./src/scripts/run_smolvlm.sh [max_samples] [task]"
echo "   • Qwen VLM 3B:     ./src/scripts/run_qwen3b.sh [max_samples] [task]"
echo "   • Qwen VLM 7B:     ./src/scripts/run_qwen7b.sh [max_samples] [task]"
echo ""
echo "   For LLM models (Text-only):"
echo "   • SmolLM2 1.7B:    ./src/scripts/run_llm_SmolLM2-1.7B-Instruct.sh"
echo "   • Qwen LLM 3B:     ./src/scripts/run_llm_qwen_llm_2.5-3B-Instruct.sh"
echo "   • Qwen LLM 7B:     ./src/scripts/run_llm_qwen_llm_2.5-7B-Instruct.sh"
echo ""
echo "3. 📊 Available tasks:"
echo "   • suc: Structured Understanding and Comprehension"
echo "   • task: Task-specific prediction"
echo "   • generation: Format generation"
echo "   • text_only_vlm: VLM text-only mode"
echo "   • all: Run all tasks"
echo ""
echo "4. 📈 Results will be saved in:"
echo "   • results/vlmpipeline/ (for VLM models)"
echo "   • results/llmpipeline/ (for LLM models)"
echo "   • results/vlmpipeline-text/ (for VLM text-only)"
echo ""
echo "5. 📋 Datasets included:"
echo "   • FEVEROUS • HybridQA • SQA • TabFact • ToTTo"
echo ""
echo "Example: ./src/scripts/run_qwen3b.sh 50 suc"
echo "        (Run Qwen 3B VLM on 50 samples for SUC task)"
