# Medical Chat System

A medical chatbot system for answering health-related questions using NLP and machine learning.

## Overview

This project creates a chatbot that can answer medical questions. It uses transformer models and RAG (retrieval-augmented generation) to provide accurate medical information.

### What it does

- Answers medical questions using AI
- Has both transformer and RAG models
- Includes safety warnings and disclaimers
- Provides API and command-line interfaces
- Evaluates model performance

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repourl>
cd medical-chat-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test the installation:
```bash
python demo.py
```

## Quick Start

### Basic Usage

```python
from models.medical_chat_model import MedicalChatModel

# Create model
model = MedicalChatModel()
model.load_model()

# Ask a question
response = model.generate_response("What are diabetes symptoms?")
print(response)
```

### Interactive Chat

```bash
python scripts/run_chat.py --interactive
```

### API Server

```bash
python api/app.py
```

## Project Structure

```
medical-chat-system/
├── data/           # Data processing and loading
├── models/         # Model implementations
├── training/       # Training scripts
├── api/           # FastAPI server
├── config/        # Configuration files
├── examples/      # Usage examples
└── scripts/       # Utility scripts
```

## Model Architecture

### Transformer Model
- Uses BART (Bidirectional and Auto-Regressive Transformers)
- Fine-tuned on medical Q&A data
- Generates responses based on input questions

### RAG Model
- Retrieval-Augmented Generation approach
- Uses sentence transformers for embeddings
- FAISS for similarity search
- Combines retrieved context with generation

## Data Processing

The system includes:
- Text preprocessing and cleaning
- Data validation and quality checks
- Train/validation/test splitting
- Medical domain augmentation

## Training

To train the model:

```bash
python training/train.py --epochs 3 --batch-size 4
```

## Evaluation

The model is evaluated using:
- ROUGE scores for text generation
- BLEU scores for translation quality
- Medical accuracy metrics
- Response quality assessment

## Example Interactions

Here are some example Q&A pairs the system can handle:

1. **Q**: What are the symptoms of diabetes?
   **A**: Common symptoms include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, and blurred vision.

2. **Q**: How is high blood pressure treated?
   **A**: Treatment involves lifestyle changes and medications. Lifestyle changes include reducing salt intake, exercising regularly, maintaining a healthy weight, limiting alcohol, and quitting smoking.

3. **Q**: What are the warning signs of a heart attack?
   **A**: Warning signs include chest pain or discomfort, pain spreading to arms/neck/jaw, shortness of breath, nausea, lightheadedness, cold sweats, and fatigue.

## API Usage

### Single Question
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are diabetes symptoms?", "model_type": "transformer"}'
```

### Batch Questions
```bash
curl -X POST "http://localhost:8000/chat/batch" \
     -H "Content-Type: application/json" \
     -d '{"questions": ["What are diabetes symptoms?", "How is high blood pressure treated?"]}'
```

## Configuration

Edit `config/config.yaml` to modify:
- Model parameters
- Training settings
- Data processing options
- API configuration

## Performance

The model achieves:
- ROUGE-1 F1: ~0.45-0.55
- Medical accuracy: ~0.70-0.80
- Response quality: Good medical information with safety warnings

