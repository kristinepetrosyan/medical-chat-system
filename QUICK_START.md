# Medical Chat System - Quick Start Guide


### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Test the Setup

```bash
# Run the test script to verify everything works
python test_setup.py
```

### 3. Create Sample Data

```bash
# The system will automatically create sample medical data
python data/download_data.py
```

### 4. Train the Model

```bash
# Train with default settings (3 epochs)
python training/train.py

# Or train with custom settings
python training/train.py --epochs 5 --batch-size 8
```

### 5. Start Chatting!

```bash
# Interactive chat session
python scripts/run_chat.py --interactive

# Or ask a single question
python scripts/run_chat.py --question "What are the symptoms of diabetes?"
```

## API Usage

### Start the API Server

```bash
# Start FastAPI server
python api/app.py

# Or use uvicorn
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### API Examples

```python
import requests

# Single chat request
response = requests.post("http://localhost:8000/chat", json={
    "question": "What are the symptoms of diabetes?",
    "model_type": "transformer"
})

print(response.json())

# Batch request
response = requests.post("http://localhost:8000/chat/batch", json={
    "questions": [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?"
    ]
})

print(response.json())
```

## Example 

### Example 1: Diabetes Symptoms
**Q**: "What are the symptoms of diabetes?"
**A**: "Common symptoms of diabetes include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, and blurred vision."

### Example 2: High Blood Pressure Treatment
**Q**: "How is high blood pressure treated?"
**A**: "High blood pressure treatment typically involves lifestyle changes and medications. Lifestyle changes include reducing salt intake, exercising regularly, maintaining a healthy weight, limiting alcohol, and quitting smoking."

### Example 3: Heart Attack Warning Signs
**Q**: "What are the warning signs of a heart attack?"
**A**: "Warning signs of a heart attack include chest pain or discomfort, pain spreading to arms, neck, jaw, or back, shortness of breath, nausea, lightheadedness, cold sweats, and fatigue."


### Common Issues

1. **Import Errors**: Make sure you've installed all dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA Errors**: The system works on CPU, but for GPU acceleration:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Model Loading Issues**: Check if the model path is correct
   ```bash
   python scripts/run_chat.py --model-path ./medical_chat_model
   ```

4. **API Connection Issues**: Make sure the server is running
   ```bash
   python api/app.py
   ```

- Check the full README.md
- Run `python test_setup.py` to diagnose issues
- Check the logs in the `logs/` directory
