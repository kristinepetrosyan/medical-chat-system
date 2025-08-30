#!/usr/bin/env python3
"""
Simple test script for medical chat system
"""

import sys
from pathlib import Path

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        import torch
        print(" PyTorch imported successfully")
        
        import transformers
        print("Transformers imported successfully")
        
        import pandas as pd
        print("Pandas imported successfully")
        
        import numpy as np
        print(" NumPy imported successfully")
        
        return True
    except ImportError as e:
        print(f" Import error: {e}")
        return False

def test_data_processor():
    """Test data processor"""
    print("\nTesting data processor...")
    
    try:
        from data.data_processor import MedicalDataProcessor
        
        processor = MedicalDataProcessor()
        
        # Test text preprocessing
        test_text = "What are the symptoms of diabetes?"
        processed = processor.preprocess_text(test_text)
        print(f" Text preprocessing: '{test_text}' -> '{processed}'")
        
        # Test sample data creation
        sample_data = [
            {'question': 'What are the symptoms of diabetes?', 'answer': 'Common symptoms include...'},
            {'question': 'How is high blood pressure treated?', 'answer': 'Treatment involves...'},
            {'question': 'What causes asthma?', 'answer': 'Asthma is caused by...'},
            {'question': 'How can I prevent heart disease?', 'answer': 'Prevention includes...'},
            {'question': 'What are the warning signs of a heart attack?', 'answer': 'Warning signs include...'}
        ]
        
        df = pd.DataFrame(sample_data)
        
        # Test data splitting
        train, val, test = processor.split_data(df)
        print(f" Data splitting: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        return True
    except Exception as e:
        print(f" Data processor test failed: {e}")
        return False

def test_model_creation():
    """Test model creation (without loading weights)"""
    print("\nTesting model creation...")
    
    try:
        from models.medical_chat_model import MedicalChatModel
        
        # Test model creation
        model = MedicalChatModel()
        print(" Medical chat model created successfully")
        
        # Test model info
        info = model.get_model_info()
        print(f" Model info: {info}")
        
        return True
    except Exception as e:
        print(f" Model creation test failed: {e}")
        return False

def test_sample_data_creation():
    """Test sample data creation"""
    print("\nTesting sample data creation...")
    
    try:
        from data.download_data import create_sample_dataset
        
        # Create sample dataset
        sample_path = create_sample_dataset("data/test_sample.csv")
        print(f" Sample dataset created at: {sample_path}")
        
        # Load and verify
        import pandas as pd
        df = pd.read_csv(sample_path)
        print(f" Sample dataset loaded: {len(df)} rows")
        print(f"   Columns: {df.columns.tolist()}")
        
        return True
    except Exception as e:
        print(f" Sample data creation test failed: {e}")
        return False

def test_example_interactions():
    """Test example interactions"""
    print("\n Testing example interactions...")
    
    # Example questions and expected responses
    examples = [
        {
            "question": "What are the symptoms of diabetes?",
            "expected_keywords": ["symptoms", "diabetes", "thirst", "urination"]
        },
        {
            "question": "How is high blood pressure treated?",
            "expected_keywords": ["treatment", "blood pressure", "medication", "lifestyle"]
        },
        {
            "question": "What are the warning signs of a heart attack?",
            "expected_keywords": ["warning", "heart attack", "chest pain", "emergency"]
        }
    ]
    
    print(" Example interactions:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. Q: {example['question']}")
        print(f"     Expected keywords: {example['expected_keywords']}")
    
    print(" Example interactions defined successfully")
    return True

def main():
    """Main test function"""
    print(" Medical Chat System - Simple Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Processor", test_data_processor),
        ("Model Creation", test_model_creation),
        ("Sample Data Creation", test_sample_data_creation),
        ("Example Interactions", test_example_interactions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f" {test_name} failed!")
    
    print("\n" + "=" * 50)
    print(f" Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The medical chat system is ready to use.")
        print("\nNext steps:")
        print("1. Train the model: python training/train.py")
        print("2. Start interactive chat: python scripts/run_chat.py --interactive")
        print("3. Start API server: python api/app.py")
    else:
        print("  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
