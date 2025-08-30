#!/usr/bin/env python3
"""
Test script to verify medical chat system setup is working
"""

import sys
from pathlib import Path

def test_imports():
    """Test if modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test data module
        from data.data_processor import MedicalDataProcessor
        print("Data processor imported successfully")
        
        # Test models module
        from models.medical_chat_model import MedicalChatModel
        print("Medical chat model imported successfully")
        
        from models.rag_model import RAGMedicalModel
        print("RAG model imported successfully")
        
        # Test training module
        from training.train import load_config
        print("Training module imported successfully")
        
        # Test API module
        from api.schemas import ChatRequest, ChatResponse
        print("API schemas imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_data_processor():
    """Test data processor functionality"""
    print("\n Testing data processor...")
    
    try:
        from data.data_processor import MedicalDataProcessor
        
        processor = MedicalDataProcessor()
        
        # Test text preprocessing
        test_text = "What are the symptoms of diabetes?"
        processed = processor.preprocess_text(test_text)
        print(f"Text preprocessing: '{test_text}' -> '{processed}'")
        
        # Test sample data creation
        sample_data = [
            {'question': 'What are the symptoms of diabetes?', 'answer': 'Common symptoms include...'},
            {'question': 'How is high blood pressure treated?', 'answer': 'Treatment involves...'}
        ]
        
        import pandas as pd
        df = pd.DataFrame(sample_data)
        
        # Test data splitting
        train, val, test = processor.split_data(df)
        print(f"Data splitting: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        return True
        
    except Exception as e:
        print(f"Data processor test failed: {e}")
        return False

def test_model_initialization():
    """Test model initialization"""
    print("\n Testing model initialization...")
    
    try:
        from models.medical_chat_model import MedicalChatModel
        
        # Test model creation (without loading weights)
        model = MedicalChatModel()
        print("Medical chat model created successfully")
        
        # Test model info
        info = model.get_model_info()
        print(f"Model info: {info}")
        
        return True
        
    except Exception as e:
        print(f"Model initialization test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from training.train import load_config
        
        config = load_config()
        print("Configuration loaded successfully")
        print(f"   Model: {config['model']['name']}")
        print(f"   Epochs: {config['training']['num_train_epochs']}")
        
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Medical Chat System - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Processor Test", test_data_processor),
        ("Model Initialization Test", test_model_initialization),
        ("Configuration Test", test_config_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"{test_name} failed!")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The medical chat system is ready to use.")
        print("\nNext steps:")
        print("1. Run training: python training/train.py")
        print("2. Start interactive chat: python scripts/run_chat.py --interactive")
        print("3. Start API server: python api/app.py")
    else:
        print(" Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
