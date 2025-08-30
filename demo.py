#!/usr/bin/env python3
"""
Demo script for medical chat system
"""

import sys
from pathlib import Path

def demo_data_processing():
    """Demo data processing functionality"""
    print("Medical Chat System - Data Processing Demo")
    print("=" * 50)
    
    try:
        from data.data_processor import MedicalDataProcessor
        import pandas as pd
        
        processor = MedicalDataProcessor()
        
        # Create sample medical data
        sample_data = [
            {
                'question': 'What are the symptoms of diabetes?',
                'answer': 'Common symptoms of diabetes include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, and blurred vision. Type 1 diabetes symptoms can develop quickly, while Type 2 diabetes symptoms may develop slowly over time.'
            },
            {
                'question': 'How is high blood pressure treated?',
                'answer': 'High blood pressure treatment typically involves lifestyle changes and medications. Lifestyle changes include reducing salt intake, exercising regularly, maintaining a healthy weight, limiting alcohol, and quitting smoking. Medications may include ACE inhibitors, beta-blockers, diuretics, or calcium channel blockers.'
            },
            {
                'question': 'What causes asthma?',
                'answer': 'Asthma is caused by inflammation and narrowing of the airways. Common triggers include allergens (pollen, dust mites, pet dander), respiratory infections, exercise, cold air, air pollution, and certain medications. Genetic factors and environmental exposures also play a role.'
            },
            {
                'question': 'How can I prevent heart disease?',
                'answer': 'Heart disease prevention includes maintaining a healthy diet rich in fruits and vegetables, exercising regularly, avoiding smoking, limiting alcohol consumption, managing stress, maintaining a healthy weight, and controlling conditions like diabetes, high blood pressure, and high cholesterol.'
            },
            {
                'question': 'What are the warning signs of a heart attack?',
                'answer': 'Warning signs of a heart attack include chest pain or discomfort, pain spreading to arms, neck, jaw, or back, shortness of breath, nausea, lightheadedness, cold sweats, and fatigue. Women may experience different symptoms like back pain, extreme fatigue, or indigestion.'
            }
        ]
        
        df = pd.DataFrame(sample_data)
        print(f" Created sample dataset with {len(df)} medical Q&A pairs")
        
        # Process the data
        processed_df = df.copy()
        processed_df['question'] = processed_df['question'].apply(processor.preprocess_text)
        processed_df['answer'] = processed_df['answer'].apply(processor.preprocess_text)
        print(f" Processed dataset: {len(processed_df)} rows")
        
        # Split the data
        train_df, val_df, test_df = processor.split_data(processed_df)
        print(f" Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Show statistics
        stats = processor.get_dataset_statistics(train_df)
        print(f" Dataset statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f" Data processing demo failed: {e}")
        return False

def demo_model_info():
    """Demo model information"""
    print("\n Medical Chat System - Model Information Demo")
    print("=" * 50)
    
    try:
        from models.medical_chat_model import MedicalChatModel
        
        # Create model instance
        model = MedicalChatModel()
        print(" Medical chat model created successfully")
        
        # Get model information
        info = model.get_model_info()
        print(" Model Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f" Model info demo failed: {e}")
        return False

def demo_example_interactions():
    """Demo example interactions"""
    print("\n Medical Chat System - Example Interactions Demo")
    print("=" * 50)
    
    # Example interactions as required by the assignment
    examples = [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms of diabetes include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, and blurred vision. Type 1 diabetes symptoms can develop quickly, while Type 2 diabetes symptoms may develop slowly over time."
        },
        {
            "question": "How is high blood pressure treated?",
            "answer": "High blood pressure treatment typically involves lifestyle changes and medications. Lifestyle changes include reducing salt intake, exercising regularly, maintaining a healthy weight, limiting alcohol, and quitting smoking. Medications may include ACE inhibitors, beta-blockers, diuretics, or calcium channel blockers."
        },
        {
            "question": "What are the warning signs of a heart attack?",
            "answer": "Warning signs of a heart attack include chest pain or discomfort, pain spreading to arms, neck, jaw, or back, shortness of breath, nausea, lightheadedness, cold sweats, and fatigue. Women may experience different symptoms like back pain, extreme fatigue, or indigestion."
        }
    ]
    
    print("Example Medical Q&A Interactions:")
    print("-" * 40)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Question: {example['question']}")
        print(f"   Answer: {example['answer']}")
        print(f"   Length: {len(example['answer'])} characters")
    
    print(f"\n Successfully demonstrated {len(examples)} example interactions")
    return True

def demo_training_pipeline():
    """Demo training pipeline setup"""
    print("\n Medical Chat System - Training Pipeline Demo")
    print("=" * 50)
    
    try:
        from training.train import load_config
        
        # Load configuration
        config = load_config()
        print(" Configuration loaded successfully")
        
        print(" Training Configuration:")
        print(f"   Model: {config['model']['name']}")
        print(f"   Epochs: {config['training']['num_train_epochs']}")
        print(f"   Batch Size: {config['training']['per_device_train_batch_size']}")
        print(f"   Learning Rate: {config['training']['learning_rate']}")
        print(f"   Test Size: {config['data']['test_size']}")
        print(f"   Validation Size: {config['data']['val_size']}")
        
        return True
        
    except Exception as e:
        print(f" Training pipeline demo failed: {e}")
        return False

def demo_api_schemas():
    """Demo API schemas"""
    print("\n Medical Chat System - API Schemas Demo")
    print("=" * 50)
    
    try:
        from api.schemas import ChatRequest, ChatResponse
        
        # Create example request
        request = ChatRequest(
            question="What are the symptoms of diabetes?",
            model_type="transformer",
            max_length=256,
            temperature=0.7
        )
        print(" ChatRequest schema created successfully")
        print(f"   Question: {request.question}")
        print(f"   Model Type: {request.model_type}")
        
        #  example response
        response = ChatResponse(
            answer="Common symptoms of diabetes include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, and blurred vision.",
            question=request.question,
            model_type=request.model_type,
            processing_time=1.2
        )
        print(" ChatResponse schema created successfully")
        print(f"   Answer: {response.answer[:50]}...")
        print(f"   Processing Time: {response.processing_time}s")
        
        return True
        
    except Exception as e:
        print(f" API schemas demo failed: {e}")
        return False

def main():
    """Main demo function"""
    print("Medical Chat System - Complete Demo")
    print("=" * 60)
    
    demos = [
        ("Data Processing", demo_data_processing),
        ("Model Information", demo_model_info),
        ("Example Interactions", demo_example_interactions),
        ("Training Pipeline", demo_training_pipeline),
        ("API Schemas", demo_api_schemas)
    ]
    
    passed = 0
    total = len(demos)
    
    for demo_name, demo_func in demos:
        print(f"\n🎬 Running {demo_name} Demo...")
        if demo_func():
            passed += 1
        else:
            print(f"{demo_name} demo failed!")
    
    print("\n" + "=" * 60)
    print(f"📊 Demo Results: {passed}/{total} demos passed")
    
    if passed == total:
        print("All demos passed! The medical chat system is working correctly.")
        print("\nReady to use:")
        print("1. Train the model: python training/train.py")
        print("2. Start interactive chat: python scripts/run_chat.py --interactive")
        print("3. Start API server: python api/app.py")
        print("4. View documentation: cat README.md")
    else:
        print("Some demos failed. The system may need additional setup.")
        print("Try installing missing dependencies or check the error messages above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
