"""
Example interactions for the medical chat system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.medical_chat_model import MedicalChatModel
from models.rag_model import RAGMedicalModel

def example_transformer_interactions():
    """Example interactions using transformer model"""
    
    print("Medical Chat System Transformer Model Examples")
    print("=" * 60)
    
    # Initialize model
    model = MedicalChatModel()
    model.load_model()
    
    # Example questions
    example_questions = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "What causes asthma?",
        "How can I prevent heart disease?",
        "What are the warning signs of a heart attack?"
    ]
    
    print("\nExample Interactions:")
    print("-" * 40)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("   Generating response...")
        
        try:
            response = model.generate_response(question)
            print(f"   Answer: {response}")
        except Exception as e:
            print(f"   Error: {str(e)}")
    
    print("\n" + "=" * 60)

def example_rag_interactions():
    """Example interactions using RAG model"""
    
    print("Medical Chat System RAG Model Examples")
    print("=" * 60)
    
    # Initialize model
    model = RAGMedicalModel()
    model.load_models()
    
    # Create sample knowledge base
    from data.download_data import create_sample_dataset
    sample_path = create_sample_dataset("data/rag_example_kb.csv")
    import pandas as pd
    knowledge_base = pd.read_csv(sample_path)
    model.build_knowledge_base(knowledge_base)
    
    # Example questions
    example_questions = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "What causes asthma?",
        "How can I prevent heart disease?",
        "What are the warning signs of a heart attack?"
    ]
    
    print("\nExample Interactions:")
    print("-" * 40)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("   Retrieving relevant information...")
        
        try:
            # Show retrieved context
            contexts = model.retrieve_relevant_context(question, k=2)
            if contexts:
                print("   Retrieved contexts:")
                for j, context in enumerate(contexts, 1):
                    print(f"     {j}. Similarity: {context['similarity_score']:.3f}")
                    print(f"        {context['question'][:50]}...")
            
            print("   Generating response...")
            response = model.generate_response(question)
            print(f"   Answer: {response}")
            
        except Exception as e:
            print(f"   Error: {str(e)}")
    
    print("\n" + "=" * 60)

def example_api_usage():
    """Example API usage"""
    
    print("🏥 Medical Chat System - API Usage Examples")
    print("=" * 60)
    
    import requests
    import json
    
    # API base URL
    base_url = "http://localhost:8000"
    
    # Example API calls
    examples = [
        {
            "endpoint": "/chat",
            "method": "POST",
            "data": {
                "question": "What are the symptoms of diabetes?",
                "model_type": "transformer",
                "max_length": 256,
                "temperature": 0.7
            }
        },
        {
            "endpoint": "/chat/batch",
            "method": "POST",
            "data": {
                "questions": [
                    "What are the symptoms of diabetes?",
                    "How is high blood pressure treated?",
                    "What causes asthma?"
                ],
                "model_type": "transformer"
            }
        },
        {
            "endpoint": "/health",
            "method": "GET",
            "data": None
        },
        {
            "endpoint": "/model-info",
            "method": "GET",
            "data": None
        }
    ]
    
    print("\nAPI Endpoint Examples:")
    print("-" * 40)
    
    for example in examples:
        print(f"\n{example['method']} {example['endpoint']}")
        
        if example['data']:
            print(f"Request Body: {json.dumps(example['data'], indent=2)}")
        
        print("Response:")
        try:
            if example['method'] == 'GET':
                response = requests.get(f"{base_url}{example['endpoint']}")
            else:
                response = requests.post(
                    f"{base_url}{example['endpoint']}", 
                    json=example['data']
                )
            
            if response.status_code == 200:
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to API server. Make sure the server is running.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 60)

def interactive_chat_demo():
    """Interactive chat demonstration"""
    
    print("🏥 Medical Chat System - Interactive Demo")
    print("=" * 60)
    print("This demo will start an interactive chat session.")
    print("Type 'quit' to exit the chat.")
    print("=" * 60)
    
    # Ask user to choose model type
    model_type = input("\nChoose model type (transformer/rag): ").lower().strip()
    
    if model_type == "rag":
        print("Initializing RAG model...")
        model = RAGMedicalModel()
        model.load_models()
        
        # Load knowledge base
        from data.download_data import create_sample_dataset
        sample_path = create_sample_dataset("data/interactive_kb.csv")
        import pandas as pd
        knowledge_base = pd.read_csv(sample_path)
        model.build_knowledge_base(knowledge_base)
        
        print("RAG model ready!")
        model.chat_interactive()
        
    else:
        print("Initializing Transformer model...")
        model = MedicalChatModel()
        model.load_model()
        
        print("Transformer model ready!")
        model.chat_interactive()

def main():
    """Main function to run examples"""
    
    print("Medical Chat System - Example Interactions")
    print("=" * 60)
    print("Choose an example to run:")
    print("1. Transformer model interactions")
    print("2. RAG model interactions")
    print("3. API usage examples")
    print("4. Interactive chat demo")
    print("5. Run all examples")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-5): ").strip()
    
    if choice == "1":
        example_transformer_interactions()
    elif choice == "2":
        example_rag_interactions()
    elif choice == "3":
        example_api_usage()
    elif choice == "4":
        interactive_chat_demo()
    elif choice == "5":
        example_transformer_interactions()
        example_rag_interactions()
        example_api_usage()
        print("\nWould you like to try the interactive demo? (y/n): ")
        if input().lower().strip() == 'y':
            interactive_chat_demo()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
