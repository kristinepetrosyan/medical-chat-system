#!/usr/bin/env python3
"""
Main chat script for medical chat system
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.medical_chat_model import MedicalChatModel
from models.rag_model import RAGMedicalModel

def main():
    """Main function for running the medical chat system"""
    
    parser = argparse.ArgumentParser(description="Medical Chat System")
    parser.add_argument("--model-type", type=str, default="transformer", 
                       choices=["transformer", "rag"], 
                       help="Type of model to use")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--interactive", action="store_true", 
                       help="Start interactive chat session")
    parser.add_argument("--question", type=str, help="Single question to ask")
    
    args = parser.parse_args()
    
    try:
        if args.model_type == "transformer":
            print("🏥 Initializing Transformer Medical Chat Model...")
            model = MedicalChatModel()
            model.load_model(args.model_path)
        else:
            print("🏥 Initializing RAG Medical Chat Model...")
            model = RAGMedicalModel()
            model.load_models()
            
            # Load knowledge base if available
            if args.model_path:
                import pandas as pd
                knowledge_base_path = Path(args.model_path) / "knowledge_base.csv"
                if knowledge_base_path.exists():
                    knowledge_base = pd.read_csv(knowledge_base_path)
                    model.build_knowledge_base(knowledge_base)
        
        print("Model loaded successfully!")
        
        if args.interactive:
            print("\nStarting interactive chat session...")
            print("Type 'quit', 'exit', or 'bye' to end the conversation.")
            print("=" * 60)
            model.chat_interactive()
        elif args.question:
            print(f"\nQuestion: {args.question}")
            print("Generating response...")
            response = model.generate_response(args.question)
            print(f"Answer: {response}")
        else:
            print("\nNo action specified. Use --interactive for chat session or --question for single query.")
            print("Example: python scripts/run_chat.py --interactive")
            print("Example: python scripts/run_chat.py --question 'What are the symptoms of diabetes?'")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
