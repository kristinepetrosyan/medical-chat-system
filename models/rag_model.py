"""
Retrieval-Augmented Generation (RAG) Model for Medical Chat System
"""

import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd

class RAGMedicalModel:
    """Retrieval-Augmented Generation model for medical Q&A"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 generator_model: str = "facebook/bart-base",
                 device: str = None):
        
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.embedding_model_name = embedding_model
        self.generator_model_name = generator_model
        
        # Initialize models
        self.embedder = None
        self.generator = None
        self.tokenizer = None
        self.index = None
        self.knowledge_base = None
        
        self.logger.info(f"Using device: {self.device}")
    
    def load_models(self):
        """Load embedding and generator models"""
        
        try:
            # Load sentence transformer for embeddings
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(self.embedding_model_name)
            
            # Load generator model
            self.logger.info(f"Loading generator model: {self.generator_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)
            self.generator.to(self.device)
            self.generator.eval()
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def build_knowledge_base(self, medical_data: pd.DataFrame):
        """Build knowledge base from medical data"""
        
        if self.embedder is None:
            self.load_models()
        
        self.logger.info("Building knowledge base...")
        
        # Prepare knowledge base
        self.knowledge_base = []
        
        for _, row in medical_data.iterrows():
            # Combine question and answer for better retrieval
            combined_text = f"Question: {row['question']} Answer: {row['answer']}"
            self.knowledge_base.append({
                'question': row['question'],
                'answer': row['answer'],
                'combined_text': combined_text
            })
        
        # Create embeddings
        texts = [item['combined_text'] for item in self.knowledge_base]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.logger.info(f"Knowledge base built with {len(self.knowledge_base)} entries")
    
    def retrieve_relevant_context(self, question: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant context for a given question"""
        
        if self.index is None or self.embedder is None:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")
        
        # Encode question
        question_embedding = self.embedder.encode([question])
        
        # Search for similar documents
        distances, indices = self.index.search(question_embedding.astype('float32'), k)
        
        # Retrieve relevant contexts
        relevant_contexts = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.knowledge_base):
                context = self.knowledge_base[idx].copy()
                context['similarity_score'] = 1.0 / (1.0 + distance)  # Convert distance to similarity
                relevant_contexts.append(context)
        
        return relevant_contexts
    
    def generate_response(self, question: str, max_length: int = 256, num_beams: int = 4) -> str:
        """Generate response using RAG approach"""
        
        if self.generator is None or self.tokenizer is None:
            raise ValueError("Generator model not loaded. Call load_models() first.")
        
        # Retrieve relevant context
        relevant_contexts = self.retrieve_relevant_context(question, k=3)
        
        if not relevant_contexts:
            return "I apologize, but I don't have enough information to answer this question."
        
        # Build context from retrieved documents
        context_text = ""
        for i, context in enumerate(relevant_contexts, 1):
            context_text += f"Context {i}: {context['combined_text']}\n"
        
        # Create input with context
        input_text = f"Context: {context_text}\nQuestion: {question}\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.generator.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    
    def chat_interactive(self):
        """Interactive chat interface for RAG model"""
        
        print("Medical RAG Chat System")
        print("=" * 50)
        print("This is for educational purposes only.")
        print("   Always consult healthcare professionals for medical advice.")
        print("=" * 50)
        print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("Your Question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("Thank you for using Medical RAG Chat System. Stay healthy!")
                    break
                
                if not user_input:
                    print("Please enter a question.")
                    continue
                
                print("Retrieving relevant information...")
                
                # Retrieve context
                relevant_contexts = self.retrieve_relevant_context(user_input, k=2)
                
                if relevant_contexts:
                    print("Found relevant information:")
                    for i, context in enumerate(relevant_contexts, 1):
                        print(f"   {i}. Similarity: {context['similarity_score']:.3f}")
                
                print("Generating response...")
                response = self.generate_response(user_input)
                print(f"Medical Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error generating response: {str(e)}")
                print("Sorry, I encountered an error. Please try again.")
    
    def evaluate_retrieval(self, test_questions: List[str], expected_answers: List[str]) -> Dict:
        """Evaluate retrieval quality"""
        
        if self.index is None:
            raise ValueError("Knowledge base not built.")
        
        retrieval_scores = []
        
        for question in test_questions:
            # Retrieve contexts
            contexts = self.retrieve_relevant_context(question, k=3)
            
            # Calculate retrieval score based on similarity
            if contexts:
                avg_similarity = np.mean([ctx['similarity_score'] for ctx in contexts])
                retrieval_scores.append(avg_similarity)
            else:
                retrieval_scores.append(0.0)
        
        return {
            'avg_retrieval_score': np.mean(retrieval_scores),
            'min_retrieval_score': np.min(retrieval_scores),
            'max_retrieval_score': np.max(retrieval_scores),
            'retrieval_scores': retrieval_scores
        }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        
        info = {
            "embedding_model": self.embedding_model_name,
            "generator_model": self.generator_model_name,
            "device": str(self.device),
            "knowledge_base_size": len(self.knowledge_base) if self.knowledge_base else 0,
            "index_built": self.index is not None,
            "models_loaded": self.embedder is not None and self.generator is not None
        }
        
        if self.generator is not None:
            total_params = sum(p.numel() for p in self.generator.parameters())
            info["generator_parameters"] = total_params
            info["generator_size_mb"] = total_params * 4 / (1024 ** 2)
        
        return info
