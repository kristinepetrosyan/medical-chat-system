"""
Medical Chat Model Implementation using Transformers
"""

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
import logging
from typing import List, Dict, Optional
import os

class MedicalQADataset(Dataset):
    """Custom dataset class for medical Q&A data"""
    
    def __init__(self, questions: List[str], answers: List[str], tokenizer, max_input_length: int = 512, max_target_length: int = 256):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        answer = str(self.answers[idx])
        
        # Format input with medical context
        input_text = f"Medical Question: {question}"
        
        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            answer,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

class MedicalChatModel:
    """Main medical chat model class"""
    
    def __init__(self, model_name: str = "facebook/bart-base", device: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        self.logger.info(f"Using device: {self.device}")
        
    def load_model(self, model_path: Optional[str] = None):
        """Load model and tokenizer"""
        
        if model_path and os.path.exists(model_path):
            # Load fine-tuned model
            self.logger.info(f"Loading fine-tuned model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            # Load pre-trained model
            self.logger.info(f"Loading pre-trained model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
    
    def prepare_datasets(self, train_df, val_df, test_df):
        """Prepare datasets for training"""
        
        train_dataset = MedicalQADataset(
            train_df['question'].tolist(),
            train_df['answer'].tolist(),
            self.tokenizer
        )
        
        val_dataset = MedicalQADataset(
            val_df['question'].tolist(),
            val_df['answer'].tolist(),
            self.tokenizer
        )
        
        test_dataset = MedicalQADataset(
            test_df['question'].tolist(),
            test_df['answer'].tolist(),
            self.tokenizer
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset, val_dataset, output_dir: str = "./medical_chat_model", **training_kwargs):
        """Train the model"""
        
        # Default training arguments
        default_args = {
            'output_dir': output_dir,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'logging_dir': './logs',
            'logging_steps': 100,
            'evaluation_strategy': "epoch",
            'save_strategy': "epoch",
            'load_best_model_at_end': True,
            'metric_for_best_model': "eval_loss",
            'greater_is_better': False,
            'save_total_limit': 2,
            'prediction_loss_only': True,
            'dataloader_pin_memory': False,
        }
        
        # Update with user-provided arguments
        default_args.update(training_kwargs)
        
        training_args = TrainingArguments(**default_args)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        self.logger.info("Starting training...")
        trainer.train()
        
        # Save the trained model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Model saved to {output_dir}")
        
        return trainer
    
    def generate_response(self, question: str, max_length: int = 256, num_beams: int = 4, temperature: float = 0.7) -> str:
        """Generate answer for medical question"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # format the input
        input_text = f"Medical Question: {question}"
        
        # tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # clean up if needed
        if "Medical Question:" in response:
            response = response.split("Medical Question:")[-1].strip()
        
        return response
    
    def chat_interactive(self):
        """Interactive chat interface"""
        
        print("Medical Chat System")
        print("=" * 50)
        print("This is for educational purposes only.")
        print("Always consult healthcare professionals for medical advice.")
        print("=" * 50)
        print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("Your Question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("Thank you for using Medical Chat System. Stay healthy!")
                    break
                
                if not user_input:
                    print("Please enter a question.")
                    continue
                
                print("Thinking...")
                response = self.generate_response(user_input)
                print(f"Medical Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error generating response: {str(e)}")
                print("Sorry, I encountered an error. Please try again.")
    
    def batch_generate(self, questions: List[str], **generation_kwargs) -> List[str]:
        """Generate responses for multiple questions"""
        
        responses = []
        for question in questions:
            try:
                response = self.generate_response(question, **generation_kwargs)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Error generating response for question '{question}': {str(e)}")
                responses.append("I apologize, but I couldn't generate a response to this question.")
        
        return responses
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        
        if self.model is None:
            return {"status": "Model not loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 ** 2),  # Assuming float32
            "status": "Loaded and ready"
        }
