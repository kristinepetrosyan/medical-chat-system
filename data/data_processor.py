"""

This module handles loading, cleaning, and preprocessing medical Q&A data.
I built this to handle various data formats and clean up messy medical text.
"""

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MedicalDataProcessor:
    """Data preprocessing and cleaning utilities for medical data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # added this for debugging - sometimes the data is messy
        self.debug_mode = False
        
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean up text data"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # make lowercase
        text = text.lower()
        
        # fix whitespace
        text = ' '.join(text.split())
        
        # remove weird chars but keep medical stuff
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\%]', '', text)
        
        # fix multiple dots/commas
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\,{2,}', ',', text)
        
        # add spaces after punctuation
        text = re.sub(r'\.(?=[^\s])', '. ', text)
        text = re.sub(r'\,(?=[^\s])', ', ', text)
        
        return text.strip()
    
    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        """Load and clean the medical dataset"""
        try:
            # Try different formats
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                # Assume CSV if no extension
                df = pd.read_csv(file_path, encoding='utf-8')
            
            self.logger.info(f"Dataset loaded with shape: {df.shape}")
            self.logger.info(f"Columns: {df.columns.tolist()}")
            
            # Identify question and answer columns (flexible naming)
            question_col = self._identify_column(df, ['question', 'query', 'q', 'input'])
            answer_col = self._identify_column(df, ['answer', 'response', 'a', 'output', 'reply'])
            
            if question_col is None or answer_col is None:
                # Use first two columns if naming is unclear
                question_col = df.columns[0]
                answer_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                self.logger.warning(f"Using columns: {question_col} as questions, {answer_col} as answers")
            
            # Clean the data
            df['question'] = df[question_col].apply(self.preprocess_text)
            df['answer'] = df[answer_col].apply(self.preprocess_text)
            
            # Remove rows with empty questions or answers
            initial_count = len(df)
            df = df.dropna(subset=['question', 'answer'])
            df = df[df['question'].str.len() > 10]  # Minimum question length
            df = df[df['answer'].str.len() > 20]   # Minimum answer length
            
            self.logger.info(f"Cleaned dataset: {len(df)} rows (removed {initial_count - len(df)} rows)")
            
            # Reset index
            df = df.reset_index(drop=True)
            
            return df[['question', 'answer']]
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _identify_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Identify column by keywords"""
        for col in df.columns:
            col_lower = col.lower()
            for keyword in keywords:
                if keyword in col_lower:
                    return col
        return None
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        
        # First split: train+val and test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            shuffle=True
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)  # Adjust ratio for remaining data
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=42,
            shuffle=True
        )
        
        self.logger.info(f"Data split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    
    def augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment dataset with paraphrased questions and additional medical Q&A"""
        
        # Add some common medical Q&A pairs for augmentation
        additional_qa = [
            {
                'question': 'what are the symptoms of high blood pressure',
                'answer': 'high blood pressure often has no symptoms, which is why it is called the silent killer. some people may experience headaches, shortness of breath, or nosebleeds, but these symptoms are not specific and usually occur when blood pressure reaches severe or life-threatening levels.'
            },
            {
                'question': 'how can i prevent heart disease',
                'answer': 'heart disease prevention includes maintaining a healthy diet rich in fruits and vegetables, exercising regularly, avoiding smoking, limiting alcohol consumption, managing stress, maintaining a healthy weight, and controlling conditions like diabetes, high blood pressure, and high cholesterol.'
            },
            {
                'question': 'what causes diabetes type 2',
                'answer': 'type 2 diabetes is caused by a combination of genetic and lifestyle factors. risk factors include being overweight, physical inactivity, family history of diabetes, age over 45, high blood pressure, and having prediabetes.'
            },
            {
                'question': 'when should i see a doctor for chest pain',
                'answer': 'seek immediate medical attention for chest pain if it is severe, crushing, or squeezing; radiates to your arm, jaw, or back; is accompanied by shortness of breath, nausea, or sweating; or if you have risk factors for heart disease. any new or worsening chest pain should be evaluated by a healthcare provider.'
            },
            {
                'question': 'what are the side effects of blood pressure medication',
                'answer': 'common side effects of blood pressure medications may include dizziness, fatigue, headache, cough (with ace inhibitors), swelling in legs or ankles, and changes in heart rate. specific side effects vary by medication type. always discuss concerns with your healthcare provider.'
            }
        ]
        
        aug_df = pd.DataFrame(additional_qa)
        combined_df = pd.concat([df, aug_df], ignore_index=True)
        
        self.logger.info(f"Augmented dataset: {len(df)} -> {len(combined_df)} samples")
        
        return combined_df
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> dict:
        """Get comprehensive statistics about the dataset"""
        
        stats = {
            'total_samples': len(df),
            'avg_question_length': df['question'].str.len().mean(),
            'avg_answer_length': df['answer'].str.len().mean(),
            'min_question_length': df['question'].str.len().min(),
            'max_question_length': df['question'].str.len().max(),
            'min_answer_length': df['answer'].str.len().min(),
            'max_answer_length': df['answer'].str.len().max(),
            'avg_question_words': df['question'].str.split().str.len().mean(),
            'avg_answer_words': df['answer'].str.split().str.len().mean(),
        }
        
        return stats
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality and integrity"""
        
        issues = []
        
        # Check for missing values
        if df['question'].isna().any() or df['answer'].isna().any():
            issues.append("Missing values found")
        
        # Check for very short questions/answers
        short_questions = (df['question'].str.len() < 10).sum()
        short_answers = (df['answer'].str.len() < 20).sum()
        
        if short_questions > 0:
            issues.append(f"{short_questions} questions are too short")
        if short_answers > 0:
            issues.append(f"{short_answers} answers are too short")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['question']).sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate questions found")
        
        if issues:
            self.logger.warning(f"Data quality issues: {'; '.join(issues)}")
            return False
        
        self.logger.info("Data quality validation passed")
        return True
