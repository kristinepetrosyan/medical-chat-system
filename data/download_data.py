"""
Data download utilities for medical chat system
"""

import os
import gdown
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

def download_medical_dataset(url: str = None, output_path: str = None) -> str:
    """
    Download medical dataset from Google Drive or create sample data
    
    Args:
        url: Google Drive URL for the dataset
        output_path: Path to save the downloaded file
    
    Returns:
        Path to the downloaded/created dataset
    """
    
    logger = logging.getLogger(__name__)
    
    # Default Google Drive URL from the assignment
    if url is None:
        url = "https://drive.google.com/file/d/1upzfj8bXP012zZsq01jcoeO9NyhmTHnQ/view?usp=drive_link"
    
    # Default output path
    if output_path is None:
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / "medical_dataset.csv"
    
    try:
        logger.info(f"Attempting to download dataset from: {url}")
        
        # Download from Google Drive
        gdown.download(url, str(output_path), fuzzy=True)
        
        if os.path.exists(output_path):
            logger.info(f"Dataset downloaded successfully to: {output_path}")
            return str(output_path)
        else:
            logger.warning("Download failed, creating sample dataset instead")
            return create_sample_dataset(output_path)
            
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        logger.info("Creating sample dataset instead")
        return create_sample_dataset(output_path)

def create_sample_dataset(output_path: str) -> str:
    """
    Create a sample medical dataset for development and testing
    
    Args:
        output_path: Path to save the sample dataset
    
    Returns:
        Path to the created sample dataset
    """
    
    logger = logging.getLogger(__name__)
    
    # Sample medical Q&A data covering various topics
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
        },
        {
            'question': 'How is depression treated?',
            'answer': 'Depression treatment typically involves psychotherapy, medication, or a combination of both. Antidepressant medications can help balance brain chemicals, while therapy helps develop coping strategies and address underlying issues. Lifestyle changes like exercise, sleep hygiene, and social support are also important.'
        },
        {
            'question': 'What are the symptoms of COVID-19?',
            'answer': 'Common COVID-19 symptoms include fever or chills, cough, shortness of breath, fatigue, muscle or body aches, headache, loss of taste or smell, sore throat, congestion, nausea, and diarrhea. Symptoms can range from mild to severe and may appear 2-14 days after exposure.'
        },
        {
            'question': 'How can I improve my sleep quality?',
            'answer': 'To improve sleep quality, maintain a consistent sleep schedule, create a relaxing bedtime routine, keep your bedroom cool and dark, avoid screens before bed, limit caffeine and alcohol, exercise regularly but not close to bedtime, and manage stress through relaxation techniques.'
        },
        {
            'question': 'What are the benefits of regular exercise?',
            'answer': 'Regular exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mood and mental health, weight management, increased energy levels, better sleep, reduced risk of chronic diseases, and improved immune function.'
        },
        {
            'question': 'How do I know if I have anxiety?',
            'answer': 'Anxiety symptoms include excessive worry, restlessness, difficulty concentrating, irritability, muscle tension, sleep problems, rapid heartbeat, sweating, trembling, and avoidance of certain situations. If these symptoms persist and interfere with daily life, consult a mental health professional.'
        },
        {
            'question': 'What causes migraines?',
            'answer': 'Migraines can be triggered by various factors including stress, hormonal changes, certain foods (chocolate, cheese, alcohol), caffeine, lack of sleep, bright lights, loud noises, strong smells, and weather changes. Genetic factors also play a role in migraine susceptibility.'
        },
        {
            'question': 'How is arthritis treated?',
            'answer': 'Arthritis treatment focuses on reducing pain and inflammation, improving joint function, and preventing joint damage. Treatment may include medications (NSAIDs, corticosteroids), physical therapy, exercise, weight management, assistive devices, and in severe cases, surgery.'
        },
        {
            'question': 'What are the symptoms of thyroid problems?',
            'answer': 'Thyroid symptoms vary by condition. Hypothyroidism (underactive thyroid) may cause fatigue, weight gain, cold sensitivity, dry skin, and depression. Hyperthyroidism (overactive thyroid) may cause weight loss, rapid heartbeat, anxiety, sweating, and difficulty sleeping.'
        },
        {
            'question': 'How can I manage stress?',
            'answer': 'Stress management techniques include regular exercise, meditation, deep breathing exercises, maintaining a healthy diet, getting adequate sleep, setting boundaries, practicing time management, seeking social support, and engaging in hobbies or activities you enjoy.'
        },
        {
            'question': 'What are the signs of dehydration?',
            'answer': 'Signs of dehydration include increased thirst, dark yellow urine, infrequent urination, dry mouth and skin, fatigue, dizziness, confusion, and in severe cases, rapid heartbeat and breathing. Severe dehydration requires immediate medical attention.'
        }
    ]
    
    # Create DataFrame and save
    df = pd.DataFrame(sample_data)
    
    # Ensure directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample medical dataset created with {len(sample_data)} Q&A pairs")
    logger.info(f"Dataset saved to: {output_path}")
    
    return str(output_path)

def validate_dataset(file_path: str) -> bool:
    """
    Validate the downloaded/created dataset
    
    Args:
        file_path: Path to the dataset file
    
    Returns:
        True if dataset is valid, False otherwise
    """
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Check basic structure
        if len(df.columns) < 2:
            logger.error("Dataset must have at least 2 columns (question and answer)")
            return False
        
        # Check for required columns
        required_columns = ['question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.info("Attempting to identify question and answer columns...")
            
            # Try to identify columns by name patterns
            question_col = None
            answer_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['question', 'query', 'q', 'input']):
                    question_col = col
                elif any(keyword in col_lower for keyword in ['answer', 'response', 'a', 'output', 'reply']):
                    answer_col = col
            
            if question_col and answer_col:
                logger.info(f"Identified columns: {question_col} (questions), {answer_col} (answers)")
                return True
            else:
                logger.error("Could not identify question and answer columns")
                return False
        
        # Check data quality
        if len(df) < 5:
            logger.warning("Dataset has very few samples")
        
        # Check for empty values
        empty_questions = df['question'].isna().sum()
        empty_answers = df['answer'].isna().sum()
        
        if empty_questions > 0 or empty_answers > 0:
            logger.warning(f"Found {empty_questions} empty questions and {empty_answers} empty answers")
        
        logger.info(f"Dataset validation passed. Shape: {df.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        return False

def setup_data_directory() -> str:
    """
    Set up the data directory structure
    
    Returns:
        Path to the data directory
    """
    
    data_dir = Path("data")
    
    # Create directory structure
    directories = [
        data_dir / "raw",
        data_dir / "processed",
        data_dir / "interim"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return str(data_dir)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set up data directory
    data_dir = setup_data_directory()
    print(f"Data directory set up at: {data_dir}")
    
    # Download or create dataset
    dataset_path = download_medical_dataset()
    
    # Validate dataset
    if validate_dataset(dataset_path):
        print(f"Dataset is ready for use: {dataset_path}")
    else:
        print("Dataset validation failed")
