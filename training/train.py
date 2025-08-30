"""
Training script for the medical chat model
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import gdown
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_processor import MedicalDataProcessor
from models.medical_chat_model import MedicalChatModel

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file"""
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'name': 'facebook/bart-base',
                'max_input_length': 512,
                'max_target_length': 256
            },
            'training': {
                'num_train_epochs': 3,
                'per_device_train_batch_size': 4,
                'per_device_eval_batch_size': 4,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'learning_rate': 5e-5,
                'save_strategy': 'epoch',
                'evaluation_strategy': 'epoch',
                'logging_steps': 100,
                'save_total_limit': 2,
                'load_best_model_at_end': True
            },
            'data': {
                'data_file': None,
                'test_size': 0.2,
                'val_size': 0.1,
                'augment_data': True
            }
        }
    
    return config

def download_sample_data():
    """Download sample medical dataset if no data file provided"""
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample medical Q&A data
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
        }
    ]
    
    # Create DataFrame and save
    df = pd.DataFrame(sample_data)
    output_path = data_dir / "sample_medical_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Sample medical dataset created at: {output_path}")
    return str(output_path)

def prepare_data(config: dict, logger: logging.Logger):
    """Prepare and preprocess the training data"""
    
    data_processor = MedicalDataProcessor()
    
    # Get data file path
    data_file = config['data']['data_file']
    
    if not data_file or not os.path.exists(data_file):
        logger.info("No data file provided or file not found. Creating sample dataset...")
        data_file = download_sample_data()
    
    # Load and clean data
    logger.info(f"Loading data from: {data_file}")
    df = data_processor.load_and_clean_data(data_file)
    
    # Validate data quality
    if not data_processor.validate_data_quality(df):
        logger.warning("Data quality issues detected. Proceeding with caution...")
    
    # Augment data if specified
    if config['data']['augment_data']:
        logger.info("Augmenting dataset...")
        df = data_processor.augment_data(df)
    
    # Split data
    train_df, val_df, test_df = data_processor.split_data(
        df, 
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size']
    )
    
    # Get dataset statistics
    train_stats = data_processor.get_dataset_statistics(train_df)
    logger.info(f"Training set statistics: {train_stats}")
    
    return train_df, val_df, test_df

def train_model(config: dict, train_df, val_df, test_df, logger: logging.Logger):
    """Train the medical chat model"""
    
    # Initialize model
    logger.info(f"Initializing model: {config['model']['name']}")
    model = MedicalChatModel(model_name=config['model']['name'])
    
    # Load model
    model.load_model()
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = model.prepare_datasets(
        train_df, val_df, test_df
    )
    
    # Training arguments
    training_args = {
        'output_dir': './medical_chat_model',
        'num_train_epochs': config['training']['num_train_epochs'],
        'per_device_train_batch_size': config['training']['per_device_train_batch_size'],
        'per_device_eval_batch_size': config['training']['per_device_eval_batch_size'],
        'warmup_steps': config['training']['warmup_steps'],
        'weight_decay': config['training']['weight_decay'],
        'learning_rate': config['training']['learning_rate'],
        'save_strategy': config['training']['save_strategy'],
        'evaluation_strategy': config['training']['evaluation_strategy'],
        'logging_steps': config['training']['logging_steps'],
        'save_total_limit': config['training']['save_total_limit'],
        'load_best_model_at_end': config['training']['load_best_model_at_end'],
        'logging_dir': './logs',
        'prediction_loss_only': True,
        'dataloader_pin_memory': False,
    }
    
    # Train the model
    logger.info("Starting training...")
    trainer = model.train(train_dataset, val_dataset, **training_args)
    
    logger.info("Training completed successfully!")
    return model, trainer

def evaluate_model(model: MedicalChatModel, test_df, logger: logging.Logger):
    """Evaluate the trained model"""
    
    logger.info("Evaluating model on test set...")
    
    # Generate responses for test questions
    test_questions = test_df['question'].tolist()
    expected_answers = test_df['answer'].tolist()
    
    generated_answers = model.batch_generate(test_questions)
    
    # Calculate basic metrics
    total_questions = len(test_questions)
    successful_generations = len([ans for ans in generated_answers if ans and len(ans) > 10])
    
    logger.info(f"Test set evaluation:")
    logger.info(f"  Total questions: {total_questions}")
    logger.info(f"  Successful generations: {successful_generations}")
    logger.info(f"  Success rate: {successful_generations/total_questions:.2%}")
    
    # Show some example interactions
    logger.info("\nExample interactions:")
    for i in range(min(3, len(test_questions))):
        logger.info(f"  Q: {test_questions[i]}")
        logger.info(f"  A: {generated_answers[i]}")
        logger.info("")
    
    return {
        'total_questions': total_questions,
        'successful_generations': successful_generations,
        'success_rate': successful_generations / total_questions,
        'example_interactions': list(zip(test_questions[:3], generated_answers[:3]))
    }

def main():
    """Main training function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train medical chat model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data-file", type=str, help="Path to training data file")
    parser.add_argument("--model-name", type=str, help="Model name to use")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting medical chat model training...")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.data_file:
            config['data']['data_file'] = args.data_file
        if args.model_name:
            config['model']['name'] = args.model_name
        if args.epochs:
            config['training']['num_train_epochs'] = args.epochs
        if args.batch_size:
            config['training']['per_device_train_batch_size'] = args.batch_size
        if args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
        
        logger.info(f"Configuration: {config}")
        
        # Prepare data
        train_df, val_df, test_df = prepare_data(config, logger)
        
        # Train model
        model, trainer = train_model(config, train_df, val_df, test_df, logger)
        
        # Evaluate model
        evaluation_results = evaluate_model(model, test_df, logger)
        
        # Save evaluation results
        import json
        with open('evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model saved to: ./medical_chat_model")
        logger.info(f"Evaluation results saved to: evaluation_results.json")
        
        # Optional: Start interactive chat
        if input("\nWould you like to test the model interactively? (y/n): ").lower() == 'y':
            model.chat_interactive()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
