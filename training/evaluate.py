"""
Evaluation script for the medical chat model
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from sacrebleu import BLEU
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_processor import MedicalDataProcessor
from models.medical_chat_model import MedicalChatModel
from models.rag_model import RAGMedicalModel

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_model(model_path: str, model_type: str = "transformer"):
    """Load the trained model"""
    
    logger = logging.getLogger(__name__)
    
    if model_type == "transformer":
        model = MedicalChatModel()
        model.load_model(model_path)
    elif model_type == "rag":
        model = RAGMedicalModel()
        model.load_models()
        # Load knowledge base if available
        knowledge_base_path = os.path.join(model_path, "knowledge_base.csv")
        if os.path.exists(knowledge_base_path):
            knowledge_base = pd.read_csv(knowledge_base_path)
            model.build_knowledge_base(knowledge_base)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Model loaded from: {model_path}")
    return model

def calculate_text_metrics(predictions: list, references: list) -> dict:
    """Calculate text generation metrics"""
    
    # Initialize scorers
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_scorer = BLEU()
    
    # Calculate ROUGE scores
    rouge_scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    for pred, ref in zip(predictions, references):
        scores = rouge_scorer_instance.score(ref, pred)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            rouge_scores[metric]['precision'].append(scores[metric].precision)
            rouge_scores[metric]['recall'].append(scores[metric].recall)
            rouge_scores[metric]['fmeasure'].append(scores[metric].fmeasure)
    
    # Calculate BLEU score
    bleu_score = bleu_scorer.corpus_score(predictions, [references])
    
    # Calculate average ROUGE scores
    avg_rouge_scores = {}
    for metric in rouge_scores:
        avg_rouge_scores[metric] = {
            'precision': np.mean(rouge_scores[metric]['precision']),
            'recall': np.mean(rouge_scores[metric]['recall']),
            'fmeasure': np.mean(rouge_scores[metric]['fmeasure'])
        }
    
    return {
        'rouge_scores': avg_rouge_scores,
        'bleu_score': bleu_score.score,
        'detailed_rouge': rouge_scores
    }

def evaluate_medical_accuracy(predictions: list, references: list) -> dict:
    """Evaluate medical accuracy of responses"""
    
    # Medical keywords to check for
    medical_keywords = [
        'symptom', 'treatment', 'diagnosis', 'medication', 'disease',
        'condition', 'health', 'medical', 'doctor', 'patient',
        'blood', 'heart', 'pain', 'fever', 'infection'
    ]
    
    medical_coverage = []
    medical_accuracy = []
    
    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower()
        ref_lower = ref.lower()
        
        # Check medical keyword coverage
        pred_keywords = sum(1 for keyword in medical_keywords if keyword in pred_lower)
        ref_keywords = sum(1 for keyword in medical_keywords if keyword in ref_lower)
        
        if ref_keywords > 0:
            coverage = pred_keywords / ref_keywords
            medical_coverage.append(coverage)
        
        # Simple medical accuracy (keyword overlap)
        pred_keyword_set = set(keyword for keyword in medical_keywords if keyword in pred_lower)
        ref_keyword_set = set(keyword for keyword in medical_keywords if keyword in ref_lower)
        
        if ref_keyword_set:
            accuracy = len(pred_keyword_set.intersection(ref_keyword_set)) / len(ref_keyword_set)
            medical_accuracy.append(accuracy)
    
    return {
        'avg_medical_coverage': np.mean(medical_coverage) if medical_coverage else 0,
        'avg_medical_accuracy': np.mean(medical_accuracy) if medical_accuracy else 0,
        'medical_coverage_scores': medical_coverage,
        'medical_accuracy_scores': medical_accuracy
    }

def evaluate_response_quality(predictions: list, references: list) -> dict:
    """Evaluate response quality metrics"""
    
    quality_metrics = {
        'avg_prediction_length': np.mean([len(pred.split()) for pred in predictions]),
        'avg_reference_length': np.mean([len(ref.split()) for ref in references]),
        'length_ratio': np.mean([len(pred.split()) / len(ref.split()) for pred, ref in zip(predictions, references) if len(ref.split()) > 0]),
        'empty_responses': sum(1 for pred in predictions if not pred.strip()),
        'very_short_responses': sum(1 for pred in predictions if len(pred.split()) < 5),
        'very_long_responses': sum(1 for pred in predictions if len(pred.split()) > 100)
    }
    
    return quality_metrics

def generate_evaluation_report(model, test_df, output_path: str = "evaluation_report.json"):
    """Generate comprehensive evaluation report"""
    
    logger = logging.getLogger(__name__)
    
    # Prepare test data
    test_questions = test_df['question'].tolist()
    expected_answers = test_df['answer'].tolist()
    
    logger.info(f"Evaluating model on {len(test_questions)} test questions...")
    
    # Generate predictions
    try:
        predictions = model.batch_generate(test_questions)
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        return None
    
    # Calculate metrics
    logger.info("Calculating text generation metrics...")
    text_metrics = calculate_text_metrics(predictions, expected_answers)
    
    logger.info("Calculating medical accuracy metrics...")
    medical_metrics = evaluate_medical_accuracy(predictions, expected_answers)
    
    logger.info("Calculating response quality metrics...")
    quality_metrics = evaluate_response_quality(predictions, expected_answers)
    
    # Compile comprehensive report
    evaluation_report = {
        'model_info': model.get_model_info(),
        'dataset_info': {
            'total_test_samples': len(test_questions),
            'avg_question_length': np.mean([len(q.split()) for q in test_questions]),
            'avg_answer_length': np.mean([len(a.split()) for a in expected_answers])
        },
        'text_generation_metrics': text_metrics,
        'medical_accuracy_metrics': medical_metrics,
        'response_quality_metrics': quality_metrics,
        'example_interactions': [
            {
                'question': q,
                'expected_answer': ref,
                'generated_answer': pred,
                'rouge_f1': text_metrics['detailed_rouge']['rouge1']['fmeasure'][i]
            }
            for i, (q, ref, pred) in enumerate(zip(test_questions[:5], expected_answers[:5], predictions[:5]))
        ]
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    logger.info(f"Evaluation report saved to: {output_path}")
    
    return evaluation_report

def print_evaluation_summary(report: dict):
    """Print evaluation summary"""
    
    print("\n" + "="*60)
    print("MEDICAL CHAT MODEL EVALUATION SUMMARY")
    print("="*60)
    
    # Model info
    print(f"\nModel Information:")
    print(f"  Model: {report['model_info'].get('model_name', 'Unknown')}")
    print(f"  Device: {report['model_info'].get('device', 'Unknown')}")
    print(f"  Parameters: {report['model_info'].get('total_parameters', 0):,}")
    
    # Dataset info
    print(f"\nDataset Information:")
    print(f"  Test samples: {report['dataset_info']['total_test_samples']}")
    print(f"  Avg question length: {report['dataset_info']['avg_question_length']:.1f} words")
    print(f"  Avg answer length: {report['dataset_info']['avg_answer_length']:.1f} words")
    
    # Text generation metrics
    print(f"\nText Generation Metrics:")
    print(f"  ROUGE-1 F1: {report['text_generation_metrics']['rouge_scores']['rouge1']['fmeasure']:.3f}")
    print(f"  ROUGE-2 F1: {report['text_generation_metrics']['rouge_scores']['rouge2']['fmeasure']:.3f}")
    print(f"  ROUGE-L F1: {report['text_generation_metrics']['rouge_scores']['rougeL']['fmeasure']:.3f}")
    print(f"  BLEU Score: {report['text_generation_metrics']['bleu_score']:.3f}")
    
    # Medical accuracy metrics
    print(f"\nMedical Accuracy Metrics:")
    print(f"  Medical Coverage: {report['medical_accuracy_metrics']['avg_medical_coverage']:.3f}")
    print(f"  Medical Accuracy: {report['medical_accuracy_metrics']['avg_medical_accuracy']:.3f}")
    
    # Quality metrics
    print(f"\nResponse Quality Metrics:")
    print(f"  Avg prediction length: {report['response_quality_metrics']['avg_prediction_length']:.1f} words")
    print(f"  Length ratio: {report['response_quality_metrics']['length_ratio']:.3f}")
    print(f"  Empty responses: {report['response_quality_metrics']['empty_responses']}")
    print(f"  Very short responses: {report['response_quality_metrics']['very_short_responses']}")
    
    # Example interactions
    print(f"\nExample Interactions:")
    for i, example in enumerate(report['example_interactions'][:3], 1):
        print(f"\n  Example {i}:")
        print(f"    Q: {example['question']}")
        print(f"    Expected: {example['expected_answer'][:100]}...")
        print(f"    Generated: {example['generated_answer'][:100]}...")
        print(f"    ROUGE-1 F1: {example['rouge_f1']:.3f}")
    
    print("\n" + "="*60)

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Evaluate medical chat model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-data", type=str, help="Path to test data file")
    parser.add_argument("--model-type", type=str, default="transformer", choices=["transformer", "rag"], help="Type of model to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="Output file for evaluation report")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting medical chat model evaluation...")
    
    try:
        # Load model
        model = load_model(args.model_path, args.model_type)
        
        # Load test data
        if args.test_data and os.path.exists(args.test_data):
            data_processor = MedicalDataProcessor()
            test_df = data_processor.load_and_clean_data(args.test_data)
        else:
            # Use sample data for evaluation
            logger.info("No test data provided, using sample data for evaluation...")
            from data.download_data import create_sample_dataset
            sample_path = create_sample_dataset("data/evaluation_sample.csv")
            data_processor = MedicalDataProcessor()
            test_df = data_processor.load_and_clean_data(sample_path)
        
        # Generate evaluation report
        report = generate_evaluation_report(model, test_df, args.output)
        
        if report:
            # Print summary
            print_evaluation_summary(report)
            
            logger.info("Evaluation completed successfully!")
            logger.info(f"Detailed report saved to: {args.output}")
        else:
            logger.error("Evaluation failed!")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
