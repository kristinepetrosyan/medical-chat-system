"""
FastAPI application for medical chat system
"""

import time
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .schemas import (
    ChatRequest, ChatResponse, HealthCheckResponse, ModelInfoResponse,
    BatchChatRequest, BatchChatResponse, ErrorResponse
)

# Import models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.medical_chat_model import MedicalChatModel
from models.rag_model import RAGMedicalModel

# Initialize FastAPI app
app = FastAPI(
    title="Medical Chat System API",
    description="AI-powered medical chat system for answering health-related queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
transformer_model: Optional[MedicalChatModel] = None
rag_model: Optional[RAGMedicalModel] = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global transformer_model, rag_model
    
    try:
        logger.info("Initializing medical chat models...")
        
        # Initialize transformer model
        transformer_model = MedicalChatModel()
        transformer_model.load_model()
        
        # Initialize RAG model
        rag_model = RAGMedicalModel()
        rag_model.load_models()
        
        try:
            from data.download_data import create_sample_dataset
            sample_path = create_sample_dataset("data/rag_knowledge_base.csv")
            import pandas as pd
            knowledge_base = pd.read_csv(sample_path)
            rag_model.build_knowledge_base(knowledge_base)
        except Exception as e:
            logger.warning(f"Could not load knowledge base for RAG model: {e}")
        
        logger.info("Models initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Chat System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = transformer_model is not None and rag_model is not None
    
    model_info = None
    if transformer_model:
        model_info = transformer_model.get_model_info()
    
    return HealthCheckResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_info=model_info
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    if not transformer_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = transformer_model.get_model_info()
    
    return ModelInfoResponse(
        model_name=info.get("model_name", "Unknown"),
        model_type="transformer",
        device=info.get("device", "Unknown"),
        total_parameters=info.get("total_parameters"),
        model_size_mb=info.get("model_size_mb"),
        status=info.get("status", "Unknown")
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Single chat endpoint"""
    
    start_time = time.time()
    
    try:
        # Validate model type
        if request.model_type not in ["transformer", "rag"]:
            raise HTTPException(status_code=400, detail="Invalid model type. Use 'transformer' or 'rag'")
        
        if request.model_type == "transformer":
            if not transformer_model:
                raise HTTPException(status_code=503, detail="Transformer model not loaded")
            model = transformer_model
        else:  # rag
            if not rag_model:
                raise HTTPException(status_code=503, detail="RAG model not loaded")
            model = rag_model
        
        # Generate response
        response = model.generate_response(
            question=request.question,
            max_length=request.max_length,
            temperature=request.temperature,
            num_beams=request.num_beams
        )
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            answer=response,
            question=request.question,
            model_type=request.model_type,
            processing_time=processing_time,
            model_info=model.get_model_info()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat/batch", response_model=BatchChatResponse)
async def batch_chat(request: BatchChatRequest):
    """Batch chat endpoint"""
    
    start_time = time.time()
    
    try:
        # Validate model type
        if request.model_type not in ["transformer", "rag"]:
            raise HTTPException(status_code=400, detail="Invalid model type. Use 'transformer' or 'rag'")
        
        # Get appropriate model
        if request.model_type == "transformer":
            if not transformer_model:
                raise HTTPException(status_code=503, detail="Transformer model not loaded")
            model = transformer_model
        else:  # rag
            if not rag_model:
                raise HTTPException(status_code=503, detail="RAG model not loaded")
            model = rag_model
        
        # Generate responses
        answers = model.batch_generate(
            questions=request.questions,
            max_length=request.max_length,
            temperature=request.temperature,
            num_beams=request.num_beams
        )
        
        processing_time = time.time() - start_time
        
        # Count successful generations
        success_count = sum(1 for ans in answers if ans and len(ans.strip()) > 0)
        
        return BatchChatResponse(
            answers=answers,
            questions=request.questions,
            model_type=request.model_type,
            processing_time=processing_time,
            success_count=success_count,
            total_count=len(request.questions)
        )
        
    except Exception as e:
        logger.error(f"Error in batch chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating responses: {str(e)}")

@app.get("/examples")
async def get_examples():
    """Get example questions for testing"""
    
    examples = [
        {
            "question": "What are the symptoms of diabetes?",
            "category": "symptoms",
            "difficulty": "easy"
        },
        {
            "question": "How is high blood pressure treated?",
            "category": "treatment",
            "difficulty": "medium"
        },
        {
            "question": "What causes asthma?",
            "category": "causes",
            "difficulty": "medium"
        },
        {
            "question": "How can I prevent heart disease?",
            "category": "prevention",
            "difficulty": "easy"
        },
        {
            "question": "What are the warning signs of a heart attack?",
            "category": "emergency",
            "difficulty": "important"
        }
    ]
    
    return {"examples": examples}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_type="internal_error",
            details=str(exc)
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
