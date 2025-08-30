"""
API schemas for medical chat system
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    
    question: str = Field(..., description="Medical question to ask", min_length=1, max_length=500)
    model_type: Optional[str] = Field(default="transformer", description="Type of model to use (transformer or rag)")
    max_length: Optional[int] = Field(default=256, description="Maximum length of response", ge=10, le=1000)
    temperature: Optional[float] = Field(default=0.7, description="Generation temperature", ge=0.0, le=2.0)
    num_beams: Optional[int] = Field(default=4, description="Number of beams for generation", ge=1, le=10)

class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    
    answer: str = Field(..., description="Generated medical answer")
    question: str = Field(..., description="Original question")
    model_type: str = Field(..., description="Type of model used")
    confidence: Optional[float] = Field(None, description="Confidence score of the response")
    processing_time: Optional[float] = Field(None, description="Time taken to generate response in seconds")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")

class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")

class ModelInfoResponse(BaseModel):
    """Model information response schema"""
    
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model")
    device: str = Field(..., description="Device being used")
    total_parameters: Optional[int] = Field(None, description="Total number of parameters")
    model_size_mb: Optional[float] = Field(None, description="Model size in MB")
    status: str = Field(..., description="Model status")

class BatchChatRequest(BaseModel):
    """Request schema for batch chat endpoint"""
    
    questions: List[str] = Field(..., description="List of medical questions", min_items=1, max_items=10)
    model_type: Optional[str] = Field(default="transformer", description="Type of model to use")
    max_length: Optional[int] = Field(default=256, description="Maximum length of responses")
    temperature: Optional[float] = Field(default=0.7, description="Generation temperature")

class BatchChatResponse(BaseModel):
    """Response schema for batch chat endpoint"""
    
    answers: List[str] = Field(..., description="Generated medical answers")
    questions: List[str] = Field(..., description="Original questions")
    model_type: str = Field(..., description="Type of model used")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    success_count: int = Field(..., description="Number of successful generations")
    total_count: int = Field(..., description="Total number of questions")

class ErrorResponse(BaseModel):
    """Error response schema"""
    
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[str] = Field(None, description="Additional error details")
