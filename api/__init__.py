"""
API module for medical chat system
"""

from .app import app
from .schemas import ChatRequest, ChatResponse

__all__ = ['app', 'ChatRequest', 'ChatResponse']
