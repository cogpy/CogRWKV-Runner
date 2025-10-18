"""
RWKV model interface for OpenCog integration.
Provides neural network capabilities to the cognitive architecture.
"""

import asyncio
import json
import sys
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

# Add the main backend-python to path to reuse existing RWKV utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../backend-python'))

try:
    from utils.rwkv import *
    from utils.torch import *
    import global_var
except ImportError as e:
    print(f"Warning: Could not import RWKV utilities: {e}")
    # Fallback implementations will be used


@dataclass
class GenerationRequest:
    """Request for text generation"""
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.3
    presence_penalty: float = 0.0
    frequency_penalty: float = 1.0
    stop_sequences: List[str] = None
    state_path: Optional[str] = None


@dataclass
class GenerationResponse:
    """Response from text generation"""
    text: str
    tokens_generated: int
    execution_time: float
    model_info: Dict[str, Any]
    state_info: Optional[Dict[str, Any]] = None


class RWKVCognitiveInterface:
    """
    Interface between RWKV neural network and OpenCog cognitive architecture.
    Provides neural processing capabilities for the AGI system.
    """
    
    def __init__(self):
        self.model = None
        self.model_name = ""
        self.model_config = {}
        self.generation_history = []
        self.state_cache = {}
        self.performance_metrics = {
            "total_generations": 0,
            "average_time": 0.0,
            "total_tokens": 0,
            "error_count": 0
        }
        
    async def initialize(self, model_path: str = None, config: Dict[str, Any] = None):
        """Initialize RWKV model"""
        try:
            # Initialize global variables if not already done
            if not hasattr(global_var, '_initialized'):
                global_var.init()
                global_var._initialized = True
            
            # Set up model configuration
            if config:
                self.model_config = config
            else:
                self.model_config = {
                    "max_tokens": 1000,
                    "temperature": 1.0,
                    "top_p": 0.3,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 1.0,
                    "penalty_decay": 0.996,
                    "global_penalty": False
                }
            
            # Try to load existing model or set up for future loading
            if model_path and os.path.exists(model_path):
                self.model_name = model_path
                # Model will be loaded on first use
                print(f"RWKV model configured: {model_path}")
            else:
                print("RWKV model not found, will use fallback generation")
                
            return True
            
        except Exception as e:
            print(f"Error initializing RWKV interface: {str(e)}")
            return False
    
    async def generate(self, request: Union[GenerationRequest, str, Dict[str, Any]]) -> GenerationResponse:
        """Generate text using RWKV model"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Parse request
            if isinstance(request, str):
                req = GenerationRequest(prompt=request)
            elif isinstance(request, dict):
                req = GenerationRequest(
                    prompt=request.get("prompt", ""),
                    max_tokens=request.get("max_tokens", 100),
                    temperature=request.get("temperature", 1.0),
                    top_p=request.get("top_p", 0.3),
                    presence_penalty=request.get("presence_penalty", 0.0),
                    frequency_penalty=request.get("frequency_penalty", 1.0),
                    stop_sequences=request.get("stop", [])
                )
            else:
                req = request
            
            # Generate text
            if self.model and hasattr(self, '_rwkv_available'):
                # Use actual RWKV model
                generated_text = await self._generate_with_rwkv(req)
            else:
                # Fallback generation for development/testing
                generated_text = await self._generate_fallback(req)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Update metrics
            self.performance_metrics["total_generations"] += 1
            self.performance_metrics["total_tokens"] += len(generated_text.split())
            self.performance_metrics["average_time"] = (
                (self.performance_metrics["average_time"] * 
                 (self.performance_metrics["total_generations"] - 1) + execution_time) /
                self.performance_metrics["total_generations"]
            )
            
            response = GenerationResponse(
                text=generated_text,
                tokens_generated=len(generated_text.split()),
                execution_time=execution_time,
                model_info={
                    "model_name": self.model_name,
                    "config": self.model_config
                }
            )
            
            # Store in history
            self.generation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "prompt": req.prompt,
                "response": generated_text,
                "execution_time": execution_time
            })
            
            return response
            
        except Exception as e:
            self.performance_metrics["error_count"] += 1
            print(f"Error in text generation: {str(e)}")
            
            return GenerationResponse(
                text=f"[Error: {str(e)}]",
                tokens_generated=0,
                execution_time=asyncio.get_event_loop().time() - start_time,
                model_info={"error": str(e)}
            )
    
    async def _generate_with_rwkv(self, request: GenerationRequest) -> str:
        """Generate text using actual RWKV model"""
        try:
            # This would integrate with the existing RWKV implementation
            # For now, we'll use a placeholder that calls the main backend
            
            # Prepare generation parameters
            gen_params = {
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "stop": request.stop_sequences or []
            }
            
            # In a full implementation, this would call the RWKV model directly
            # For now, return a cognitive response
            return await self._generate_cognitive_response(request.prompt)
            
        except Exception as e:
            raise Exception(f"RWKV generation failed: {str(e)}")
    
    async def _generate_fallback(self, request: GenerationRequest) -> str:
        """Fallback text generation for testing"""
        return await self._generate_cognitive_response(request.prompt)
    
    async def _generate_cognitive_response(self, prompt: str) -> str:
        """Generate a cognitive response based on prompt analysis"""
        prompt_lower = prompt.lower()
        
        # Simple pattern-based responses for testing
        if "what" in prompt_lower and "?" in prompt:
            if "name" in prompt_lower:
                return "I am an OpenCog-RWKV cognitive agent, designed to integrate neural and symbolic reasoning for AGI capabilities."
            elif "time" in prompt_lower:
                return f"The current time is {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC."
            elif "opencog" in prompt_lower:
                return "OpenCog is an open-source AGI framework that combines symbolic and connectionist approaches to achieve artificial general intelligence through cognitive synergy."
            elif "rwkv" in prompt_lower:
                return "RWKV is a novel RNN architecture that combines the benefits of transformers and RNNs, offering efficient training and inference for large language models."
            else:
                return "I understand you're asking a question. Let me think about this using my cognitive architecture to provide the best response."
        
        elif "how" in prompt_lower and "?" in prompt:
            return "This involves a process that I can analyze using both neural pattern recognition and symbolic reasoning. Let me break this down step by step."
        
        elif "why" in prompt_lower and "?" in prompt:
            return "That's an interesting question that requires causal reasoning. Based on my knowledge representation and inference capabilities, I can explore the underlying reasons."
        
        elif "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! I'm an OpenCog-RWKV cognitive agent. I combine neural network processing with symbolic reasoning to provide intelligent responses. How can I help you today?"
        
        elif "think" in prompt_lower or "reason" in prompt_lower:
            return "I'm engaging my cognitive architecture to process this. My reasoning involves both bottom-up neural pattern recognition and top-down symbolic inference, allowing for comprehensive analysis."
        
        elif "learn" in prompt_lower:
            return "Learning is a core capability of my architecture. I can acquire new knowledge through both neural adaptation and symbolic concept formation, integrating experiences into my atomspace knowledge representation."
        
        else:
            # General cognitive response
            return f"I'm processing your input: '{prompt}' through my cognitive architecture. This involves neural pattern analysis, symbolic reasoning, and knowledge integration to formulate an appropriate response."
    
    async def get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Placeholder embeddings - in real implementation would use RWKV
        embeddings = []
        for text in texts:
            # Simple hash-based embedding for testing
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
            # Pad or truncate to standard size (e.g., 768)
            while len(embedding) < 768:
                embedding.extend(embedding)
            embedding = embedding[:768]
            embeddings.append(embedding)
        
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    async def save_state(self, state_name: str) -> bool:
        """Save current model state"""
        try:
            state_data = {
                "model_config": self.model_config,
                "generation_history": self.generation_history[-10:],  # Keep last 10
                "performance_metrics": self.performance_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.state_cache[state_name] = state_data
            
            # In full implementation, would save to disk
            return True
            
        except Exception as e:
            print(f"Error saving state: {str(e)}")
            return False
    
    async def load_state(self, state_name: str) -> bool:
        """Load saved model state"""
        try:
            if state_name in self.state_cache:
                state_data = self.state_cache[state_name]
                
                # Restore configuration
                self.model_config.update(state_data.get("model_config", {}))
                
                # Restore some history
                if "generation_history" in state_data:
                    self.generation_history.extend(state_data["generation_history"])
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.performance_metrics.copy()
    
    def get_generation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent generation history"""
        return self.generation_history[-limit:]
    
    def clear_history(self):
        """Clear generation history"""
        self.generation_history.clear()
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update model configuration"""
        self.model_config.update(new_config)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "config": self.model_config,
            "is_loaded": self.model is not None,
            "performance": self.performance_metrics
        }