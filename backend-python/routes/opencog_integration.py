"""
OpenCog integration route for main RWKV-Runner backend.
Provides AGI capabilities through OpenCog backend integration.
"""

import asyncio
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
import aiohttp

router = APIRouter()

# OpenCog backend configuration
OPENCOG_BASE_URL = "http://127.0.0.1:8001/api/v1"
OPENCOG_TIMEOUT = 30


class OpenCogRequest(BaseModel):
    """Request for OpenCog processing"""
    prompt: str = Field(..., description="Input prompt for AGI processing")
    use_reasoning: bool = Field(default=True, description="Enable reasoning capabilities")
    reasoning_depth: int = Field(default=3, ge=1, le=10)
    max_tokens: int = Field(default=150, ge=1, le=2000)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    cognitive_mode: str = Field(default="standard", description="Cognitive processing mode")


class OpenCogGoal(BaseModel):
    """Goal for OpenCog agent"""
    type: str = Field(..., description="Goal type")
    description: str = Field(..., description="Goal description")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class OpenCogKnowledge(BaseModel):
    """Knowledge operation request"""
    operation: str = Field(..., description="Operation type (add_concept, add_relation, query)")
    data: Dict[str, Any] = Field(..., description="Operation data")


async def check_opencog_availability() -> bool:
    """Check if OpenCog backend is available"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{OPENCOG_BASE_URL}/opencog/status") as response:
                return response.status == 200
    except Exception:
        return False


async def call_opencog_api(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Call OpenCog backend API"""
    url = f"{OPENCOG_BASE_URL}{endpoint}"
    timeout = aiohttp.ClientTimeout(total=OPENCOG_TIMEOUT)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"OpenCog API error: {await response.text()}"
                        )
            else:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"OpenCog API error: {await response.text()}"
                        )
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=503,
            detail=f"OpenCog backend unavailable: {str(e)}"
        )


@router.get("/opencog/status", tags=["OpenCog Integration"])
async def opencog_status():
    """Get OpenCog backend status"""
    available = await check_opencog_availability()
    
    if not available:
        return {
            "available": False,
            "status": "unavailable",
            "message": "OpenCog backend is not running"
        }
    
    try:
        status_data = await call_opencog_api("/opencog/status")
        return {
            "available": True,
            "status": "running",
            "opencog_data": status_data
        }
    except Exception as e:
        # Log the actual error for debugging
        print(f"OpenCog status check error: {type(e).__name__}: {str(e)}")
        
        return {
            "available": False,
            "status": "error",
            "error": "Unable to retrieve OpenCog backend status"
        }


@router.post("/opencog/cognitive/process", tags=["OpenCog Integration"])
async def opencog_cognitive_process(request: OpenCogRequest):
    """Process input through OpenCog cognitive architecture"""
    if not await check_opencog_availability():
        raise HTTPException(
            status_code=503,
            detail="OpenCog backend is not available"
        )
    
    # Prepare request for OpenCog
    opencog_request = {
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "use_reasoning": request.use_reasoning,
        "reasoning_depth": request.reasoning_depth
    }
    
    try:
        # Call OpenCog cognitive processing
        result = await call_opencog_api(
            "/opencog/cognitive/process",
            method="POST",
            data=opencog_request
        )
        
        # Enhance response with integration metadata
        enhanced_result = {
            **result,
            "integration_info": {
                "backend": "opencog",
                "mode": request.cognitive_mode,
                "timestamp": datetime.utcnow().isoformat(),
                "rwkv_runner_version": "opencog_integrated"
            }
        }
        
        return enhanced_result
        
    except Exception as e:
        # Log the actual error for debugging
        print(f"OpenCog processing error: {type(e).__name__}: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail="OpenCog processing failed. Please check your request and try again."
        )


@router.post("/opencog/enhanced/completion", tags=["OpenCog Integration"])
async def opencog_enhanced_completion(
    prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.8,
    use_agi_reasoning: bool = True
):
    """Enhanced completion using OpenCog AGI capabilities"""
    
    # Check if this is a complex query that would benefit from AGI processing
    complex_indicators = [
        "why", "how", "explain", "analyze", "reason", "think", "understand",
        "relationship", "compare", "contrast", "implications", "consequences"
    ]
    
    is_complex = any(indicator in prompt.lower() for indicator in complex_indicators)
    
    if not is_complex or not use_agi_reasoning:
        # Use standard RWKV processing
        # This would integrate with the existing RWKV completion endpoint
        return {
            "text": f"[Standard RWKV Response for: {prompt}]",
            "backend": "rwkv_standard",
            "tokens": max_tokens,
            "reasoning_used": False
        }
    
    # Use OpenCog for complex reasoning
    if not await check_opencog_availability():
        # Fallback to standard processing
        return {
            "text": f"[Fallback RWKV Response for: {prompt}]",
            "backend": "rwkv_fallback", 
            "tokens": max_tokens,
            "reasoning_used": False,
            "note": "OpenCog backend unavailable, using fallback"
        }
    
    try:
        # Process through OpenCog
        opencog_request = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_reasoning": True,
            "reasoning_depth": 3
        }
        
        result = await call_opencog_api(
            "/opencog/cognitive/process",
            method="POST",
            data=opencog_request
        )
        
        return {
            "text": result["output"],
            "backend": "opencog_agi",
            "tokens": len(result["output"].split()),
            "reasoning_used": True,
            "goal_id": result.get("goal_id"),
            "agent_state": result.get("agent_state")
        }
        
    except Exception as e:
        # Log the actual error for debugging
        print(f"AGI processing error: {type(e).__name__}: {str(e)}")
        
        # Fallback to standard processing on error
        return {
            "text": f"[Error in AGI processing, using fallback response for the provided prompt]",
            "backend": "rwkv_error_fallback",
            "tokens": max_tokens,
            "reasoning_used": False,
            "error": "AGI processing temporarily unavailable"
        }


@router.post("/opencog/goals/add", tags=["OpenCog Integration"]) 
async def add_opencog_goal(goal: OpenCogGoal):
    """Add a goal to OpenCog cognitive agent"""
    if not await check_opencog_availability():
        raise HTTPException(
            status_code=503,
            detail="OpenCog backend is not available"
        )
    
    goal_data = {
        "type": goal.type,
        "description": goal.description,
        "parameters": goal.parameters,
        "priority": goal.priority
    }
    
    try:
        result = await call_opencog_api(
            "/opencog/agent/goals",
            method="POST",
            data=goal_data
        )
        
        return {
            "success": True,
            "goal_id": result["goal_id"],
            "message": f"Goal added to OpenCog agent: {goal.description}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add goal: {str(e)}"
        )


@router.post("/opencog/knowledge/operation", tags=["OpenCog Integration"])
async def opencog_knowledge_operation(request: OpenCogKnowledge):
    """Perform knowledge base operations in OpenCog"""
    if not await check_opencog_availability():
        raise HTTPException(
            status_code=503,
            detail="OpenCog backend is not available"
        )
    
    try:
        if request.operation == "add_concept":
            # Add concept to atomspace
            concept_data = {
                "type": "CONCEPT",
                "name": request.data.get("name"),
                "truth_value": request.data.get("truth_value", 0.8),
                "confidence": request.data.get("confidence", 0.7)
            }
            
            result = await call_opencog_api(
                "/opencog/atoms",
                method="POST",
                data=concept_data
            )
            
            return {
                "success": True,
                "operation": "add_concept",
                "atom_id": result["atom_id"],
                "message": f"Concept '{concept_data['name']}' added to knowledge base"
            }
        
        elif request.operation == "add_relation":
            # Add relationship between concepts
            relation_data = {
                "type": request.data.get("relation_type", "INHERITANCE"),
                "source_id": request.data.get("source_id"),
                "target_id": request.data.get("target_id"),
                "truth_value": request.data.get("truth_value", 0.8)
            }
            
            result = await call_opencog_api(
                "/opencog/links",
                method="POST",
                data=relation_data
            )
            
            return {
                "success": True,
                "operation": "add_relation",
                "link_id": result["link_id"],
                "message": "Relationship added to knowledge base"
            }
        
        elif request.operation == "query":
            # Query knowledge base
            query_data = {
                "query_type": request.data.get("query_type", "concepts"),
                "parameters": request.data.get("parameters", {}),
                "limit": request.data.get("limit", 20)
            }
            
            result = await call_opencog_api(
                "/opencog/knowledge/query",
                method="POST",
                data=query_data
            )
            
            return {
                "success": True,
                "operation": "query",
                "results": result["results"],
                "count": result["count"]
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown operation: {request.operation}"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge operation failed: {str(e)}"
        )


@router.get("/opencog/agent/performance", tags=["OpenCog Integration"])
async def get_opencog_agent_performance():
    """Get OpenCog agent performance metrics"""
    if not await check_opencog_availability():
        raise HTTPException(
            status_code=503,
            detail="OpenCog backend is not available"
        )
    
    try:
        result = await call_opencog_api("/opencog/agent/performance")
        
        return {
            "available": True,
            "performance_data": result,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance data: {str(e)}"
        )


@router.post("/opencog/metta/evaluate", tags=["OpenCog Integration"])
async def evaluate_metta_expression(expression: str, context: Dict[str, Any] = None):
    """Evaluate MeTTa expression in OpenCog"""
    if not await check_opencog_availability():
        raise HTTPException(
            status_code=503,
            detail="OpenCog backend is not available"
        )
    
    metta_request = {
        "expression": expression,
        "context": context or {}
    }
    
    try:
        result = await call_opencog_api(
            "/opencog/metta/evaluate",
            method="POST",
            data=metta_request
        )
        
        return {
            "success": True,
            "expression": expression,
            "result": result["result"],
            "evaluation_time": result.get("evaluation_time")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MeTTa evaluation failed: {str(e)}"
        )


@router.get("/opencog/export", tags=["OpenCog Integration"])
async def export_opencog_knowledge():
    """Export OpenCog knowledge base"""
    if not await check_opencog_availability():
        raise HTTPException(
            status_code=503,
            detail="OpenCog backend is not available"
        )
    
    try:
        result = await call_opencog_api("/opencog/export")
        
        return {
            "success": True,
            "knowledge_base": result["data"],
            "atom_count": result["atom_count"],
            "exported_at": result["exported_at"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )


# Background task for OpenCog integration monitoring
async def monitor_opencog_integration():
    """Monitor OpenCog backend integration"""
    while True:
        try:
            available = await check_opencog_availability()
            # Log status or perform maintenance tasks
            if not available:
                print(f"Warning: OpenCog backend unavailable at {datetime.utcnow()}")
        except Exception as e:
            print(f"Error monitoring OpenCog: {e}")
        
        await asyncio.sleep(60)  # Check every minute


# Start monitoring task when module is imported
asyncio.create_task(monitor_opencog_integration())