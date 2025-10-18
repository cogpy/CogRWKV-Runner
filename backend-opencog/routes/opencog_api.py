"""
OpenCog API routes for RWKV-Runner.
Provides REST API endpoints for cognitive processing and AGI functionality.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field

from ..core.atomspace import AtomSpace, Atom, AtomType
from ..core.metta_bridge import MeTTaInterpreter
from ..core.cognitive_agent import CognitiveAgent, Goal, GoalType, AgentState
from ..utils.rwkv_interface import RWKVCognitiveInterface, GenerationRequest


# API Models
class AtomRequest(BaseModel):
    type: str = Field(..., description="Atom type (e.g., 'CONCEPT', 'PREDICATE')")
    name: str = Field(..., description="Atom name")
    truth_value: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LinkRequest(BaseModel):
    type: str = Field(..., description="Link type (e.g., 'INHERITANCE', 'SIMILARITY')")
    source_id: str = Field(..., description="Source atom ID")
    target_id: str = Field(..., description="Target atom ID")
    truth_value: float = Field(default=0.5, ge=0.0, le=1.0)


class MeTTaQuery(BaseModel):
    expression: str = Field(..., description="MeTTa expression to evaluate")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation context")


class CognitiveRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for cognitive processing")
    max_tokens: int = Field(default=150, ge=1, le=2000)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    use_reasoning: bool = Field(default=True, description="Enable reasoning capabilities")
    reasoning_depth: int = Field(default=3, ge=1, le=10)


class GoalRequest(BaseModel):
    type: str = Field(..., description="Goal type")
    description: str = Field(..., description="Goal description")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    deadline_minutes: Optional[int] = Field(default=None, description="Deadline in minutes")


class KnowledgeQuery(BaseModel):
    query_type: str = Field(..., description="Type of query (concepts, relations, pattern)")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=20, ge=1, le=100)


router = APIRouter()

# Global instances - in production, these would be properly managed
atomspace = AtomSpace()
rwkv_interface = RWKVCognitiveInterface()
metta_interpreter = MeTTaInterpreter(atomspace, rwkv_interface)
cognitive_agent = CognitiveAgent("main_agent", atomspace, metta_interpreter, rwkv_interface)

# Initialize RWKV interface
asyncio.create_task(rwkv_interface.initialize())


@router.get("/opencog/status", tags=["OpenCog"])
async def get_opencog_status():
    """Get OpenCog system status"""
    agent_status = cognitive_agent.get_status()
    performance = cognitive_agent.get_performance_metrics()
    
    return {
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "atomspace_size": len(atomspace.atoms),
        "agent_status": agent_status,
        "performance_metrics": performance,
        "rwkv_status": rwkv_interface.get_model_info(),
        "version": "1.0.0"
    }


@router.post("/opencog/atoms", tags=["Atomspace"])
async def create_atom(request: AtomRequest):
    """Create a new atom in the atomspace"""
    try:
        atom_type = AtomType(request.type.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid atom type: {request.type}"
        )
    
    atom = atomspace.create_atom(
        atom_type=atom_type,
        name=request.name,
        truth_value=request.truth_value,
        confidence=request.confidence
    )
    
    return {
        "success": True,
        "atom_id": atom.id,
        "atom": {
            "id": atom.id,
            "type": atom.type.value,
            "name": atom.name,
            "truth_value": atom.truth_value,
            "confidence": atom.confidence,
            "created_at": atom.created_at.isoformat()
        }
    }


@router.get("/opencog/atoms/{atom_id}", tags=["Atomspace"])
async def get_atom(atom_id: str):
    """Get atom by ID"""
    atom = atomspace.get_atom_by_id(atom_id)
    
    if not atom:
        raise HTTPException(
            status_code=404,
            detail=f"Atom not found: {atom_id}"
        )
    
    return {
        "atom": {
            "id": atom.id,
            "type": atom.type.value,
            "name": atom.name,
            "truth_value": atom.truth_value,
            "confidence": atom.confidence,
            "attention_value": atom.attention_value,
            "incoming": list(atom.incoming),
            "outgoing": list(atom.outgoing),
            "created_at": atom.created_at.isoformat(),
            "updated_at": atom.updated_at.isoformat(),
            "metadata": atom.metadata
        }
    }


@router.post("/opencog/links", tags=["Atomspace"])
async def create_link(request: LinkRequest):
    """Create a link between atoms"""
    try:
        link_type = AtomType(request.type.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid link type: {request.type}"
        )
    
    link = atomspace.create_link(
        link_type=link_type,
        source_id=request.source_id,
        target_id=request.target_id,
        truth_value=request.truth_value
    )
    
    if not link:
        raise HTTPException(
            status_code=400,
            detail="Failed to create link - check that source and target atoms exist"
        )
    
    return {
        "success": True,
        "link_id": link.id,
        "link": {
            "id": link.id,
            "type": link.type.value,
            "name": link.name,
            "truth_value": link.truth_value,
            "source_id": request.source_id,
            "target_id": request.target_id
        }
    }


@router.get("/opencog/atoms", tags=["Atomspace"])
async def list_atoms(
    atom_type: Optional[str] = None,
    name_pattern: Optional[str] = None,
    limit: int = 50
):
    """List atoms with optional filtering"""
    atoms = list(atomspace.atoms.values())
    
    # Apply filters
    if atom_type:
        try:
            filter_type = AtomType(atom_type.upper())
            atoms = [a for a in atoms if a.type == filter_type]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid atom type: {atom_type}"
            )
    
    if name_pattern:
        atoms = [a for a in atoms if name_pattern.lower() in a.name.lower()]
    
    # Limit results
    atoms = atoms[:limit]
    
    return {
        "atoms": [
            {
                "id": atom.id,
                "type": atom.type.value,
                "name": atom.name,
                "truth_value": atom.truth_value,
                "confidence": atom.confidence,
                "attention_value": atom.attention_value
            }
            for atom in atoms
        ],
        "total_count": len(atoms),
        "filtered": atom_type is not None or name_pattern is not None
    }


@router.post("/opencog/metta/evaluate", tags=["MeTTa"])
async def evaluate_metta(request: MeTTaQuery):
    """Evaluate MeTTa expression"""
    try:
        result = await metta_interpreter.evaluate(
            request.expression,
            request.context or {}
        )
        
        return {
            "success": True,
            "expression": request.expression,
            "result": result,
            "evaluation_time": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"MeTTa evaluation error: {str(e)}"
        )


@router.post("/opencog/metta/define", tags=["MeTTa"])
async def define_metta_procedure(name: str, body: str):
    """Define a new MeTTa procedure"""
    try:
        metta_interpreter.define_procedure(name, body)
        
        return {
            "success": True,
            "procedure_name": name,
            "body": body,
            "defined_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Procedure definition error: {str(e)}"
        )


@router.get("/opencog/metta/trace", tags=["MeTTa"])
async def get_cognitive_trace():
    """Get cognitive reasoning trace"""
    trace = metta_interpreter.get_cognitive_trace()
    
    return {
        "trace": trace,
        "total_steps": len(trace),
        "retrieved_at": datetime.utcnow().isoformat()
    }


@router.delete("/opencog/metta/trace", tags=["MeTTa"])
async def clear_cognitive_trace():
    """Clear cognitive reasoning trace"""
    metta_interpreter.clear_cognitive_trace()
    
    return {
        "success": True,
        "cleared_at": datetime.utcnow().isoformat()
    }


@router.post("/opencog/cognitive/process", tags=["Cognitive Processing"])
async def cognitive_process(request: CognitiveRequest):
    """Process input through cognitive architecture"""
    try:
        # Create a goal for processing this request
        goal_id = str(uuid.uuid4())
        
        if "?" in request.prompt:
            goal_type = GoalType.ANSWER_QUESTION
            parameters = {"question": request.prompt}
        else:
            goal_type = GoalType.GENERATE_TEXT
            parameters = {
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }
        
        goal = Goal(
            id=goal_id,
            type=goal_type,
            description=f"Process: {request.prompt[:50]}...",
            parameters=parameters,
            priority=0.8
        )
        
        # Add goal to agent
        await cognitive_agent.add_goal(goal)
        
        # Process goals
        await cognitive_agent.process_goals()
        
        # Get the latest cognitive step for this goal
        relevant_steps = [
            step for step in cognitive_agent.cognitive_steps
            if step.step_type.endswith(goal_type.value)
        ]
        
        if relevant_steps:
            latest_step = relevant_steps[-1]
            result_text = latest_step.output_data.get("generated_text") or \
                         latest_step.output_data.get("answer", "No response generated")
        else:
            # Fallback generation
            gen_request = GenerationRequest(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            response = await rwkv_interface.generate(gen_request)
            result_text = response.text
        
        return {
            "success": True,
            "input": request.prompt,
            "output": result_text,
            "goal_id": goal_id,
            "processing_time": datetime.utcnow().isoformat(),
            "agent_state": cognitive_agent.state.value,
            "reasoning_used": request.use_reasoning
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cognitive processing error: {str(e)}"
        )


@router.post("/opencog/agent/goals", tags=["Cognitive Agent"])
async def add_goal(request: GoalRequest):
    """Add a new goal to the cognitive agent"""
    try:
        goal_type = GoalType(request.type.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid goal type: {request.type}"
        )
    
    goal_id = str(uuid.uuid4())
    deadline = None
    
    if request.deadline_minutes:
        deadline = datetime.utcnow() + timedelta(minutes=request.deadline_minutes)
    
    goal = Goal(
        id=goal_id,
        type=goal_type,
        description=request.description,
        parameters=request.parameters,
        priority=request.priority,
        deadline=deadline
    )
    
    await cognitive_agent.add_goal(goal)
    
    return {
        "success": True,
        "goal_id": goal_id,
        "goal": {
            "id": goal.id,
            "type": goal.type.value,
            "description": goal.description,
            "priority": goal.priority,
            "deadline": goal.deadline.isoformat() if goal.deadline else None,
            "status": goal.status
        }
    }


@router.get("/opencog/agent/goals", tags=["Cognitive Agent"])
async def list_goals():
    """List agent goals"""
    return {
        "goals": [
            {
                "id": goal.id,
                "type": goal.type.value,
                "description": goal.description,
                "priority": goal.priority,
                "progress": goal.progress,
                "status": goal.status,
                "created_at": goal.created_at.isoformat(),
                "deadline": goal.deadline.isoformat() if goal.deadline else None
            }
            for goal in cognitive_agent.goals
        ]
    }


@router.post("/opencog/agent/process-goals", tags=["Cognitive Agent"])
async def process_goals(background_tasks: BackgroundTasks):
    """Trigger goal processing"""
    background_tasks.add_task(cognitive_agent.process_goals)
    
    return {
        "success": True,
        "message": "Goal processing started",
        "agent_state": cognitive_agent.state.value
    }


@router.get("/opencog/agent/performance", tags=["Cognitive Agent"])
async def get_agent_performance():
    """Get agent performance metrics"""
    return {
        "performance": cognitive_agent.get_performance_metrics(),
        "status": cognitive_agent.get_status(),
        "retrieved_at": datetime.utcnow().isoformat()
    }


@router.post("/opencog/knowledge/query", tags=["Knowledge Base"])
async def query_knowledge(request: KnowledgeQuery):
    """Query knowledge base"""
    try:
        if request.query_type == "concepts":
            atoms = atomspace.find_atoms_by_type(AtomType.CONCEPT)[:request.limit]
            results = [
                {
                    "id": atom.id,
                    "name": atom.name,
                    "truth_value": atom.truth_value,
                    "confidence": atom.confidence
                }
                for atom in atoms
            ]
        
        elif request.query_type == "pattern":
            pattern = request.parameters
            matches = atomspace.pattern_match(pattern)[:request.limit]
            results = matches
        
        elif request.query_type == "relations":
            atom_id = request.parameters.get("atom_id")
            if not atom_id:
                raise HTTPException(status_code=400, detail="atom_id required for relations query")
            
            neighbors = atomspace.get_neighbors(atom_id)[:request.limit]
            results = [
                {
                    "neighbor_id": nid,
                    "neighbor": atomspace.get_atom_by_id(nid).name if atomspace.get_atom_by_id(nid) else "Unknown"
                }
                for nid in neighbors
            ]
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown query type: {request.query_type}"
            )
        
        return {
            "success": True,
            "query_type": request.query_type,
            "results": results,
            "count": len(results),
            "limit": request.limit
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge query error: {str(e)}"
        )


@router.get("/opencog/export", tags=["Data Management"])
async def export_atomspace():
    """Export atomspace to JSON"""
    try:
        data = atomspace.export_to_dict()
        
        return {
            "success": True,
            "data": data,
            "exported_at": datetime.utcnow().isoformat(),
            "atom_count": len(atomspace.atoms)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Export error: {str(e)}"
        )


@router.post("/opencog/import", tags=["Data Management"])
async def import_atomspace(data: Dict[str, Any]):
    """Import atomspace from JSON"""
    try:
        atomspace.import_from_dict(data)
        
        return {
            "success": True,
            "imported_at": datetime.utcnow().isoformat(),
            "atom_count": len(atomspace.atoms)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Import error: {str(e)}"
        )