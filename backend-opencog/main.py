"""
OpenCog Backend Server for RWKV-Runner.
Provides AGI capabilities through neural-symbolic integration.
"""

import asyncio
import argparse
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Union, Sequence

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from routes.opencog_api import router as opencog_router
from core.atomspace import AtomSpace, AtomType
from core.metta_bridge import MeTTaInterpreter
from core.cognitive_agent import CognitiveAgent, Goal, GoalType
from utils.rwkv_interface import RWKVCognitiveInterface


def get_args(args: Union[Sequence[str], None] = None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="OpenCog Backend Server for RWKV-Runner"
    )
    
    # Server configuration
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on (default: 8001)"
    )
    server_group.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the server on (default: 127.0.0.1)"
    )
    
    # OpenCog configuration
    opencog_group = parser.add_argument_group("OpenCog Configuration")
    opencog_group.add_argument(
        "--atomspace-size",
        type=int,
        default=10000,
        help="Initial atomspace capacity (default: 10000)"
    )
    opencog_group.add_argument(
        "--cognitive-cycles",
        type=int,
        default=10,
        help="Cognitive processing cycles per second (default: 10)"
    )
    
    # RWKV configuration
    rwkv_group = parser.add_argument_group("RWKV Configuration")
    rwkv_group.add_argument(
        "--rwkv-model",
        type=str,
        default=None,
        help="Path to RWKV model file"
    )
    rwkv_group.add_argument(
        "--rwkv-config",
        type=str,
        default=None,
        help="Path to RWKV configuration file"
    )
    
    # Development options
    dev_group = parser.add_argument_group("Development Options")
    dev_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    dev_group.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes"
    )
    dev_group.add_argument(
        "--demo-mode",
        action="store_true",
        help="Initialize with demo knowledge base"
    )
    
    return parser.parse_args(args)


# Global components
atomspace = None
metta_interpreter = None
cognitive_agent = None
rwkv_interface = None
background_tasks = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await initialize_opencog_system()
    
    # Start background cognitive processing
    background_task = asyncio.create_task(cognitive_processing_loop())
    background_tasks.append(background_task)
    
    yield
    
    # Shutdown
    for task in background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def initialize_opencog_system():
    """Initialize the OpenCog AGI system"""
    global atomspace, metta_interpreter, cognitive_agent, rwkv_interface
    
    print("Initializing OpenCog-RWKV AGI System...")
    
    # Initialize core components
    atomspace = AtomSpace()
    print(f"✓ Atomspace initialized")
    
    # Initialize RWKV interface
    rwkv_interface = RWKVCognitiveInterface()
    await rwkv_interface.initialize()
    print(f"✓ RWKV interface initialized")
    
    # Initialize MeTTa interpreter
    metta_interpreter = MeTTaInterpreter(atomspace, rwkv_interface)
    print(f"✓ MeTTa interpreter initialized")
    
    # Initialize cognitive agent
    cognitive_agent = CognitiveAgent(
        "main_opencog_agent",
        atomspace,
        metta_interpreter,
        rwkv_interface
    )
    print(f"✓ Cognitive agent initialized")
    
    # Initialize demo knowledge if requested
    if hasattr(get_args(), 'demo_mode') and get_args().demo_mode:
        await initialize_demo_knowledge()
        print(f"✓ Demo knowledge base loaded")
    
    print("OpenCog-RWKV AGI System ready!")


async def initialize_demo_knowledge():
    """Initialize demo knowledge base for testing"""
    # Create fundamental concepts
    concepts = [
        ("Artificial Intelligence", 0.9, 0.8),
        ("Machine Learning", 0.8, 0.7),
        ("Neural Networks", 0.8, 0.7),
        ("RWKV", 0.9, 0.8),
        ("OpenCog", 0.9, 0.8),
        ("AGI", 0.8, 0.7),
        ("Reasoning", 0.7, 0.6),
        ("Knowledge", 0.8, 0.7),
        ("Language Model", 0.8, 0.7),
        ("Cognitive Architecture", 0.8, 0.7)
    ]
    
    concept_atoms = {}
    for name, truth, confidence in concepts:
        atom = atomspace.create_atom(
            AtomType.CONCEPT, name, truth, confidence
        )
        concept_atoms[name] = atom
    
    # Create relationships
    relationships = [
        ("RWKV", "Neural Networks", 0.9),
        ("RWKV", "Language Model", 0.9),
        ("OpenCog", "AGI", 0.9),
        ("OpenCog", "Cognitive Architecture", 0.9),
        ("Machine Learning", "Artificial Intelligence", 0.8),
        ("Neural Networks", "Machine Learning", 0.8),
        ("AGI", "Artificial Intelligence", 0.9),
        ("Reasoning", "AGI", 0.8),
        ("Knowledge", "AGI", 0.8)
    ]
    
    for source, target, strength in relationships:
        if source in concept_atoms and target in concept_atoms:
            atomspace.create_link(
                AtomType.INHERITANCE,
                concept_atoms[source].id,
                concept_atoms[target].id,
                strength
            )
    
    # Add some initial goals for demonstration
    demo_goals = [
        Goal(
            id="demo_goal_1",
            type=GoalType.EXPLORE_KNOWLEDGE,
            description="Explore relationships between AI concepts",
            parameters={"topic": "Artificial Intelligence", "depth": 3},
            priority=0.6
        ),
        Goal(
            id="demo_goal_2", 
            type=GoalType.LEARN_CONCEPT,
            description="Learn about neural-symbolic integration",
            parameters={
                "concept": "Neural-Symbolic Integration",
                "examples": ["RWKV+OpenCog", "Neural+Logic", "Connectionist+Symbolic"]
            },
            priority=0.7
        )
    ]
    
    for goal in demo_goals:
        await cognitive_agent.add_goal(goal)


async def cognitive_processing_loop():
    """Background cognitive processing loop"""
    while True:
        try:
            # Process agent goals
            await cognitive_agent.process_goals()
            
            # Perform attention allocation
            await update_attention_allocation()
            
            # Sleep based on cognitive cycles setting
            await asyncio.sleep(1.0 / 10)  # 10 cycles per second default
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in cognitive processing loop: {e}")
            await asyncio.sleep(1.0)


async def update_attention_allocation():
    """Update attention allocation across atoms"""
    # Simple attention decay
    for atom in atomspace.atoms.values():
        if atom.attention_value > 0.1:
            # Decay attention over time
            new_attention = atom.attention_value * 0.99
            atomspace.update_attention(atom.id, new_attention)


# Create FastAPI application
app = FastAPI(
    title="OpenCog-RWKV AGI Backend",
    description="Neural-symbolic AGI backend combining RWKV and OpenCog",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(opencog_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "OpenCog-RWKV AGI Backend Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "api": "/api/v1/opencog/",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        system_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "atomspace": len(atomspace.atoms) if atomspace else 0,
                "cognitive_agent": cognitive_agent.state.value if cognitive_agent else "not_initialized",
                "rwkv_interface": "initialized" if rwkv_interface else "not_initialized",
                "metta_interpreter": "ready" if metta_interpreter else "not_initialized"
            }
        }
        
        return JSONResponse(content=system_status, status_code=200)
        
    except Exception as e:
        # Log the actual error for debugging
        print(f"Health check error: {type(e).__name__}: {str(e)}")
        
        error_status = {
            "status": "unhealthy", 
            "timestamp": time.time(),
            "error": "System health check failed"
        }
        return JSONResponse(content=error_status, status_code=503)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    # Log the actual error for debugging
    print(f"Internal error: {type(exc).__name__}: {str(exc)}")
    
    # Return sanitized error to user
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )


if __name__ == "__main__":
    args = get_args()
    
    print(f"Starting OpenCog-RWKV Backend Server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    
    # Set environment variables
    os.environ["OPENCOG_RWKV_PORT"] = str(args.port)
    os.environ["OPENCOG_RWKV_HOST"] = args.host
    
    if args.debug:
        os.environ["OPENCOG_RWKV_DEBUG"] = "1"
    
    # Run server
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.debug else "info",
        workers=1  # Single worker for proper state management
    )