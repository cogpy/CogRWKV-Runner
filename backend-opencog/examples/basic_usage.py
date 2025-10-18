"""
Basic usage examples for OpenCog-RWKV backend.
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any


class OpenCogRWKVClient:
    """Client for interacting with OpenCog-RWKV backend"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8001/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to backend"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def get(self, endpoint: str) -> Dict[str, Any]:
        """Make GET request to backend"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def example_basic_cognitive_processing():
    """Example: Basic cognitive processing"""
    print("=== Basic Cognitive Processing ===")
    
    client = OpenCogRWKVClient()
    
    # Test system status
    try:
        status = client.get("/opencog/status")
        print(f"System Status: {status['status']}")
        print(f"Atomspace Size: {status['atomspace_size']}")
    except Exception as e:
        print(f"Error getting status: {e}")
        return
    
    # Process a question through cognitive architecture
    request = {
        "prompt": "What is the relationship between neural networks and artificial intelligence?",
        "max_tokens": 200,
        "temperature": 0.8,
        "use_reasoning": True,
        "reasoning_depth": 3
    }
    
    print(f"\nProcessing: {request['prompt']}")
    
    try:
        result = client.post("/opencog/cognitive/process", request)
        
        print(f"Response: {result['output']}")
        print(f"Goal ID: {result['goal_id']}")
        print(f"Agent State: {result['agent_state']}")
        
    except Exception as e:
        print(f"Error in cognitive processing: {e}")


def example_knowledge_base_operations():
    """Example: Knowledge base operations"""
    print("\n=== Knowledge Base Operations ===")
    
    client = OpenCogRWKVClient()
    
    # Create concepts
    concepts = [
        {"type": "CONCEPT", "name": "Deep Learning", "truth_value": 0.9, "confidence": 0.8},
        {"type": "CONCEPT", "name": "Transformer Architecture", "truth_value": 0.85, "confidence": 0.75},
        {"type": "CONCEPT", "name": "Attention Mechanism", "truth_value": 0.8, "confidence": 0.7}
    ]
    
    concept_ids = {}
    
    for concept in concepts:
        try:
            result = client.post("/opencog/atoms", concept)
            concept_ids[concept["name"]] = result["atom_id"]
            print(f"Created concept: {concept['name']} (ID: {result['atom_id']})")
        except Exception as e:
            print(f"Error creating concept {concept['name']}: {e}")
    
    # Create relationships
    if len(concept_ids) >= 2:
        relationships = [
            {
                "type": "INHERITANCE",
                "source_id": concept_ids["Attention Mechanism"],
                "target_id": concept_ids["Transformer Architecture"],
                "truth_value": 0.9
            },
            {
                "type": "INHERITANCE", 
                "source_id": concept_ids["Transformer Architecture"],
                "target_id": concept_ids["Deep Learning"],
                "truth_value": 0.85
            }
        ]
        
        for rel in relationships:
            try:
                result = client.post("/opencog/links", rel)
                print(f"Created link: {rel['type']} (ID: {result['link_id']})")
            except Exception as e:
                print(f"Error creating link: {e}")
    
    # Query knowledge
    query = {
        "query_type": "concepts",
        "parameters": {},
        "limit": 10
    }
    
    try:
        result = client.post("/opencog/knowledge/query", query)
        print(f"\nFound {result['count']} concepts:")
        for concept in result['results'][:5]:  # Show first 5
            print(f"  - {concept['name']} (truth: {concept['truth_value']:.2f})")
    except Exception as e:
        print(f"Error querying knowledge: {e}")


def example_metta_programming():
    """Example: MeTTa programming"""
    print("\n=== MeTTa Programming ===")
    
    client = OpenCogRWKVClient()
    
    # Define a MeTTa procedure
    try:
        result = client.post("/opencog/metta/define", {
            "name": "find-ai-concepts",
            "body": "(match (ConceptNode AI) atomspace)"
        })
        print(f"Defined MeTTa procedure: {result['procedure_name']}")
    except Exception as e:
        print(f"Error defining procedure: {e}")
    
    # Evaluate MeTTa expressions
    expressions = [
        '(knowledge-query "concepts")',
        '(cognitive-step "attention")',
        '(rwkv-generate "Explain neural networks" 50 0.8)',
        '(match (ConceptNode "AI") atomspace)'
    ]
    
    for expr in expressions:
        try:
            result = client.post("/opencog/metta/evaluate", {
                "expression": expr,
                "context": {}
            })
            print(f"Expression: {expr}")
            print(f"Result: {json.dumps(result['result'], indent=2)[:200]}...")
            print()
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")


def example_goal_based_processing():
    """Example: Goal-based cognitive processing"""
    print("\n=== Goal-Based Processing ===")
    
    client = OpenCogRWKVClient()
    
    # Add different types of goals
    goals = [
        {
            "type": "learn_concept",
            "description": "Learn about quantum computing",
            "parameters": {
                "concept": "Quantum Computing",
                "examples": ["qubit", "superposition", "quantum gate", "entanglement"]
            },
            "priority": 0.8
        },
        {
            "type": "answer_question",
            "description": "Answer question about AI safety",
            "parameters": {
                "question": "What are the main challenges in AI alignment and safety?"
            },
            "priority": 0.9
        },
        {
            "type": "explore_knowledge",
            "description": "Explore machine learning concepts",
            "parameters": {
                "topic": "Machine Learning",
                "depth": 3
            },
            "priority": 0.6
        }
    ]
    
    goal_ids = []
    
    for goal in goals:
        try:
            result = client.post("/opencog/agent/goals", goal)
            goal_ids.append(result["goal_id"])
            print(f"Added goal: {goal['description']} (ID: {result['goal_id']})")
        except Exception as e:
            print(f"Error adding goal: {e}")
    
    # Trigger goal processing
    try:
        result = client.post("/opencog/agent/process-goals", {})
        print(f"Goal processing triggered: {result['message']}")
        
        # Wait a bit for processing
        time.sleep(2)
        
        # Check goal status
        goals_status = client.get("/opencog/agent/goals")
        print(f"\nGoal Status:")
        for goal in goals_status['goals']:
            print(f"  - {goal['description'][:50]}... Status: {goal['status']}")
            
    except Exception as e:
        print(f"Error in goal processing: {e}")


def example_performance_monitoring():
    """Example: Performance monitoring"""
    print("\n=== Performance Monitoring ===")
    
    client = OpenCogRWKVClient()
    
    try:
        # Get agent performance metrics
        performance = client.get("/opencog/agent/performance")
        
        print("Agent Performance:")
        perf_data = performance['performance']
        print(f"  - Total cognitive steps: {perf_data['total_steps']}")
        print(f"  - Success rate: {perf_data['success_rate']:.2%}")
        print(f"  - Average execution time: {perf_data['average_execution_time']:.3f}s")
        
        print("\nAgent Status:")
        status_data = performance['status']
        print(f"  - Current state: {status_data['state']}")
        print(f"  - Active goals: {status_data['active_goals']}")
        print(f"  - Atomspace size: {status_data['atomspace_size']}")
        
        # Get cognitive trace
        trace = client.get("/opencog/metta/trace")
        print(f"\nCognitive trace has {trace['total_steps']} steps")
        
    except Exception as e:
        print(f"Error getting performance data: {e}")


def run_all_examples():
    """Run all examples"""
    print("OpenCog-RWKV Backend Examples")
    print("=" * 40)
    
    examples = [
        example_basic_cognitive_processing,
        example_knowledge_base_operations,
        example_metta_programming,
        example_goal_based_processing,
        example_performance_monitoring
    ]
    
    for example_func in examples:
        try:
            example_func()
            print("\n" + "-" * 40)
            time.sleep(1)  # Brief pause between examples
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
            print("-" * 40)
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    run_all_examples()