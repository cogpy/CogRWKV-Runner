"""
Cognitive Agent implementation for OpenCog-RWKV integration.
Provides autonomous reasoning and decision-making capabilities.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

from .atomspace import AtomSpace, Atom, AtomType
from .metta_bridge import MeTTaInterpreter, MeTTaExpression


class AgentState(Enum):
    """Cognitive agent states"""
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    REASONING = "reasoning"
    COMMUNICATING = "communicating"
    ERROR = "error"


class GoalType(Enum):
    """Types of agent goals"""
    GENERATE_TEXT = "generate_text"
    ANSWER_QUESTION = "answer_question"
    LEARN_CONCEPT = "learn_concept"
    SOLVE_PROBLEM = "solve_problem"
    EXPLORE_KNOWLEDGE = "explore_knowledge"
    MAINTAIN_COHERENCE = "maintain_coherence"


@dataclass
class Goal:
    """Agent goal representation"""
    id: str
    type: GoalType
    description: str
    parameters: Dict[str, Any]
    priority: float = 0.5
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    progress: float = 0.0
    status: str = "pending"


@dataclass
class CognitiveStep:
    """Individual cognitive processing step"""
    id: str
    agent_id: str
    step_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CognitiveAgent:
    """
    Autonomous cognitive agent for AGI processing.
    Integrates RWKV neural processing with OpenCog symbolic reasoning.
    """
    
    def __init__(self, agent_id: str, atomspace: AtomSpace, 
                 metta_interpreter: MeTTaInterpreter, rwkv_interface=None):
        self.agent_id = agent_id
        self.atomspace = atomspace
        self.metta_interpreter = metta_interpreter
        self.rwkv_interface = rwkv_interface
        
        # Agent state
        self.state = AgentState.IDLE
        self.goals: List[Goal] = []
        self.memory: Dict[str, Any] = {}
        self.beliefs: Dict[str, float] = {}  # Belief strengths
        self.attention_focus: List[str] = []  # Atom IDs under focus
        
        # Performance metrics
        self.cognitive_steps: List[CognitiveStep] = []
        self.learning_rate = 0.1
        self.confidence_threshold = 0.6
        
        # Cognitive processes
        self.reasoning_strategies: Dict[str, Callable] = {
            "forward_chaining": self._forward_chaining,
            "backward_chaining": self._backward_chaining,
            "abductive_reasoning": self._abductive_reasoning,
            "analogy_reasoning": self._analogy_reasoning
        }
        
        # Initialize basic concepts
        self._initialize_basic_concepts()
    
    def _initialize_basic_concepts(self):
        """Initialize basic cognitive concepts in atomspace"""
        # Create fundamental concepts
        self_concept = self.atomspace.create_atom(
            AtomType.CONCEPT, f"Agent_{self.agent_id}", 1.0, 0.9
        )
        
        knowledge_concept = self.atomspace.create_atom(
            AtomType.CONCEPT, "Knowledge", 0.8, 0.7
        )
        
        reasoning_concept = self.atomspace.create_atom(
            AtomType.CONCEPT, "Reasoning", 0.8, 0.7
        )
        
        # Create relationships
        self.atomspace.create_link(
            AtomType.INHERITANCE, self_concept.id, knowledge_concept.id, 0.9
        )
        
        # Initialize memory structures
        self.memory["self_id"] = self_concept.id
        self.memory["session_start"] = datetime.utcnow()
        self.memory["interactions"] = []
    
    async def add_goal(self, goal: Goal):
        """Add a new goal to the agent"""
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)
        
        # Create goal atom in atomspace
        goal_atom = self.atomspace.create_atom(
            AtomType.GOAL, f"Goal_{goal.id}", goal.priority, 0.8
        )
        
        # Link goal to agent
        if "self_id" in self.memory:
            self.atomspace.create_link(
                AtomType.EVALUATION, self.memory["self_id"], goal_atom.id, 0.9
            )
    
    async def process_goals(self):
        """Process pending goals"""
        if self.state != AgentState.IDLE:
            return
        
        self.state = AgentState.PROCESSING
        
        try:
            for goal in self.goals[:]:  # Copy list to allow modification
                if goal.status != "pending":
                    continue
                
                # Check if goal is expired
                if goal.deadline and datetime.utcnow() > goal.deadline:
                    goal.status = "expired"
                    continue
                
                # Process goal based on type
                await self._process_goal(goal)
                
                # Remove completed or failed goals
                if goal.status in ["completed", "failed", "expired"]:
                    self.goals.remove(goal)
        
        finally:
            self.state = AgentState.IDLE
    
    async def _process_goal(self, goal: Goal):
        """Process a specific goal"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if goal.type == GoalType.GENERATE_TEXT:
                result = await self._handle_text_generation(goal)
            elif goal.type == GoalType.ANSWER_QUESTION:
                result = await self._handle_question_answering(goal)
            elif goal.type == GoalType.LEARN_CONCEPT:
                result = await self._handle_concept_learning(goal)
            elif goal.type == GoalType.SOLVE_PROBLEM:
                result = await self._handle_problem_solving(goal)
            elif goal.type == GoalType.EXPLORE_KNOWLEDGE:
                result = await self._handle_knowledge_exploration(goal)
            else:
                result = {"error": "Unknown goal type"}
            
            goal.status = "completed" if result.get("success", False) else "failed"
            goal.progress = 1.0
            
            # Record cognitive step
            step = CognitiveStep(
                id=f"step_{len(self.cognitive_steps)}",
                agent_id=self.agent_id,
                step_type=f"goal_{goal.type.value}",
                input_data=goal.parameters,
                output_data=result,
                execution_time=asyncio.get_event_loop().time() - start_time,
                success=result.get("success", False)
            )
            
            self.cognitive_steps.append(step)
        
        except Exception as e:
            goal.status = "failed"
            print(f"Error processing goal {goal.id}: {str(e)}")
    
    async def _handle_text_generation(self, goal: Goal) -> Dict[str, Any]:
        """Handle text generation goal"""
        prompt = goal.parameters.get("prompt", "")
        max_tokens = goal.parameters.get("max_tokens", 100)
        
        if not self.rwkv_interface:
            return {"success": False, "error": "No RWKV interface available"}
        
        # Enhance prompt with knowledge from atomspace
        enhanced_prompt = await self._enhance_prompt_with_knowledge(prompt)
        
        # Generate text using RWKV
        metta_code = f'(rwkv-generate "{enhanced_prompt}" {max_tokens} 1.0)'
        result = await self.metta_interpreter.evaluate(metta_code)
        
        # Learn from generation
        await self._learn_from_generation(prompt, result)
        
        return {
            "success": True,
            "generated_text": result,
            "enhanced_prompt": enhanced_prompt
        }
    
    async def _handle_question_answering(self, goal: Goal) -> Dict[str, Any]:
        """Handle question answering goal"""
        question = goal.parameters.get("question", "")
        
        # Parse question to identify key concepts
        key_concepts = await self._extract_concepts_from_text(question)
        
        # Search for relevant knowledge
        relevant_knowledge = []
        for concept in key_concepts:
            atoms = self.atomspace.find_atoms_by_name(concept)
            relevant_knowledge.extend(atoms)
        
        # Use reasoning to formulate answer
        reasoning_result = await self._reason_about_question(question, relevant_knowledge)
        
        # Generate answer using RWKV
        if reasoning_result.get("answer_prompt"):
            answer = await self.metta_interpreter.evaluate(
                f'(rwkv-generate "{reasoning_result["answer_prompt"]}" 150 0.8)'
            )
        else:
            answer = "I don't have sufficient knowledge to answer this question."
        
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "key_concepts": key_concepts,
            "reasoning_steps": reasoning_result.get("steps", [])
        }
    
    async def _handle_concept_learning(self, goal: Goal) -> Dict[str, Any]:
        """Handle concept learning goal"""
        concept_name = goal.parameters.get("concept", "")
        examples = goal.parameters.get("examples", [])
        
        # Create or update concept in atomspace
        concept_atom = self.atomspace.create_atom(
            AtomType.CONCEPT, concept_name, 0.7, 0.6
        )
        
        # Process examples to strengthen concept
        for example in examples:
            example_atom = self.atomspace.create_atom(
                AtomType.CONCEPT, f"Example_{example}", 0.6, 0.5
            )
            
            # Create inheritance link
            self.atomspace.create_link(
                AtomType.INHERITANCE, example_atom.id, concept_atom.id, 0.8
            )
        
        # Update belief about concept
        self.beliefs[concept_name] = min(1.0, self.beliefs.get(concept_name, 0.5) + 0.1)
        
        return {
            "success": True,
            "concept": concept_name,
            "learned_examples": len(examples),
            "belief_strength": self.beliefs[concept_name]
        }
    
    async def _handle_problem_solving(self, goal: Goal) -> Dict[str, Any]:
        """Handle problem solving goal"""
        problem = goal.parameters.get("problem", "")
        
        # Decompose problem into subproblems
        subproblems = await self._decompose_problem(problem)
        
        # Apply reasoning strategies
        solutions = []
        for strategy_name, strategy_func in self.reasoning_strategies.items():
            try:
                solution = await strategy_func(problem, subproblems)
                if solution:
                    solutions.append({
                        "strategy": strategy_name,
                        "solution": solution,
                        "confidence": solution.get("confidence", 0.5)
                    })
            except Exception as e:
                print(f"Error in {strategy_name}: {str(e)}")
        
        # Select best solution
        best_solution = max(solutions, key=lambda s: s["confidence"]) if solutions else None
        
        return {
            "success": best_solution is not None,
            "problem": problem,
            "solutions": solutions,
            "best_solution": best_solution
        }
    
    async def _handle_knowledge_exploration(self, goal: Goal) -> Dict[str, Any]:
        """Handle knowledge exploration goal"""
        topic = goal.parameters.get("topic", "")
        depth = goal.parameters.get("depth", 3)
        
        # Start from topic concept
        topic_atoms = self.atomspace.find_atoms_by_name(topic)
        if not topic_atoms:
            return {"success": False, "error": f"Topic '{topic}' not found in knowledge base"}
        
        # Explore connected concepts
        explored_concepts = set()
        exploration_queue = [(topic_atoms[0].id, 0)]
        exploration_results = []
        
        while exploration_queue and len(explored_concepts) < 20:  # Limit exploration
            atom_id, current_depth = exploration_queue.pop(0)
            
            if current_depth >= depth or atom_id in explored_concepts:
                continue
            
            explored_concepts.add(atom_id)
            atom = self.atomspace.get_atom_by_id(atom_id)
            
            if atom:
                exploration_results.append({
                    "concept": atom.name,
                    "truth_value": atom.truth_value,
                    "attention": atom.attention_value,
                    "depth": current_depth
                })
                
                # Add neighbors to queue
                neighbors = self.atomspace.get_neighbors(atom_id)
                for neighbor_id in neighbors:
                    exploration_queue.append((neighbor_id, current_depth + 1))
        
        return {
            "success": True,
            "topic": topic,
            "explored_concepts": exploration_results,
            "total_explored": len(explored_concepts)
        }
    
    async def _enhance_prompt_with_knowledge(self, prompt: str) -> str:
        """Enhance prompt with relevant knowledge from atomspace"""
        # Extract key concepts from prompt
        concepts = await self._extract_concepts_from_text(prompt)
        
        # Find related knowledge
        context_info = []
        for concept in concepts[:3]:  # Limit to top 3 concepts
            atoms = self.atomspace.find_atoms_by_name(concept)
            if atoms:
                context_info.append(f"Knowledge about {concept}: {atoms[0].name}")
        
        if context_info:
            enhanced = f"Context: {'; '.join(context_info)}\n\nUser: {prompt}"
        else:
            enhanced = prompt
        
        return enhanced
    
    async def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction - could be improved with NLP
        words = text.lower().split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        concepts = [word.strip(".,!?") for word in words if word not in stop_words and len(word) > 2]
        
        return concepts[:5]  # Return top 5 concepts
    
    async def _reason_about_question(self, question: str, knowledge: List[Atom]) -> Dict[str, Any]:
        """Apply reasoning to answer a question"""
        reasoning_steps = []
        
        # Simple reasoning based on available knowledge
        if knowledge:
            reasoning_steps.append("Found relevant concepts in knowledge base")
            
            # Create answer prompt based on knowledge
            knowledge_context = "; ".join([atom.name for atom in knowledge[:3]])
            answer_prompt = f"Based on knowledge about {knowledge_context}, answer: {question}"
            
            return {
                "steps": reasoning_steps,
                "answer_prompt": answer_prompt,
                "confidence": 0.7
            }
        
        reasoning_steps.append("No relevant knowledge found")
        return {
            "steps": reasoning_steps,
            "confidence": 0.3
        }
    
    async def _decompose_problem(self, problem: str) -> List[str]:
        """Decompose complex problem into simpler subproblems"""
        # Simple decomposition - could be enhanced
        if "and" in problem.lower():
            return problem.lower().split("and")
        elif "or" in problem.lower():
            return problem.lower().split("or")
        else:
            return [problem]
    
    async def _forward_chaining(self, problem: str, subproblems: List[str]) -> Optional[Dict[str, Any]]:
        """Forward chaining inference"""
        # Start with known facts and derive new ones
        facts = self.atomspace.find_atoms_by_type(AtomType.CONCEPT)
        
        if facts:
            return {
                "type": "forward_chaining",
                "derived_facts": [f.name for f in facts[:3]],
                "confidence": 0.6
            }
        
        return None
    
    async def _backward_chaining(self, problem: str, subproblems: List[str]) -> Optional[Dict[str, Any]]:
        """Backward chaining inference"""
        # Start with goal and work backwards
        goal_concepts = await self._extract_concepts_from_text(problem)
        
        if goal_concepts:
            return {
                "type": "backward_chaining",
                "goal_concepts": goal_concepts,
                "confidence": 0.5
            }
        
        return None
    
    async def _abductive_reasoning(self, problem: str, subproblems: List[str]) -> Optional[Dict[str, Any]]:
        """Abductive reasoning - inference to best explanation"""
        explanations = []
        
        # Generate possible explanations
        for subproblem in subproblems:
            explanations.append(f"Explanation for: {subproblem.strip()}")
        
        if explanations:
            return {
                "type": "abductive_reasoning",
                "explanations": explanations,
                "confidence": 0.4
            }
        
        return None
    
    async def _analogy_reasoning(self, problem: str, subproblems: List[str]) -> Optional[Dict[str, Any]]:
        """Analogy-based reasoning"""
        # Find similar problems in memory
        similar_cases = [step for step in self.cognitive_steps 
                        if "problem" in step.input_data and 
                        any(word in step.input_data.get("problem", "") 
                            for word in problem.split()[:3])]
        
        if similar_cases:
            return {
                "type": "analogy_reasoning",
                "similar_cases": len(similar_cases),
                "confidence": 0.6
            }
        
        return None
    
    async def _learn_from_generation(self, prompt: str, generated_text: str):
        """Learn from text generation experience"""
        # Create association between prompt concepts and generated concepts
        prompt_concepts = await self._extract_concepts_from_text(prompt)
        generated_concepts = await self._extract_concepts_from_text(generated_text)
        
        for p_concept in prompt_concepts:
            for g_concept in generated_concepts:
                # Create or strengthen association
                p_atoms = self.atomspace.find_atoms_by_name(p_concept)
                g_atoms = self.atomspace.find_atoms_by_name(g_concept)
                
                if not p_atoms:
                    p_atom = self.atomspace.create_atom(AtomType.CONCEPT, p_concept, 0.6, 0.5)
                else:
                    p_atom = p_atoms[0]
                
                if not g_atoms:
                    g_atom = self.atomspace.create_atom(AtomType.CONCEPT, g_concept, 0.6, 0.5)
                else:
                    g_atom = g_atoms[0]
                
                # Create similarity link
                self.atomspace.create_link(
                    AtomType.SIMILARITY, p_atom.id, g_atom.id, 0.7
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "active_goals": len([g for g in self.goals if g.status == "pending"]),
            "total_goals": len(self.goals),
            "cognitive_steps": len(self.cognitive_steps),
            "atomspace_size": len(self.atomspace.atoms),
            "beliefs": len(self.beliefs),
            "attention_focus": len(self.attention_focus)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        successful_steps = [step for step in self.cognitive_steps if step.success]
        
        return {
            "total_steps": len(self.cognitive_steps),
            "successful_steps": len(successful_steps),
            "success_rate": len(successful_steps) / max(len(self.cognitive_steps), 1),
            "average_execution_time": np.mean([step.execution_time for step in self.cognitive_steps]) if self.cognitive_steps else 0,
            "learning_rate": self.learning_rate,
            "confidence_threshold": self.confidence_threshold
        }