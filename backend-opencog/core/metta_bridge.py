"""
MeTTa programming language bridge for RWKV-Runner.
Provides self-modifying code capabilities and AGI reasoning.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .atomspace import AtomSpace, Atom, AtomType


class MeTTaOperator(Enum):
    """MeTTa operators for AGI reasoning"""
    MATCH = "match"
    BIND = "bind"
    EVAL = "eval"
    EXECUTE = "execute" 
    IF = "if"
    LET = "let"
    CASE = "case"
    SUPERPOSE = "superpose"
    COLLAPSE = "collapse"
    UNIFY = "unify"
    RWKV_GENERATE = "rwkv-generate"
    KNOWLEDGE_QUERY = "knowledge-query"
    COGNITIVE_STEP = "cognitive-step"


@dataclass
class MeTTaExpression:
    """MeTTa expression representation"""
    operator: str
    arguments: List[Union[str, 'MeTTaExpression']]
    bindings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.bindings is None:
            self.bindings = {}


class MeTTaInterpreter:
    """
    MeTTa language interpreter for AGI reasoning within RWKV-Runner.
    Handles self-modifying code and neural-symbolic integration.
    """
    
    def __init__(self, atomspace: AtomSpace, rwkv_interface=None):
        self.atomspace = atomspace
        self.rwkv_interface = rwkv_interface
        self.global_bindings: Dict[str, Any] = {}
        self.procedures: Dict[str, MeTTaExpression] = {}
        self.cognitive_steps: List[Dict[str, Any]] = []
        
    def parse_expression(self, metta_code: str) -> MeTTaExpression:
        """Parse MeTTa code string into expression tree"""
        # Simplified S-expression parser
        metta_code = metta_code.strip()
        
        if not metta_code.startswith('('):
            # Atomic expression
            return MeTTaExpression("atom", [metta_code])
        
        # Remove outer parentheses
        inner = metta_code[1:-1].strip()
        tokens = self._tokenize(inner)
        
        if not tokens:
            return MeTTaExpression("empty", [])
        
        operator = tokens[0]
        arguments = []
        
        i = 1
        while i < len(tokens):
            if tokens[i].startswith('('):
                # Nested expression - find matching parenthesis
                depth = 1
                j = i + 1
                nested_expr = tokens[i][1:]  # Remove opening paren
                
                while j < len(tokens) and depth > 0:
                    token = tokens[j]
                    if '(' in token:
                        depth += token.count('(')
                    if ')' in token:
                        depth -= token.count(')')
                    
                    if depth > 0:
                        nested_expr += ' ' + token
                    else:
                        nested_expr += ' ' + token[:-1]  # Remove closing paren
                    j += 1
                
                arguments.append(self.parse_expression(f"({nested_expr})"))
                i = j
            else:
                arguments.append(tokens[i])
                i += 1
        
        return MeTTaExpression(operator, arguments)
    
    def _tokenize(self, code: str) -> List[str]:
        """Tokenize MeTTa code"""
        # Simple tokenizer - could be improved
        tokens = []
        current_token = ""
        paren_depth = 0
        
        for char in code:
            if char in ' \t\n' and paren_depth == 0:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    async def evaluate(self, expression: Union[str, MeTTaExpression], 
                      context: Dict[str, Any] = None) -> Any:
        """Evaluate MeTTa expression"""
        if context is None:
            context = {}
        
        if isinstance(expression, str):
            expression = self.parse_expression(expression)
        
        return await self._evaluate_expression(expression, context)
    
    async def _evaluate_expression(self, expr: MeTTaExpression, 
                                 context: Dict[str, Any]) -> Any:
        """Internal expression evaluation"""
        operator = expr.operator
        args = expr.arguments
        
        # Built-in operators
        if operator == "match":
            return await self._eval_match(args, context)
        elif operator == "bind":
            return await self._eval_bind(args, context)
        elif operator == "eval":
            return await self._eval_eval(args, context)
        elif operator == "if":
            return await self._eval_if(args, context)
        elif operator == "let":
            return await self._eval_let(args, context)
        elif operator == "rwkv-generate":
            return await self._eval_rwkv_generate(args, context)
        elif operator == "knowledge-query":
            return await self._eval_knowledge_query(args, context)
        elif operator == "cognitive-step":
            return await self._eval_cognitive_step(args, context)
        elif operator == "superpose":
            return await self._eval_superpose(args, context)
        elif operator == "atom":
            return args[0] if args else None
        elif operator == "empty":
            return None
        else:
            # Check if it's a user-defined procedure
            if operator in self.procedures:
                return await self._eval_procedure(operator, args, context)
            else:
                # Try to evaluate as function call
                return await self._eval_function_call(operator, args, context)
    
    async def _eval_match(self, args: List[Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate match operation - pattern matching in atomspace"""
        if len(args) < 2:
            return []
        
        pattern = args[0] if isinstance(args[0], str) else str(args[0])
        space = args[1] if len(args) > 1 else "atomspace"
        
        # Convert pattern to atomspace query
        if pattern.startswith("(") and "ConceptNode" in pattern:
            # Extract concept name
            match = re.search(r'ConceptNode\s+([^)]+)', pattern)
            if match:
                concept_name = match.group(1).strip('"')
                atoms = self.atomspace.find_atoms_by_name(concept_name)
                return [{"atom": atom.id, "name": atom.name} for atom in atoms]
        
        return []
    
    async def _eval_bind(self, args: List[Any], context: Dict[str, Any]) -> Any:
        """Evaluate bind operation - variable binding"""
        if len(args) < 2:
            return None
        
        var_name = args[0]
        value_expr = args[1]
        
        # Evaluate value expression
        if isinstance(value_expr, MeTTaExpression):
            value = await self._evaluate_expression(value_expr, context)
        else:
            value = value_expr
        
        # Bind variable
        context[var_name] = value
        self.global_bindings[var_name] = value
        
        return value
    
    async def _eval_eval(self, args: List[Any], context: Dict[str, Any]) -> Any:
        """Evaluate eval operation - dynamic evaluation"""
        if not args:
            return None
        
        expr_to_eval = args[0]
        if isinstance(expr_to_eval, str):
            expr_to_eval = self.parse_expression(expr_to_eval)
        
        return await self._evaluate_expression(expr_to_eval, context)
    
    async def _eval_if(self, args: List[Any], context: Dict[str, Any]) -> Any:
        """Evaluate if conditional"""
        if len(args) < 2:
            return None
        
        condition = args[0]
        then_branch = args[1]
        else_branch = args[2] if len(args) > 2 else None
        
        # Evaluate condition
        if isinstance(condition, MeTTaExpression):
            cond_result = await self._evaluate_expression(condition, context)
        else:
            cond_result = condition
        
        # Choose branch
        if self._is_truthy(cond_result):
            if isinstance(then_branch, MeTTaExpression):
                return await self._evaluate_expression(then_branch, context)
            else:
                return then_branch
        elif else_branch:
            if isinstance(else_branch, MeTTaExpression):
                return await self._evaluate_expression(else_branch, context)
            else:
                return else_branch
        
        return None
    
    async def _eval_let(self, args: List[Any], context: Dict[str, Any]) -> Any:
        """Evaluate let binding with local scope"""
        if len(args) < 2:
            return None
        
        bindings = args[0]  # Should be list of [var, value] pairs
        body = args[1]
        
        # Create new context with local bindings
        new_context = context.copy()
        
        # Process bindings
        if isinstance(bindings, list):
            for i in range(0, len(bindings), 2):
                if i + 1 < len(bindings):
                    var = bindings[i]
                    val_expr = bindings[i + 1]
                    
                    if isinstance(val_expr, MeTTaExpression):
                        val = await self._evaluate_expression(val_expr, new_context)
                    else:
                        val = val_expr
                    
                    new_context[var] = val
        
        # Evaluate body with new context
        if isinstance(body, MeTTaExpression):
            return await self._evaluate_expression(body, new_context)
        else:
            return body
    
    async def _eval_rwkv_generate(self, args: List[Any], context: Dict[str, Any]) -> str:
        """Generate text using RWKV model"""
        if not self.rwkv_interface or not args:
            return ""
        
        prompt = args[0]
        max_tokens = int(args[1]) if len(args) > 1 else 100
        temperature = float(args[2]) if len(args) > 2 else 1.0
        
        # Use RWKV interface to generate text
        try:
            result = await self.rwkv_interface.generate(
                prompt=str(prompt),
                max_tokens=max_tokens,
                temperature=temperature
            )
            return result.get("text", "")
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _eval_knowledge_query(self, args: List[Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query knowledge from atomspace"""
        if not args:
            return []
        
        query_type = args[0]
        
        if query_type == "concepts":
            atoms = self.atomspace.find_atoms_by_type(AtomType.CONCEPT)
            return [{"id": atom.id, "name": atom.name, "truth": atom.truth_value} for atom in atoms]
        elif query_type == "related" and len(args) > 1:
            concept_name = args[1]
            atoms = self.atomspace.find_atoms_by_name(concept_name)
            if atoms:
                neighbors = self.atomspace.get_neighbors(atoms[0].id)
                return [{"id": nid} for nid in neighbors]
        
        return []
    
    async def _eval_cognitive_step(self, args: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a cognitive reasoning step"""
        step_type = args[0] if args else "default"
        
        step_result = {
            "type": step_type,
            "timestamp": asyncio.get_event_loop().time(),
            "context": context.copy(),
            "result": None
        }
        
        if step_type == "attention":
            # Focus attention on high-value atoms
            top_atoms = sorted(
                self.atomspace.atoms.values(),
                key=lambda a: a.attention_value,
                reverse=True
            )[:5]
            step_result["result"] = [{"id": atom.id, "attention": atom.attention_value} 
                                   for atom in top_atoms]
        
        elif step_type == "inference":
            # Simple inference step
            concepts = self.atomspace.find_atoms_by_type(AtomType.CONCEPT)
            if len(concepts) >= 2:
                # Create new inference link
                link = self.atomspace.create_link(
                    AtomType.SIMILARITY,
                    concepts[0].id,
                    concepts[1].id,
                    truth_value=0.6
                )
                step_result["result"] = {"new_link": link.id}
        
        self.cognitive_steps.append(step_result)
        return step_result
    
    async def _eval_superpose(self, args: List[Any], context: Dict[str, Any]) -> List[Any]:
        """Create quantum-like superposition of values"""
        results = []
        for arg in args:
            if isinstance(arg, MeTTaExpression):
                result = await self._evaluate_expression(arg, context)
            else:
                result = arg
            results.append(result)
        return results
    
    async def _eval_procedure(self, proc_name: str, args: List[Any], 
                            context: Dict[str, Any]) -> Any:
        """Evaluate user-defined procedure"""
        if proc_name not in self.procedures:
            return None
        
        proc = self.procedures[proc_name]
        # Simple procedure call - could be expanded
        return await self._evaluate_expression(proc, context)
    
    async def _eval_function_call(self, func_name: str, args: List[Any], 
                                context: Dict[str, Any]) -> Any:
        """Evaluate general function call"""
        # Check global bindings first
        if func_name in self.global_bindings:
            return self.global_bindings[func_name]
        
        # Check context
        if func_name in context:
            return context[func_name]
        
        # Default return
        return f"unknown_function({func_name})"
    
    def _is_truthy(self, value: Any) -> bool:
        """Check if value is truthy"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.lower() not in ["false", "", "nil", "null"]
        if isinstance(value, list):
            return len(value) > 0
        return True
    
    def define_procedure(self, name: str, body: str):
        """Define a new MeTTa procedure"""
        expr = self.parse_expression(body)
        self.procedures[name] = expr
    
    def get_cognitive_trace(self) -> List[Dict[str, Any]]:
        """Get trace of cognitive steps"""
        return self.cognitive_steps.copy()
    
    def clear_cognitive_trace(self):
        """Clear cognitive step trace"""
        self.cognitive_steps.clear()