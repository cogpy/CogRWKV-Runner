"""
OpenCog Atomspace implementation for RWKV-Runner AGI architecture.
Provides distributed knowledge representation and symbolic reasoning.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import networkx as nx
import numpy as np


class AtomType(Enum):
    """Atom types in the Atomspace"""
    CONCEPT = "ConceptNode"
    PREDICATE = "PredicateNode" 
    LINK = "Link"
    EVALUATION = "EvaluationLink"
    INHERITANCE = "InheritanceLink"
    SIMILARITY = "SimilarityLink"
    EXECUTION = "ExecutionLink"
    PROCEDURE = "ProcedureNode"
    VARIABLE = "VariableNode"
    RWKV_STATE = "RWKVStateNode"
    MEMORY = "MemoryNode"
    GOAL = "GoalNode"


@dataclass
class Atom:
    """Basic atom structure in the OpenCog Atomspace"""
    id: str
    type: AtomType
    name: str
    truth_value: float = 0.5  # Strength
    confidence: float = 0.5   # Confidence
    attention_value: float = 0.0
    incoming: Set[str] = None
    outgoing: Set[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.incoming is None:
            self.incoming = set()
        if self.outgoing is None:
            self.outgoing = set()
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class AtomSpace:
    """
    Distributed Atomspace for knowledge representation and reasoning.
    Integrates with RWKV for neural-symbolic processing.
    """
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}
        self.graph = nx.MultiDiGraph()  # For relationship tracking
        self.attention_bank: Dict[str, float] = {}
        self.rwkv_state_cache: Dict[str, Any] = {}
        
    def create_atom(self, atom_type: AtomType, name: str, 
                   truth_value: float = 0.5, confidence: float = 0.5) -> Atom:
        """Create a new atom in the Atomspace"""
        atom_id = str(uuid.uuid4())
        atom = Atom(
            id=atom_id,
            type=atom_type,
            name=name,
            truth_value=truth_value,
            confidence=confidence
        )
        
        self.atoms[atom_id] = atom
        self.graph.add_node(atom_id, atom=atom)
        return atom
    
    def create_link(self, link_type: AtomType, source_id: str, 
                   target_id: str, truth_value: float = 0.5) -> Optional[Atom]:
        """Create a link between two atoms"""
        if source_id not in self.atoms or target_id not in self.atoms:
            return None
            
        link_name = f"{link_type.value}({source_id}, {target_id})"
        link = self.create_atom(link_type, link_name, truth_value)
        
        # Update relationships
        self.atoms[source_id].outgoing.add(link.id)
        self.atoms[target_id].incoming.add(link.id)
        link.outgoing.add(source_id)
        link.outgoing.add(target_id)
        
        # Add to graph
        self.graph.add_edge(source_id, target_id, link_id=link.id)
        
        return link
    
    def find_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Find all atoms of a specific type"""
        return [atom for atom in self.atoms.values() if atom.type == atom_type]
    
    def find_atoms_by_name(self, name: str) -> List[Atom]:
        """Find atoms by name pattern"""
        return [atom for atom in self.atoms.values() if name.lower() in atom.name.lower()]
    
    def get_atom_by_id(self, atom_id: str) -> Optional[Atom]:
        """Retrieve atom by ID"""
        return self.atoms.get(atom_id)
    
    def update_truth_value(self, atom_id: str, truth_value: float, confidence: float = None):
        """Update atom's truth value and confidence"""
        if atom_id in self.atoms:
            atom = self.atoms[atom_id]
            atom.truth_value = truth_value
            if confidence is not None:
                atom.confidence = confidence
            atom.updated_at = datetime.utcnow()
    
    def update_attention(self, atom_id: str, attention_value: float):
        """Update attention value for an atom"""
        if atom_id in self.atoms:
            self.atoms[atom_id].attention_value = attention_value
            self.attention_bank[atom_id] = attention_value
    
    def get_neighbors(self, atom_id: str, direction: str = "both") -> List[str]:
        """Get neighboring atoms"""
        if atom_id not in self.graph:
            return []
            
        if direction == "incoming":
            return list(self.graph.predecessors(atom_id))
        elif direction == "outgoing":
            return list(self.graph.successors(atom_id))
        else:  # both
            return list(set(self.graph.predecessors(atom_id)) | 
                       set(self.graph.successors(atom_id)))
    
    def pattern_match(self, pattern: Dict[str, Any]) -> List[Dict[str, str]]:
        """Simple pattern matching in the Atomspace"""
        matches = []
        
        # This is a simplified pattern matcher
        # In a full implementation, this would use more sophisticated logic
        if "type" in pattern:
            target_type = AtomType(pattern["type"])
            candidate_atoms = self.find_atoms_by_type(target_type)
            
            for atom in candidate_atoms:
                match = {"atom_id": atom.id}
                
                # Check additional constraints
                if "name_contains" in pattern:
                    if pattern["name_contains"].lower() not in atom.name.lower():
                        continue
                
                if "truth_value_gt" in pattern:
                    if atom.truth_value <= pattern["truth_value_gt"]:
                        continue
                
                matches.append(match)
        
        return matches
    
    def cognitive_merge(self, other_atomspace: 'AtomSpace', merge_threshold: float = 0.7):
        """Merge knowledge from another atomspace (cognitive synergy)"""
        for atom in other_atomspace.atoms.values():
            # Check if similar atom exists
            similar_atoms = self.find_atoms_by_name(atom.name)
            
            if similar_atoms:
                # Merge truth values using weighted average
                existing_atom = similar_atoms[0]
                total_confidence = existing_atom.confidence + atom.confidence
                
                if total_confidence > 0:
                    new_truth = ((existing_atom.truth_value * existing_atom.confidence) +
                               (atom.truth_value * atom.confidence)) / total_confidence
                    
                    self.update_truth_value(existing_atom.id, new_truth, total_confidence)
            else:
                # Create new atom
                new_atom = self.create_atom(atom.type, atom.name, 
                                          atom.truth_value, atom.confidence)
                new_atom.metadata.update(atom.metadata)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export atomspace to dictionary format"""
        return {
            "atoms": {aid: {
                **asdict(atom),
                "incoming": list(atom.incoming),
                "outgoing": list(atom.outgoing),
                "created_at": atom.created_at.isoformat(),
                "updated_at": atom.updated_at.isoformat(),
                "type": atom.type.value
            } for aid, atom in self.atoms.items()},
            "attention_bank": self.attention_bank,
            "graph_edges": list(self.graph.edges(data=True))
        }
    
    def import_from_dict(self, data: Dict[str, Any]):
        """Import atomspace from dictionary format"""
        # Clear existing data
        self.atoms.clear()
        self.graph.clear()
        self.attention_bank.clear()
        
        # Import atoms
        for aid, atom_data in data.get("atoms", {}).items():
            atom = Atom(
                id=aid,
                type=AtomType(atom_data["type"]),
                name=atom_data["name"],
                truth_value=atom_data["truth_value"],
                confidence=atom_data["confidence"],
                attention_value=atom_data["attention_value"],
                incoming=set(atom_data["incoming"]),
                outgoing=set(atom_data["outgoing"]),
                created_at=datetime.fromisoformat(atom_data["created_at"]),
                updated_at=datetime.fromisoformat(atom_data["updated_at"]),
                metadata=atom_data["metadata"]
            )
            self.atoms[aid] = atom
            self.graph.add_node(aid, atom=atom)
        
        # Import graph edges
        for source, target, edge_data in data.get("graph_edges", []):
            self.graph.add_edge(source, target, **edge_data)
        
        # Import attention bank
        self.attention_bank.update(data.get("attention_bank", {}))