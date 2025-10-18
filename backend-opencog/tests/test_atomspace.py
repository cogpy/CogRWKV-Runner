"""
Tests for OpenCog Atomspace implementation.
"""

import pytest
import asyncio
from datetime import datetime

from core.atomspace import AtomSpace, Atom, AtomType


class TestAtomSpace:
    """Test cases for AtomSpace"""
    
    def setup_method(self):
        """Setup test atomspace"""
        self.atomspace = AtomSpace()
    
    def test_create_concept_atom(self):
        """Test creating a concept atom"""
        atom = self.atomspace.create_atom(
            AtomType.CONCEPT, 
            "TestConcept", 
            truth_value=0.8, 
            confidence=0.9
        )
        
        assert atom.type == AtomType.CONCEPT
        assert atom.name == "TestConcept"
        assert atom.truth_value == 0.8
        assert atom.confidence == 0.9
        assert atom.id in self.atomspace.atoms
    
    def test_create_link(self):
        """Test creating links between atoms"""
        # Create source and target atoms
        source = self.atomspace.create_atom(AtomType.CONCEPT, "Source", 0.7, 0.6)
        target = self.atomspace.create_atom(AtomType.CONCEPT, "Target", 0.8, 0.7)
        
        # Create link
        link = self.atomspace.create_link(
            AtomType.INHERITANCE, 
            source.id, 
            target.id, 
            truth_value=0.9
        )
        
        assert link is not None
        assert link.type == AtomType.INHERITANCE
        assert link.id in source.outgoing
        assert link.id in target.incoming
    
    def test_find_atoms_by_type(self):
        """Test finding atoms by type"""
        # Create multiple concepts
        for i in range(5):
            self.atomspace.create_atom(AtomType.CONCEPT, f"Concept_{i}")
        
        # Create a predicate
        self.atomspace.create_atom(AtomType.PREDICATE, "TestPredicate")
        
        concepts = self.atomspace.find_atoms_by_type(AtomType.CONCEPT)
        predicates = self.atomspace.find_atoms_by_type(AtomType.PREDICATE)
        
        assert len(concepts) == 5
        assert len(predicates) == 1
    
    def test_find_atoms_by_name(self):
        """Test finding atoms by name pattern"""
        self.atomspace.create_atom(AtomType.CONCEPT, "Neural Network")
        self.atomspace.create_atom(AtomType.CONCEPT, "Neural Processing")
        self.atomspace.create_atom(AtomType.CONCEPT, "Machine Learning")
        
        neural_atoms = self.atomspace.find_atoms_by_name("neural")
        
        assert len(neural_atoms) == 2
        assert all("neural" in atom.name.lower() for atom in neural_atoms)
    
    def test_update_truth_value(self):
        """Test updating atom truth values"""
        atom = self.atomspace.create_atom(AtomType.CONCEPT, "TestAtom", 0.5, 0.5)
        original_updated = atom.updated_at
        
        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.01)
        
        self.atomspace.update_truth_value(atom.id, 0.8, 0.9)
        
        updated_atom = self.atomspace.get_atom_by_id(atom.id)
        assert updated_atom.truth_value == 0.8
        assert updated_atom.confidence == 0.9
        assert updated_atom.updated_at > original_updated
    
    def test_attention_management(self):
        """Test attention value updates"""
        atom = self.atomspace.create_atom(AtomType.CONCEPT, "AttentionTest")
        
        self.atomspace.update_attention(atom.id, 0.7)
        
        updated_atom = self.atomspace.get_atom_by_id(atom.id)
        assert updated_atom.attention_value == 0.7
        assert self.atomspace.attention_bank[atom.id] == 0.7
    
    def test_get_neighbors(self):
        """Test getting neighboring atoms"""
        # Create atom network
        center = self.atomspace.create_atom(AtomType.CONCEPT, "Center")
        neighbor1 = self.atomspace.create_atom(AtomType.CONCEPT, "Neighbor1")
        neighbor2 = self.atomspace.create_atom(AtomType.CONCEPT, "Neighbor2")
        
        # Create links
        self.atomspace.create_link(AtomType.INHERITANCE, center.id, neighbor1.id)
        self.atomspace.create_link(AtomType.SIMILARITY, neighbor2.id, center.id)
        
        neighbors = self.atomspace.get_neighbors(center.id)
        
        assert len(neighbors) >= 2  # Should include both neighbors
        assert neighbor1.id in neighbors or neighbor2.id in neighbors
    
    def test_pattern_match(self):
        """Test pattern matching"""
        # Create test atoms
        self.atomspace.create_atom(AtomType.CONCEPT, "AI", 0.9, 0.8)
        self.atomspace.create_atom(AtomType.CONCEPT, "ML", 0.7, 0.6)
        self.atomspace.create_atom(AtomType.PREDICATE, "IsRelated", 0.5, 0.4)
        
        # Pattern: find concepts with truth value > 0.8
        pattern = {
            "type": "ConceptNode",
            "truth_value_gt": 0.8
        }
        
        matches = self.atomspace.pattern_match(pattern)
        
        assert len(matches) >= 1
        assert any(match["atom_id"] for match in matches)
    
    def test_export_import(self):
        """Test atomspace export and import"""
        # Create test data
        atom1 = self.atomspace.create_atom(AtomType.CONCEPT, "ExportTest1", 0.8, 0.7)
        atom2 = self.atomspace.create_atom(AtomType.CONCEPT, "ExportTest2", 0.9, 0.8)
        link = self.atomspace.create_link(AtomType.INHERITANCE, atom1.id, atom2.id, 0.85)
        
        # Export
        exported_data = self.atomspace.export_to_dict()
        
        # Create new atomspace and import
        new_atomspace = AtomSpace()
        new_atomspace.import_from_dict(exported_data)
        
        # Verify import
        assert len(new_atomspace.atoms) == len(self.atomspace.atoms)
        
        # Find exported atoms in new atomspace
        imported_atoms = new_atomspace.find_atoms_by_name("ExportTest")
        assert len(imported_atoms) == 2
    
    def test_cognitive_merge(self):
        """Test merging knowledge from another atomspace"""
        # Create second atomspace with overlapping knowledge
        other_atomspace = AtomSpace()
        
        # Common concept with different truth values
        self.atomspace.create_atom(AtomType.CONCEPT, "SharedConcept", 0.6, 0.5)
        other_atomspace.create_atom(AtomType.CONCEPT, "SharedConcept", 0.8, 0.7)
        
        # Unique concepts
        other_atomspace.create_atom(AtomType.CONCEPT, "UniqueConcept", 0.9, 0.8)
        
        original_count = len(self.atomspace.atoms)
        
        # Perform merge
        self.atomspace.cognitive_merge(other_atomspace)
        
        # Verify merge results
        merged_atoms = self.atomspace.find_atoms_by_name("SharedConcept")
        unique_atoms = self.atomspace.find_atoms_by_name("UniqueConcept")
        
        assert len(merged_atoms) == 1  # Should merge, not duplicate
        assert len(unique_atoms) == 1  # Should add unique concept
        assert len(self.atomspace.atoms) > original_count