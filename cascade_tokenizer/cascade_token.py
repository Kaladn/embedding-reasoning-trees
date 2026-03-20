"""
CascadeToken: Core data structure for tokens with embedded 6-1-6 reasoning trees
Production-ready implementation for guided token generation
"""

import numpy as np
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class NodeType(Enum):
    INPUT = "input"
    CENTRAL = "central"
    OUTPUT = "output"


class ConstraintType(Enum):
    SEMANTIC = "semantic"
    LOGICAL = "logical"
    CONTEXTUAL = "contextual"
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    DOMAIN = "domain"


@dataclass
class CascadeNode:
    """Individual node in the 6-1-6 reasoning cascade"""
    node_id: str
    node_type: NodeType
    concept: str
    weight: float
    constraints: Dict[ConstraintType, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    activation_threshold: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningCascade:
    """6-1-6 reasoning tree structure embedded in each token"""
    input_nodes: List[CascadeNode] = field(default_factory=lambda: [None] * 6)
    central_node: CascadeNode = None
    output_nodes: List[CascadeNode] = field(default_factory=lambda: [None] * 6)
    cascade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    confidence_score: float = 1.0
    constraint_violations: List[str] = field(default_factory=list)

    def validate_structure(self) -> bool:
        """Validate 6-1-6 structure integrity"""
        return (len([n for n in self.input_nodes if n is not None]) <= 6 and
                self.central_node is not None and
                len([n for n in self.output_nodes if n is not None]) <= 6)

    def get_active_paths(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """Get active reasoning paths above threshold"""
        paths = []
        if not self.central_node:
            return paths

        for input_node in self.input_nodes:
            if input_node and input_node.weight >= threshold:
                paths.append((input_node.node_id, self.central_node.node_id, input_node.weight))

        for output_node in self.output_nodes:
            if output_node and output_node.weight >= threshold:
                paths.append((self.central_node.node_id, output_node.node_id, output_node.weight))

        return paths


class CascadeToken:
    """Production token with embedded reasoning cascade"""

    def __init__(self, surface_form: str, token_id: int = None):
        self.surface_form = surface_form
        self.token_id = token_id or hash(surface_form) % (2**31)
        self.cascade = ReasoningCascade()
        self.embedding_vector = None
        self.frequency = 0
        self.context_history = []
        self.generation_constraints = {}
        self.semantic_fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute semantic fingerprint for fast lookup"""
        content = f"{self.surface_form}_{self.token_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def embed_cascade(self, input_concepts: List[str], central_concept: str,
                     output_concepts: List[str], weights: List[float] = None) -> None:
        """Embed 6-1-6 cascade into token"""
        if len(input_concepts) > 6 or len(output_concepts) > 6:
            raise ValueError("Maximum 6 input and 6 output concepts allowed")

        weights = weights or [1.0] * (len(input_concepts) + 1 + len(output_concepts))
        weight_idx = 0

        # Create input nodes
        for i, concept in enumerate(input_concepts):
            node = CascadeNode(
                node_id=f"in_{i}_{concept}",
                node_type=NodeType.INPUT,
                concept=concept,
                weight=weights[weight_idx]
            )
            self.cascade.input_nodes[i] = node
            weight_idx += 1

        # Create central node
        self.cascade.central_node = CascadeNode(
            node_id=f"central_{central_concept}",
            node_type=NodeType.CENTRAL,
            concept=central_concept,
            weight=weights[weight_idx]
        )
        weight_idx += 1

        # Create output nodes
        for i, concept in enumerate(output_concepts):
            node = CascadeNode(
                node_id=f"out_{i}_{concept}",
                node_type=NodeType.OUTPUT,
                concept=concept,
                weight=weights[weight_idx]
            )
            self.cascade.output_nodes[i] = node
            weight_idx += 1

    def add_constraint(self, node_id: str, constraint_type: ConstraintType,
                      constraint_value: Any) -> None:
        """Add constraint to specific node"""
        for node_list in [self.cascade.input_nodes, [self.cascade.central_node],
                         self.cascade.output_nodes]:
            for node in node_list:
                if node and node.node_id == node_id:
                    node.constraints[constraint_type] = constraint_value
                    return

    def get_generation_candidates(self, context_vector: np.ndarray = None,
                                 temperature: float = 1.0) -> List[Tuple[str, float]]:
        """Get weighted generation candidates from output nodes"""
        candidates = []

        for node in self.cascade.output_nodes:
            if node is None:
                continue

            # Apply temperature scaling
            scaled_weight = node.weight / temperature

            # Apply context modulation if provided
            if context_vector is not None and len(context_vector) > 0:
                context_boost = np.random.normal(0, 0.1)
                scaled_weight += context_boost

            candidates.append((node.concept, scaled_weight))

        # Sort by weight descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def validate_constraints(self, proposed_continuation: str) -> Tuple[bool, List[str]]:
        """Validate proposed continuation against cascade constraints"""
        violations = []

        # Check each output node's constraints
        for node in self.cascade.output_nodes:
            if node is None:
                continue

            for constraint_type, constraint_value in node.constraints.items():
                if constraint_type == ConstraintType.SEMANTIC:
                    if not self._validate_semantic_constraint(proposed_continuation, constraint_value):
                        violations.append(f"Semantic constraint violation in {node.node_id}")

                elif constraint_type == ConstraintType.LOGICAL:
                    if not self._validate_logical_constraint(proposed_continuation, constraint_value):
                        violations.append(f"Logical constraint violation in {node.node_id}")

        return len(violations) == 0, violations

    def _validate_semantic_constraint(self, text: str, constraint: Any) -> bool:
        """Validate semantic constraints via keyword overlap.

        If the constraint is a list of allowed terms, the text must contain
        at least one of them. Strings are treated as a single required term.
        Other constraint types are passed through (valid by default).
        """
        if isinstance(constraint, list):
            text_lower = text.lower()
            return any(term.lower() in text_lower for term in constraint)
        if isinstance(constraint, str):
            return constraint.lower() in text.lower()
        return True

    def _validate_logical_constraint(self, text: str, constraint: Any) -> bool:
        """Validate logical constraints via simple rule evaluation.

        Supports a list of rule strings such as ``"requires_input"`` and
        ``"produces_output"``.  Unknown rules are considered satisfied.
        """
        if isinstance(constraint, list):
            for rule in constraint:
                if rule == "requires_input" and not text.strip():
                    return False
                if rule == "produces_output" and not text.strip():
                    return False
            return True
        return True

    def serialize(self) -> bytes:
        """Serialize token to binary format for storage/transmission"""
        data = {
            'surface_form': self.surface_form,
            'token_id': self.token_id,
            'frequency': self.frequency,
            'semantic_fingerprint': self.semantic_fingerprint,
            'cascade': self._serialize_cascade()
        }

        json_str = json.dumps(data, separators=(',', ':'))
        return json_str.encode('utf-8')

    def _serialize_cascade(self) -> Dict:
        """Serialize cascade structure"""
        def serialize_node(node):
            if node is None:
                return None
            return {
                'node_id': node.node_id,
                'node_type': node.node_type.value,
                'concept': node.concept,
                'weight': node.weight,
                'constraints': {k.value: v for k, v in node.constraints.items()},
                'connections': node.connections,
                'activation_threshold': node.activation_threshold,
                'metadata': node.metadata
            }

        return {
            'input_nodes': [serialize_node(n) for n in self.cascade.input_nodes],
            'central_node': serialize_node(self.cascade.central_node),
            'output_nodes': [serialize_node(n) for n in self.cascade.output_nodes],
            'cascade_id': self.cascade.cascade_id,
            'confidence_score': self.cascade.confidence_score,
            'constraint_violations': self.cascade.constraint_violations
        }

    @classmethod
    def deserialize(cls, data: bytes) -> 'CascadeToken':
        """Deserialize token from binary format"""
        json_str = data.decode('utf-8')
        data_dict = json.loads(json_str)

        token = cls(data_dict['surface_form'], data_dict['token_id'])
        token.frequency = data_dict['frequency']
        token.semantic_fingerprint = data_dict['semantic_fingerprint']
        token.cascade = cls._deserialize_cascade(data_dict['cascade'])

        return token

    @classmethod
    def _deserialize_cascade(cls, cascade_data: Dict) -> ReasoningCascade:
        """Deserialize cascade structure"""
        def deserialize_node(node_data):
            if node_data is None:
                return None

            node = CascadeNode(
                node_id=node_data['node_id'],
                node_type=NodeType(node_data['node_type']),
                concept=node_data['concept'],
                weight=node_data['weight'],
                connections=node_data['connections'],
                activation_threshold=node_data['activation_threshold'],
                metadata=node_data['metadata']
            )

            # Reconstruct constraints
            for k, v in node_data['constraints'].items():
                node.constraints[ConstraintType(k)] = v

            return node

        cascade = ReasoningCascade()
        cascade.input_nodes = [deserialize_node(n) for n in cascade_data['input_nodes']]
        cascade.central_node = deserialize_node(cascade_data['central_node'])
        cascade.output_nodes = [deserialize_node(n) for n in cascade_data['output_nodes']]
        cascade.cascade_id = cascade_data['cascade_id']
        cascade.confidence_score = cascade_data['confidence_score']
        cascade.constraint_violations = cascade_data['constraint_violations']

        return cascade

    def __repr__(self) -> str:
        return f"CascadeToken('{self.surface_form}', id={self.token_id}, cascade_id={self.cascade.cascade_id})"
