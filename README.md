# Cascade Tokenizer: Production-Ready AI with Embedded Reasoning Trees

This project implements Guided Token Generation, embedding deterministic 6-1-6 reasoning trees within each token for controlled, accountable AI generation.

## Key Features
- Hybrid symbolic/statistical token generation
- Embedded 6-1-6 reasoning trees for each token
- Deterministic constraints to prevent hallucination
- Traceable, accountable output
- Cascade-guided attention and constraint validation

## Main Classes & Parameters

### CascadeToken
- `surface_form: str` — Token text representation
- `token_id: int` — Unique identifier
- `cascade: ReasoningCascade` — Embedded reasoning tree
- `embedding_vector: np.ndarray` — Dense representation
- `frequency: int` — Corpus frequency
- `context_history: List[str]` — Usage contexts
- `generation_constraints: Dict` — Runtime constraints
- `semantic_fingerprint: str` — Fast lookup hash

### ReasoningCascade
- `input_nodes: List[CascadeNode]` — 6 input reasoning nodes
- `central_node: CascadeNode` — 1 central semantic anchor
- `output_nodes: List[CascadeNode]` — 6 output generation nodes
- `cascade_id: str` — Unique cascade identifier
- `confidence_score: float` — Overall cascade confidence
- `constraint_violations: List[str]` — Validation failure log

### CascadeNode
- `node_id: str` — Unique node identifier
- `node_type: NodeType` — INPUT, CENTRAL, or OUTPUT
- `concept: str` — Semantic concept representation
- `weight: float` — Activation strength [0.0, 1.0]
- `constraints: Dict[ConstraintType, Any]` — Node-specific constraints
- `connections: List[str]` — Connected node IDs
- `activation_threshold: float` — Minimum activation level
- `metadata: Dict[str, Any]` — Additional node information

## Configuration Parameters

### Model & Inference
- `device`: 'cpu' or 'cuda' (default: 'cpu')
- `vocab_size`: Vocabulary size (default: 30,000)
- `hidden_size`: Model hidden dimension (default: 768)
- `cascade_embedding_size`: Cascade node embedding size (default: 32)
- `cascade_guidance_strength`: Weight for cascade guidance (default: 0.8)
- `constraint_weight`: Weight for constraint loss (default: 1.0)
- `batch_size`: Inference batch size (default: 8)
- `max_length`: Maximum generation length (default: 100)
- `temperature`: Sampling temperature (default: 0.8)

### Constraint Types
- Semantic, Logical, Contextual, Temporal, Emotional, Domain

## Example Usage

### System Initialization
```python
from cascade_model import CascadeSystem
system = CascadeSystem(device='cuda')
system.build_from_corpus(corpus, vocab_size=30000)
```

### Generation API
```python
result = system.generate(
	prompt="The future of AI is",
	max_length=50,
	temperature=0.7,
	cascade_guidance_strength=0.8
)
print(result['generated_text'])
```

### REST API Example
```python
from flask import Flask, request, jsonify
app = Flask(__name__)
cascade_system = CascadeSystem.load('./model')

@app.route('/generate', methods=['POST'])
def generate():
	data = request.json
	result = cascade_system.generate(
		prompt=data['prompt'],
		max_length=data.get('max_length', 100),
		temperature=data.get('temperature', 0.8),
		cascade_guidance_strength=data.get('guidance_strength', 0.8)
	)
	return jsonify(result)
```

## Files
- `cascade_tokenizer.py`: Tokenizer logic
- `cascade_token.py`: Token and reasoning tree structures
- `cascade_model.py`: Model integration
- `cascade_inference.py`: Inference pipeline
- `cascade_trainer.py`: Training routines
- `cascade_demo.py`: Demo script

## System Requirements
- Python 3.8+
- CPU: 4+ cores (8+ recommended)
- RAM: 8GB+ (16GB+ recommended)
- GPU: 8GB+ VRAM for production

## License
See project documentation for license details.
