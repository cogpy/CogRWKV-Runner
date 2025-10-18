# OpenCog Backend for RWKV-Runner

This backend integrates OpenCog's AGI architecture with RWKV neural networks to provide advanced cognitive capabilities for the RWKV-Runner system.

## Features

### Core Components

- **Atomspace**: Distributed knowledge representation system storing concepts, predicates, and relationships
- **MeTTa Bridge**: Programming interface for self-modifying AGI code and neural-symbolic integration
- **Cognitive Agent**: Autonomous reasoning system with goal-oriented processing
- **RWKV Interface**: Neural network integration for language generation and pattern recognition

### AGI Capabilities

- **Neural-Symbolic Integration**: Combines RWKV's neural processing with OpenCog's symbolic reasoning
- **Cognitive Synergy**: Multiple AI paradigms working together (neural nets, logic, learning)
- **Self-Modifying Code**: MeTTa programming language for dynamic cognitive architectures
- **Autonomous Goal Processing**: Agent-based system for complex task decomposition and execution
- **Knowledge Evolution**: Dynamic learning and knowledge base expansion
- **Attention Allocation**: Intelligent focus management across concepts and tasks

## Installation

1. Install dependencies:
```bash
cd backend-opencog
pip install -r requirements.txt
```

2. Start the OpenCog backend server:
```bash
python main.py --demo-mode --debug
```

3. The server will be available at `http://127.0.0.1:8001`

## API Usage

### Basic Cognitive Processing

```bash
# Process text through cognitive architecture
curl -X POST "http://127.0.0.1:8001/api/v1/opencog/cognitive/process" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the relationship between RWKV and neural networks?",
    "max_tokens": 150,
    "temperature": 0.8,
    "use_reasoning": true
  }'
```

### Atomspace Operations

```bash
# Create a concept
curl -X POST "http://127.0.0.1:8001/api/v1/opencog/atoms" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "CONCEPT",
    "name": "Artificial General Intelligence",
    "truth_value": 0.9,
    "confidence": 0.8
  }'

# Create a relationship
curl -X POST "http://127.0.0.1:8001/api/v1/opencog/links" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "INHERITANCE", 
    "source_id": "atom_id_1",
    "target_id": "atom_id_2",
    "truth_value": 0.8
  }'
```

### MeTTa Programming

```bash
# Evaluate MeTTa expression
curl -X POST "http://127.0.0.1:8001/api/v1/opencog/metta/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "expression": "(match (ConceptNode \"AI\") atomspace)",
    "context": {}
  }'

# Define a MeTTa procedure
curl -X POST "http://127.0.0.1:8001/api/v1/opencog/metta/define" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "smart-query",
    "body": "(lambda (concept) (knowledge-query \"related\" concept))"
  }'
```

### Goal-Based Processing

```bash
# Add a cognitive goal
curl -X POST "http://127.0.0.1:8001/api/v1/opencog/agent/goals" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "learn_concept",
    "description": "Learn about quantum computing",
    "parameters": {
      "concept": "Quantum Computing",
      "examples": ["qubit", "superposition", "entanglement"]
    },
    "priority": 0.8
  }'

# Process goals
curl -X POST "http://127.0.0.1:8001/api/v1/opencog/agent/process-goals"
```

### Knowledge Queries

```bash
# Query concepts
curl -X POST "http://127.0.0.1:8001/api/v1/opencog/knowledge/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "concepts",
    "parameters": {},
    "limit": 20
  }'

# Pattern matching
curl -X POST "http://127.0.0.1:8001/api/v1/opencog/knowledge/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "pattern",
    "parameters": {
      "type": "CONCEPT",
      "name_contains": "neural"
    },
    "limit": 10
  }'
```

## Configuration

Edit `config.yaml` to customize:

- Server settings (host, port, debug mode)
- Atomspace parameters (capacity, thresholds)
- Cognitive agent behavior (processing cycles, learning rate)
- MeTTa interpreter settings (evaluation limits)
- RWKV integration (model path, generation parameters)

## Integration with Main RWKV-Runner

The OpenCog backend can be used alongside the main RWKV-Runner:

1. **Standalone Mode**: Run as independent AGI server on port 8001
2. **Integrated Mode**: Main backend forwards complex reasoning tasks to OpenCog
3. **Hybrid Mode**: Distribute processing between neural and symbolic components

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │  Main Backend   │    │ OpenCog Backend │
│   (Wails/Go)    │◄──►│  (Python/FastAPI│◄──►│  (AGI/Python)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                        ┌───────▼────────┐      ┌────────▼────────┐
                        │  RWKV Models   │      │   Atomspace     │
                        │  (Neural Net)  │      │  (Symbolic KB)  │
                        └────────────────┘      └─────────────────┘
                                │                        │
                        ┌───────▼────────────────────────▼────────┐
                        │        Cognitive Agent                  │
                        │     (Neural-Symbolic AGI)               │
                        └─────────────────────────────────────────┘
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Adding New Reasoning Strategies

1. Implement new reasoning method in `cognitive_agent.py`
2. Add MeTTa operators in `metta_bridge.py`
3. Register new API endpoints in `routes/opencog_api.py`
4. Update configuration in `config.yaml`

### Extending Atomspace

1. Define new atom types in `core/atomspace.py`
2. Implement specialized reasoning for new types
3. Add serialization support for persistence
4. Create API endpoints for new operations

## Examples

See the `/examples` directory for:
- Basic cognitive processing workflows
- Knowledge base construction examples
- MeTTa programming tutorials
- Integration patterns with main RWKV backend

## Performance Considerations

- Atomspace scales to ~1M atoms efficiently
- Cognitive processing: 10-100 cycles/second depending on complexity
- MeTTa evaluation: sub-second for most expressions
- RWKV integration: depends on model size and available hardware

## Troubleshooting

1. **Startup Issues**: Check Python dependencies and port availability
2. **Performance**: Adjust cognitive cycles and attention parameters
3. **Memory Usage**: Monitor atomspace size and implement cleanup policies
4. **Integration**: Verify network connectivity between backends

## Contributing

1. Follow existing code patterns and documentation standards
2. Add tests for new functionality
3. Update API documentation for new endpoints
4. Consider backward compatibility and migration paths