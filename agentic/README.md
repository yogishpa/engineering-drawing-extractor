# Barrel Dimension Extractor - True Multi-Agent Architecture (v4)

A sophisticated multi-agent AI system using the Strands framework for extracting precise dimensional measurements from engineering drawings. This version implements true agent collaboration, autonomous decision-making, and dynamic workflows.

## Key Differences from v3

### **True Multi-Agent Architecture**
- **Agent Autonomy**: Each agent makes independent decisions and chooses appropriate tools
- **Dynamic Collaboration**: Agents communicate, debate, and reach consensus
- **Specialized Expertise**: Each agent has unique knowledge domains and reasoning patterns
- **Adaptive Workflows**: Agents determine optimal collaboration strategies based on drawing complexity

### **Agent Specializations**

#### **1. Vision Analyst Agent**
- **Expertise**: Computer vision, geometric feature detection, spatial relationships
- **Tools**: Multiple vision models (Claude Opus, Sonnet), image preprocessing, geometric validators
- **Autonomy**: Chooses best vision model based on drawing complexity

#### **2. OCR Specialist Agent** 
- **Expertise**: Text extraction, dimension reading, tolerance interpretation
- **Tools**: AWS Textract, region-based OCR, text validation
- **Autonomy**: Selects OCR strategy based on text clarity and layout

#### **3. Engineering Analyst Agent**
- **Expertise**: Mechanical engineering principles, dimensional validation, tolerance analysis
- **Tools**: Engineering calculators, ratio validators, consistency checkers
- **Autonomy**: Applies engineering logic to validate and correct measurements

#### **4. Quality Assurance Agent**
- **Expertise**: Cross-validation, confidence scoring, error detection
- **Tools**: Statistical analyzers, outlier detection, consensus builders
- **Autonomy**: Decides when additional analysis is needed

#### **5. Coordinator Agent**
- **Expertise**: Workflow orchestration, conflict resolution, final decision making
- **Tools**: Result aggregators, conflict resolvers, report generators
- **Autonomy**: Manages agent interactions and ensures quality outcomes

## Installation

```bash
cd v4
pip install -r requirements.txt
```

## Usage

### Streamlit Interface
```bash
streamlit run multi_agent_app.py
```

### Programmatic Usage
```python
from multi_agent_extractor import TrueMultiAgentExtractor

extractor = TrueMultiAgentExtractor()
result = extractor.extract_dimensions("drawing.png")
```

## Architecture Benefits

1. **Improved Accuracy**: Multiple specialized agents cross-validate results
2. **Adaptive Processing**: Agents adjust strategies based on drawing characteristics  
3. **Robust Error Handling**: Agents collaborate to resolve ambiguities
4. **Scalable Design**: Easy to add new specialist agents
5. **Transparent Decision Making**: Full audit trail of agent interactions

## Agent Collaboration Patterns

- **Parallel Analysis**: Multiple agents analyze simultaneously
- **Peer Review**: Agents validate each other's findings
- **Consensus Building**: Agents negotiate final values
- **Escalation**: Complex cases trigger additional specialist agents
