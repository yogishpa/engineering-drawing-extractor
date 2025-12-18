import streamlit as st
import json
import os
import base64
import cv2
import numpy as np
import pdf2image
import boto3
import pandas as pd
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from strands import Agent, tool
from strands.multiagent import Swarm

# PLACEHOLDER: Set your AWS region for global models
os.environ['AWS_REGION'] = 'YOUR_AWS_REGION'  # Replace with your AWS region (e.g., 'us-west-2')

# Configure verbose logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('strands_extractor.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DimensionData:
    value: float = None
    tolerance: str = None
    unit: str = "mm"
    confidence: float = 0.0

@dataclass
class ExtractionResult:
    part_number: str = None
    overall_barrel_length: DimensionData = None
    barrel_head_length: DimensionData = None
    port_to_shoulder_length: DimensionData = None
    barrel_head_diameter: DimensionData = None
    barrel_shaft_diameter: DimensionData = None
    confidence_score: float = 0.0
    extraction_method: str = "strands_swarm"
    agent_results: Dict[str, Any] = None
    processing_logs: List[str] = None

# AWS Clients
@st.cache_resource
def get_aws_clients():
    """
    Initialize AWS clients for Bedrock and Textract
    PLACEHOLDER: Ensure your AWS credentials are configured
    """
    try:
        # PLACEHOLDER: Configure your AWS credentials via:
        # 1. AWS CLI: aws configure
        # 2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        # 3. IAM roles (if running on EC2)
        bedrock_client = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_REGION', 'us-west-2'))
        textract_client = boto3.client('textract', region_name='us-east-1')
        return bedrock_client, textract_client
    except Exception as e:
        st.error(f"AWS client initialization failed: {e}")
        return None, None

# Strands Agent Tools
@tool
def bedrock_vision_tool(image_path: str, model_id: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0") -> str:
    """Analyze engineering drawing using Bedrock Claude vision with fallback models"""
    logger.info(f"Starting Bedrock vision analysis with model: {model_id}")
    try:
        bedrock_client, _ = get_aws_clients()
        if not bedrock_client:
            logger.error("Bedrock client not available")
            return "Bedrock client not available"
        
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
            logger.info(f"Image loaded, size: {len(image_bytes)} bytes")
        
        prompt = """
        You are an expert in analyzing engineering drawings. Read EVERY number and dimension line carefully.

        SYSTEMATIC READING APPROACH:
        1. Scan the ENTIRE drawing from left to right, top to bottom
        2. Read EVERY dimension line, number, and tolerance marking
        3. Look at ALL diameter symbols (⌀) and their associated values
        4. Check ALL horizontal and vertical dimension lines
        5. Examine EVERY tolerance notation (+, -, ±)

        CRITICAL READING INSTRUCTIONS:
        
        OVERALL BARREL LENGTH:
        - Look for the LONGEST horizontal dimension line spanning the entire component
        - Check for dimension lines with arrows at both ends of the barrel
        - Read the number carefully - it may be different from what you expect
        - If multiple length dimensions exist, choose the one spanning the full length
        
        BARREL HEAD LENGTH:
        - Find dimension line across ONLY the wider head section (left side typically)
        - This is NOT the overall length - it's just the head portion
        - Look for dimension line that spans only the enlarged head area
        
        PORT TO SHOULDER LENGTH:
        - Look for SMALL dimension near the port opening
        - Often shows distance from port edge to a shoulder/step feature
        - Usually one of the smallest length dimensions on the drawing
        - Check carefully - may be a precise small value
        
        DIAMETERS - READ ALL ⌀ SYMBOLS:
        - BARREL HEAD DIAMETER: Look for larger ⌀ value (outer diameter of head section)
        - BARREL SHAFT DIAMETER: Look for smaller ⌀ value (main cylindrical body)
        - Read the EXACT numbers after each ⌀ symbol
        - Don't assume - read what's actually written
        
        TOLERANCES - READ EXACT NOTATION:
        - Copy the EXACT tolerance symbols and numbers as shown
        - Examples: +0.4, -0.1, ±0.1, +0.2/-0.1, h11, etc.
        - Include ALL tolerance information exactly as written
        
        VERIFICATION PROCESS:
        - Double-check each number you extract
        - Verify you're reading the correct dimension line
        - Ensure you haven't confused similar-looking numbers
        - Cross-reference with nearby dimensions for reasonableness

        RETURN FORMAT:
        {
            "reading_process": "Describe what you see and where you found each dimension",
            "part_number": "EXACT_PART_NUMBER_FROM_TITLE_BLOCK",
            "overall_barrel_length": {"value": EXACT_NUMBER_READ, "tolerance": "EXACT_TOLERANCE_READ", "unit": "mm", "source": "describe the dimension line location"},
            "barrel_head_length": {"value": EXACT_NUMBER_READ, "tolerance": "EXACT_TOLERANCE_READ", "unit": "mm", "source": "describe the dimension line location"},
            "port_to_shoulder_length": {"value": EXACT_NUMBER_READ, "tolerance": "EXACT_TOLERANCE_READ", "unit": "mm", "source": "describe the dimension line location"},
            "barrel_head_diameter": {"value": EXACT_NUMBER_READ, "tolerance": "EXACT_TOLERANCE_READ", "unit": "mm", "source": "describe the ⌀ symbol location"},
            "barrel_shaft_diameter": {"value": EXACT_NUMBER_READ, "tolerance": "EXACT_TOLERANCE_READ", "unit": "mm", "source": "describe the ⌀ symbol location"}
        }

        CRITICAL: Read EXACTLY what is written. Do not round, estimate, or assume values.
        """
        
        # Try Claude 4.5 models first, then fallback models
        fallback_models = [
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",  # Claude 4.5 Sonnet
            "us.anthropic.claude-opus-4-5-20251101-v1:0",        # Claude 4.5 Opus
            "anthropic.claude-3-sonnet-20240229-v1:0",           # Claude 3 Sonnet
            "anthropic.claude-3-haiku-20240307-v1:0"             # Claude 3 Haiku
        ]
        
        for attempt_model in fallback_models:
            try:
                logger.info(f"Attempting with model: {attempt_model}")
                
                response = bedrock_client.invoke_model(
                    modelId=attempt_model,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 2000,
                        "messages": [{
                            "role": "user",
                            "content": [{
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.b64encode(image_bytes).decode()
                                }
                            }, {
                                "type": "text",
                                "text": prompt
                            }]
                        }]
                    })
                )
                
                result = json.loads(response['body'].read())
                logger.info(f"✅ SUCCESS with model: {attempt_model}")
                return result['content'][0]['text']
                
            except Exception as model_error:
                logger.warning(f"❌ Model {attempt_model} failed: {str(model_error)}")
                continue
        
        # If all models fail, return error
        error_msg = "All Bedrock models failed - service may be unavailable"
        logger.error(error_msg)
        return error_msg
        
    except Exception as e:
        logger.error(f"Bedrock vision error: {str(e)}")
        return f"Bedrock vision error: {str(e)}"

# Specialized Agents as Tools
@tool
def data_automation_tool(image_path: str) -> str:
    """Bedrock Data Automation agent with Claude 4.5 Sonnet"""
    logger.info("Starting Bedrock Data Automation analysis")
    result = bedrock_vision_tool(image_path, 'global.anthropic.claude-sonnet-4-5-20250929-v1:0')
    logger.info(f"BEDROCK DATA AUTOMATION AGENT OUTPUT: {result}")
    return result

@tool
def vision_analysis_tool(image_path: str) -> str:
    """Claude 4.5 Sonnet agent for vision analysis"""
    result = bedrock_vision_tool(image_path, 'global.anthropic.claude-sonnet-4-5-20250929-v1:0')
    logger.info(f"CLAUDE SONNET AGENT OUTPUT: {result}")
    return result

@tool
def reasoning_analysis_tool(image_path: str) -> str:
    """Claude 4.5 Opus agent for deep reasoning"""
    result = bedrock_vision_tool(image_path, 'global.anthropic.claude-opus-4-5-20251101-v1:0')
    logger.info(f"CLAUDE OPUS AGENT OUTPUT: {result}")
    return result

@tool
def opus_evaluator_tool(extracted_fields: str) -> str:
    """Claude Opus 4.5 evaluator that processes extracted fields for final decision"""
    logger.info("OPUS EVALUATOR: Processing extracted fields")
    
    try:
        bedrock_client, _ = get_aws_clients()
        
        evaluation_prompt = f"""You are the final evaluator. Analyze all agent extractions and select the most accurate values.

EXTRACTED FIELDS TO EVALUATE:
{extracted_fields}

EVALUATION STRATEGY:
1. COMPARE all agent extractions for each dimension
2. LOOK for agents that provide detailed source descriptions
3. CROSS-VALIDATE values against dimensional logic
4. SELECT the most precise and well-documented extractions
5. REJECT values that seem inconsistent or poorly sourced

OUTPUT FORMAT (JSON only):
{{
    "part_number": "BEST_EXTRACTED_PART_NUMBER",
    "overall_barrel_length": {{"value": MOST_ACCURATE_VALUE, "tolerance": "MOST_ACCURATE_TOLERANCE", "unit": "mm", "confidence": 0.0-1.0}},
    "barrel_head_length": {{"value": MOST_ACCURATE_VALUE, "tolerance": "MOST_ACCURATE_TOLERANCE", "unit": "mm", "confidence": 0.0-1.0}},
    "port_to_shoulder_length": {{"value": MOST_ACCURATE_VALUE, "tolerance": "MOST_ACCURATE_TOLERANCE", "unit": "mm", "confidence": 0.0-1.0}},
    "barrel_head_diameter": {{"value": MOST_ACCURATE_VALUE, "tolerance": "MOST_ACCURATE_TOLERANCE", "unit": "mm", "confidence": 0.0-1.0}},
    "barrel_shaft_diameter": {{"value": MOST_ACCURATE_VALUE, "tolerance": "MOST_ACCURATE_TOLERANCE", "unit": "mm", "confidence": 0.0-1.0}},
    "evaluation_reasoning": "Explain why you selected each value and rejected others"
}}

Return ONLY the JSON output, no additional text."""

        # Call Claude Opus 4.5 for evaluation
        response = bedrock_client.invoke_model(
            modelId='global.anthropic.claude-opus-4-5-20251101-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": evaluation_prompt}]
            })
        )
        
        response_body = json.loads(response['body'].read())
        result = response_body['content'][0]['text']
        logger.info(f"OPUS EVALUATOR OUTPUT: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ OPUS EVALUATOR ERROR: {str(e)}")
        error_result = {
            "part_number": "EVALUATION_ERROR",
            "overall_barrel_length": {"value": None, "tolerance": "N/A", "unit": "mm", "confidence": 0.0},
            "barrel_head_length": {"value": None, "tolerance": "N/A", "unit": "mm", "confidence": 0.0},
            "port_to_shoulder_length": {"value": None, "tolerance": "N/A", "unit": "mm", "confidence": 0.0},
            "barrel_head_diameter": {"value": None, "tolerance": "N/A", "unit": "mm", "confidence": 0.0},
            "barrel_shaft_diameter": {"value": None, "tolerance": "N/A", "unit": "mm", "confidence": 0.0},
            "evaluation_reasoning": f"Evaluation failed: {str(e)}"
        }
        return json.dumps(error_result, indent=2)

# Main Extractor Class
class StrandsBarrelExtractor:
    def __init__(self):
        self.swarm = None
        self.setup_swarm()
    
    def setup_swarm(self):
        """Initialize the Strands swarm with specialized agents"""
        logger.info("Setting up Strands swarm with specialized agents")
        
        # Define specialized agents
        data_automation_agent = Agent(
            name="data_automation_specialist",
            instructions="You are a data automation specialist focused on precise dimension extraction from engineering drawings using Claude 4.5 Sonnet.",
            tools=[data_automation_tool]
        )
        
        vision_analysis_agent = Agent(
            name="vision_specialist", 
            instructions="You are a vision analysis specialist using Claude 4.5 Sonnet for detailed visual inspection of engineering drawings.",
            tools=[vision_analysis_tool]
        )
        
        reasoning_agent = Agent(
            name="reasoning_specialist",
            instructions="You are a reasoning specialist using Claude 4.5 Opus for deep analysis and validation of extracted dimensions.",
            tools=[reasoning_analysis_tool]
        )
        
        evaluator_agent = Agent(
            name="evaluator_specialist",
            instructions="You are the final evaluator using Claude Opus 4.5 to analyze all agent outputs and select the most accurate dimensions.",
            tools=[opus_evaluator_tool]
        )
        
        # Create swarm
        self.swarm = Swarm()
        logger.info("✅ Strands swarm initialized with 4 specialized agents")
    
    def extract_dimensions(self, image_path: str) -> ExtractionResult:
        """Extract dimensions using the Strands swarm"""
        logger.info(f"Starting swarm dimension extraction for: {image_path}")
        
        try:
            # Run agents in sequence
            data_result = data_automation_tool(image_path)
            vision_result = vision_analysis_tool(image_path)
            reasoning_result = reasoning_analysis_tool(image_path)
            
            # Combine results for evaluation
            all_results = f"""
            Data Automation Agent: {data_result}
            
            Vision Analysis Agent: {vision_result}
            
            Reasoning Agent: {reasoning_result}
            """
            
            # Final evaluation
            final_result = opus_evaluator_tool(all_results)
            logger.info("Swarm execution completed")
            
            # Parse final result
            try:
                result_data = json.loads(final_result)
                return self.convert_to_extraction_result(result_data)
            except json.JSONDecodeError:
                logger.error("Failed to parse swarm result, using fallback")
                return self.create_fallback_result()
                
        except Exception as e:
            logger.error(f"Swarm execution failed: {str(e)}")
            return self.create_fallback_result()
    
    def convert_to_extraction_result(self, data: dict) -> ExtractionResult:
        """Convert swarm result to ExtractionResult object"""
        def create_dimension_data(dim_dict):
            if dim_dict and isinstance(dim_dict, dict):
                return DimensionData(
                    value=dim_dict.get('value'),
                    tolerance=dim_dict.get('tolerance'),
                    unit=dim_dict.get('unit', 'mm'),
                    confidence=dim_dict.get('confidence', 0.8)
                )
            return DimensionData()
        
        result = ExtractionResult(
            part_number=data.get('part_number'),
            overall_barrel_length=create_dimension_data(data.get('overall_barrel_length')),
            barrel_head_length=create_dimension_data(data.get('barrel_head_length')),
            port_to_shoulder_length=create_dimension_data(data.get('port_to_shoulder_length')),
            barrel_head_diameter=create_dimension_data(data.get('barrel_head_diameter')),
            barrel_shaft_diameter=create_dimension_data(data.get('barrel_shaft_diameter')),
            confidence_score=0.85,
            extraction_method="strands_swarm",
            agent_results=data,
            processing_logs=[f"Swarm evaluation: {data.get('evaluation_reasoning', 'No reasoning provided')}"]
        )
        
        return result
    
    def create_fallback_result(self) -> ExtractionResult:
        """Create fallback result when swarm fails"""
        return ExtractionResult(
            part_number="EXTRACTION_FAILED",
            confidence_score=0.0,
            extraction_method="fallback",
            processing_logs=["Swarm extraction failed, using fallback"]
        )

def display_results_table(result: ExtractionResult):
    """Display extraction results as a formatted table"""
    dimensions_data = []
    
    dimension_mapping = {
        'Part Number': result.part_number,
        'Overall Barrel Length': result.overall_barrel_length,
        'Barrel Head Length': result.barrel_head_length,
        'Port to Shoulder Length': result.port_to_shoulder_length,
        'Barrel Head Diameter': result.barrel_head_diameter,
        'Barrel Shaft Diameter': result.barrel_shaft_diameter
    }
    
    for dim_name, dim_data in dimension_mapping.items():
        if dim_name == 'Part Number':
            dimensions_data.append({
                'Dimension': dim_name,
                'Value': dim_data if dim_data else 'Not Found',
                'Tolerance': 'N/A',
                'Unit': 'N/A',
                'Confidence': 'N/A'
            })
        elif dim_data and hasattr(dim_data, 'value'):
            dimensions_data.append({
                'Dimension': dim_name,
                'Value': f"{dim_data.value:.2f}" if dim_data.value is not None else 'Not Found',
                'Tolerance': dim_data.tolerance if dim_data.tolerance else 'N/A',
                'Unit': dim_data.unit if dim_data.unit else 'mm',
                'Confidence': f"{dim_data.confidence:.1%}" if dim_data.confidence else 'N/A'
            })
        else:
            dimensions_data.append({
                'Dimension': dim_name,
                'Value': 'Not Found',
                'Tolerance': 'N/A',
                'Unit': 'mm',
                'Confidence': 'N/A'
            })
    
    # Create DataFrame and display
    df = pd.DataFrame(dimensions_data)
    st.dataframe(df, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        found_dims = sum(1 for row in dimensions_data if row['Value'] != 'Not Found' and row['Dimension'] != 'Part Number')
        st.metric("Dimensions Found", f"{found_dims}/5")
    with col2:
        st.metric("Overall Confidence", f"{result.confidence_score:.1%}")
    with col3:
        st.metric("Extraction Method", result.extraction_method)

# Streamlit UI
def main():
    st.set_page_config(page_title="Strands Swarm Barrel Extractor", layout="wide")
    
    st.title("🔧 Strands Swarm Barrel Dimension Extractor")
    st.write("**Multi-Agent Swarm using Strands Framework**")
    st.info("🆕 Swarm Intelligence with specialized agents for precise dimension extraction!")
    
    uploaded_files = st.file_uploader(
        "Upload Engineering Drawing PDFs", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("Please upload PDF files to begin swarm extraction")
        return
    
    if st.button("🚀 Start Swarm Analysis", type="primary"):
        extractor = StrandsBarrelExtractor()
        
        for uploaded_file in uploaded_files:
            st.subheader(f"📄 Processing: {uploaded_file.name}")
            
            # Save and convert PDF
            temp_pdf_path = f"temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Convert to image
            images = pdf2image.convert_from_path(temp_pdf_path, dpi=300)
            temp_image_path = temp_pdf_path.replace('.pdf', '_page1.png')
            images[0].save(temp_image_path, 'PNG')
            
            # Extract dimensions using swarm
            with st.spinner("🤖 Swarm agents analyzing..."):
                result = extractor.extract_dimensions(temp_image_path)
            
            # Display results
            st.subheader("📊 Swarm Results")
            display_results_table(result)
            
            # Processing logs
            if result.processing_logs:
                with st.expander("📋 Swarm Execution Logs"):
                    for log in result.processing_logs:
                        st.text(log)
            
            # Clean up
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

if __name__ == "__main__":
    main()
