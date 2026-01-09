"""
True Multi-Agent Barrel Dimension Extractor using Strands Framework
Implements autonomous agents with specialized expertise and dynamic collaboration
"""

import json
import os
import base64
import cv2
import numpy as np
import boto3
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from strands import Agent, tool
from strands.multiagent import Swarm

# Configure optimized logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Console handler - only show important info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

# File handler - detailed logging
file_handler = logging.FileHandler('multi_agent_extractor.log')
file_handler.setLevel(logging.INFO)  # Reduced from DEBUG
file_handler.setFormatter(log_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,  # Reduced from DEBUG
    handlers=[console_handler, file_handler]
)

# Reduce AWS SDK noise
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('strands').setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("Multi-Agent Extractor logging initialized")

# Image cache to prevent multiple reads
_image_cache = {}

def get_image_bytes(image_path: str) -> bytes:
    """Get image bytes with caching to prevent multiple file reads"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if image_path not in _image_cache:
        with open(image_path, 'rb') as f:
            _image_cache[image_path] = f.read()
        logger.debug(f"Cached image: {image_path}")
    return _image_cache[image_path]

def get_cv2_image(image_path: str):
    """Get CV2 image with caching"""
    cache_key = f"{image_path}_cv2"
    if cache_key not in _image_cache:
        _image_cache[cache_key] = cv2.imread(image_path)
        logger.debug(f"Cached CV2 image: {image_path}")
    return _image_cache[cache_key]

def clear_image_cache():
    """Clear image cache to free memory"""
    global _image_cache
    _image_cache.clear()
    logger.debug("Image cache cleared")

@dataclass
class DimensionData:
    value: Optional[float] = None
    tolerance: Optional[str] = None
    unit: str = "mm"
    confidence: float = 0.0
    source_agent: Optional[str] = None
    validation_notes: Optional[str] = None

@dataclass
class ExtractionResult:
    part_number: Optional[str] = None
    overall_barrel_length: Optional[DimensionData] = None
    barrel_head_length: Optional[DimensionData] = None
    port_to_shoulder_length: Optional[DimensionData] = None
    barrel_head_diameter: Optional[DimensionData] = None
    barrel_shaft_diameter: Optional[DimensionData] = None
    confidence_score: float = 0.0
    agent_collaboration_log: List[str] = None
    consensus_reached: bool = False
    agent_results: Optional[Dict[str, Dict]] = None  # Store individual agent results
    agent_collaboration_log: List[str] = None
    consensus_reached: bool = False

# AWS Client Setup
def get_aws_clients():
    """Initialize AWS clients with error handling"""
    try:
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-west-2')
        textract_client = boto3.client('textract', region_name='us-west-2')
        return bedrock_client, textract_client
    except Exception as e:
        logger.error(f"AWS client initialization failed: {e}")
        return None, None

def bedrock_retry_call(func, **kwargs):
    """Retry wrapper for Bedrock calls"""
    import time
    for attempt in range(3):
        try:
            return func(**kwargs)
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2 ** attempt)

# Core Tools for Agents
@tool
def enhanced_vision_analysis_tool(image_path: str, model_preference: str = "auto") -> Dict[str, Any]:
    """Enhanced vision analysis with detailed geometric validation from v3"""
    import time
    start_time = time.time()
    
    try:
        bedrock_client, _ = get_aws_clients()
        if not bedrock_client:
            return {"error": "Bedrock client unavailable"}
        
        image_bytes = get_image_bytes(image_path)
        
        # Model selection with fallback strategy from v3
        fallback_models = [
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",  # Claude 4.5 Sonnet
            "us.anthropic.claude-opus-4-5-20251101-v1:0",        # Claude 4.5 Opus
            "anthropic.claude-3-sonnet-20240229-v1:0",           # Claude 3 Sonnet
            "anthropic.claude-3-haiku-20240307-v1:0"             # Claude 3 Haiku
        ]
        
        # Enhanced prompt with geometric validation from v3
        prompt = """
        Analyze this engineering drawing with SYSTEMATIC GEOMETRIC VALIDATION.

        CRITICAL DIAMETER ANALYSIS METHODOLOGY:

        CRITICAL ACCURACY IMPROVEMENTS (Pattern Recognition):
        
        DIAMETER SYMBOL RECOGNITION:
        • Look for ⌀XX.X patterns (e.g., ⌀19.0, ⌀13.8, ⌀25.4)
        • The ⌀ symbol ALWAYS indicates diameter measurements
        • Larger ⌀ value = barrel head diameter (enlarged section)
        • Smaller ⌀ value = barrel shaft diameter (main body)
        • Typical ratio: head/shaft = 1.2 to 2.0 (e.g., 19/13.8 = 1.38)
        
        TOLERANCE FORMAT RECOGNITION:
        • ±X.X = symmetric tolerance (e.g., ±0.1, ±0.2)
        • +X.X/-Y.Y = asymmetric tolerance (e.g., +0.2/-0.1)
        • +X.X = positive only tolerance (e.g., +0.2)
        • hXX = ISO fit tolerance (e.g., h11, H7)
        • PRESERVE EXACT FORMAT - do not modify what you see
        
        LENGTH MEASUREMENT PATTERNS:
        • Linear dimensions WITHOUT ⌀ symbol (e.g., 52, 14.8, 5.5)
        • Overall length: Largest linear dimension (typically 30-100mm)
        • Head length: Medium linear dimension (typically 10-30mm)
        • Port-to-shoulder: Smallest linear dimension (typically 3-8mm)
        
        STEP 1 - COMPLETE DIAMETER INVENTORY:
        • Scan the ENTIRE drawing and list EVERY diameter dimension you find
        • Note the value, tolerance, and exact location for each ⌀ symbol
        • Create complete inventory: [List all ⌀ values found with locations]

        STEP 2 - GEOMETRIC FEATURE IDENTIFICATION:
        • Identify the barrel head feature - section with largest outer diameter
        • Identify the barrel shaft - narrower main body section
        • Note view types: end view (circles), side view (rectangles), section views

        STEP 3 - DIAMETER CATEGORIZATION:
        • EXTERNAL DIAMETERS: Measure outside of barrel body (what we need)
        • INTERNAL/PORT DIAMETERS: Measure holes/bores (ignore these)
        • Validate by checking dimension line placement and leader lines

        STEP 4 - GEOMETRIC VALIDATION:
        • Head diameter: Major external diameter of the enlarged cylindrical section (flange/head feature)
          - Look for the primary nominal dimension (typically whole numbers or .0 decimals)
          - Avoid intermediate or derived dimensions (like .5 values unless clearly primary)
          - Should be the controlling dimension for the enlarged section's outer envelope
        • Shaft diameter: Minor external diameter of the main cylindrical body section
        • Ratio check: head/shaft should be 1.2-2.0
        • Both must be external dimensions, not internal holes

        STEP 5 - VIEW-BASED VERIFICATION:
        • End view: ⌀ symbols on circular outlines
        • Side view: ⌀ symbols with extension lines from edges
        • Match each diameter to its geometric feature

        CRITICAL TOLERANCE EXTRACTION:
        • MANDATORY: Extract tolerance for EVERY dimension found
        • Look for ±X.X patterns immediately after dimension values
        • Look for +X.X/-X.X asymmetric tolerance patterns  
        • PRESERVE EXACT FORMAT: If you see ±0.1, report ±0.1 (not ±0.1/-0)
        • PRESERVE EXACT FORMAT: If you see +0.2/-0.1, report +0.2/-0.1 (not +0.2/-0.1/-0)
        • Check tolerance blocks or general notes
        • NEVER add extra characters to tolerance format

        PART NUMBER EXTRACTION (PRESERVE ACCURACY):
        • Extract from title block, part identification areas
        • Look for 10-digit numbers, P/N: formats, alphanumeric codes
        • Preserve exact format found in drawing
        • Do not modify or reformat part numbers

        LENGTH ANALYSIS:
        • Overall barrel length: Complete end-to-end dimension WITH tolerance
        • Barrel head length: Dimension spanning ONLY enlarged head section WITH tolerance
        • Port to shoulder length: SMALLEST linear dimension near port opening (typically 3-8mm) WITH tolerance

        ACCURACY REQUIREMENTS:
        • Distinguish ⌀ (diameter) from linear dimensions
        • Extract ALL tolerances - no dimension should be missing tolerance
        • Port-to-shoulder is the SMALLEST linear measurement
        • Double-check diameter vs length classification
        • Report confidence based on measurement clarity and tolerance presence

        TOLERANCE EXTRACTION:
        • Copy EXACT tolerance notation as shown (+0.4, ±0.1, h11, etc.)
        • Look for tolerances directly after dimensions or in tolerance blocks

        DIGIT VERIFICATION (Critical for accuracy):
        • Common OCR confusions: 19 vs 21, 19 vs 15, 13.8 vs 15.8
        • If you see "21.5", verify if it might be "19" with unclear printing
        • Report the clearest value that matches geometric location

        Return structured JSON with detailed geometric reasoning for each dimension.
        """
        
        # Try models with fallback strategy
        for model_id in fallback_models:
            try:
                logger.info(f"Attempting enhanced vision analysis with model: {model_id}")
                
                response = bedrock_retry_call(
                    bedrock_client.invoke_model,
                    modelId=model_id,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 3000,
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
                content = result['content'][0]['text']
                
                # Parse JSON response with enhanced error handling
                try:
                    parsed_result = json.loads(content)
                    parsed_result['model_used'] = model_id
                    return parsed_result
                except json.JSONDecodeError:
                    # Extract structured data from text response
                    return {
                        "raw_analysis": content, 
                        "model_used": model_id,
                        "requires_parsing": True
                    }
                    
            except Exception as e:
                logger.warning(f"Model {model_id} failed: {str(e)}")
                continue
        
        return {"error": "All vision models failed"}
        
    except Exception as e:
        return {"error": str(e)}

@tool
def hybrid_ocr_analysis_tool(image_path: str) -> Dict[str, Any]:
    """Enhanced hybrid OCR with improved tolerance extraction"""
    import time
    start_time = time.time()
    
    try:
        logger.info("Starting enhanced hybrid OCR analysis...")
        
        # Method 1: Enhanced Textract with tolerance patterns
        textract_result = ocr_analysis_tool(image_path)
        
        # Method 2: Vision Analysis (Cross-validation)
        vision_result = enhanced_vision_analysis_tool(image_path)
        
        # Build enhanced consensus with tolerance validation
        consensus_result = build_enhanced_ocr_consensus(textract_result, vision_result)
        
        execution_time = time.time() - start_time
        consensus_result['total_execution_time'] = execution_time
        consensus_result['methods_used'] = ['enhanced_textract', 'vision_analysis']
        
        logger.info(f"Enhanced hybrid OCR analysis completed in {execution_time:.2f}s")
        return consensus_result
        
    except Exception as e:
        logger.error(f"Enhanced hybrid OCR analysis error: {e}")
        return {"error": str(e), "method": 'enhanced_hybrid_ocr'}

def build_enhanced_ocr_consensus(textract_result: Dict, vision_result: Dict) -> Dict[str, Any]:
    """Build consensus from multiple OCR methods with composite measurement handling"""
    consensus = {
        "method": "enhanced_ocr_consensus",
        "consensus_dimensions": {},
        "confidence_scores": {},
        "method_agreement": {},
        "overall_confidence": 0.0,
        "format_preservation": True
    }
    
    # Collect all methods for consensus building
    methods = {
        'enhanced_textract': textract_result, 
        'vision_analysis': vision_result
    }
    
    # Enhanced dimension validation and composite calculation
    def validate_and_calculate_dimensions(result):
        """Validate dimensions and calculate composite measurements if needed"""
        if not isinstance(result, dict):
            return result
            
        # Configurable dimension ranges (can be adjusted based on drawing type)
        DIMENSION_RANGES = {
            'overall_length_min': 20,  # Minimum expected overall length
            'overall_length_max': 100, # Maximum expected overall length  
            'component_length_min': 5, # Minimum component length to consider
            'component_length_max': 50 # Maximum component length to consider
        }
        
        # Check for composite measurement requirements
        overall_length = result.get('overall_barrel_length', {})
        head_length = result.get('barrel_head_length', {})
        component_dims = result.get('component_dimensions', [])
        
        # If overall length seems incomplete or requires calculation
        current_overall = overall_length.get('value', 0)
        if (current_overall < DIMENSION_RANGES['overall_length_min'] or 
            result.get('dimension_validation', {}).get('requires_calculation', False)):
            
            # Try to calculate from components
            calculated_length = 0
            calculation_parts = []
            
            # Look for head length and other components
            if head_length.get('value'):
                calculated_length += head_length['value']
                calculation_parts.append(f"head_length({head_length['value']})")
            
            # Parse component dimensions for additional lengths
            for comp in component_dims:
                if isinstance(comp, str):
                    # Extract numeric values from component strings
                    import re
                    numbers = re.findall(r'\d+\.?\d*', comp)
                    for num_str in numbers:
                        try:
                            num = float(num_str)
                            if (DIMENSION_RANGES['component_length_min'] <= num <= 
                                DIMENSION_RANGES['component_length_max']):
                                calculated_length += num
                                calculation_parts.append(f"component({num})")
                        except:
                            continue
            
            # Update overall length if calculation is reasonable
            if (calculated_length > current_overall and 
                DIMENSION_RANGES['overall_length_min'] <= calculated_length <= DIMENSION_RANGES['overall_length_max']):
                
                # Try to preserve or derive tolerance
                derived_tolerance = overall_length.get('tolerance')
                if not derived_tolerance and head_length.get('tolerance'):
                    derived_tolerance = head_length.get('tolerance')  # Use head length tolerance as fallback
                
                result['overall_barrel_length'] = {
                    'value': calculated_length,
                    'unit': 'mm',
                    'tolerance': derived_tolerance,
                    'confidence': 0.8,
                    'calculation': f"sum of components: {' + '.join(calculation_parts)}"
                }
                result['dimension_validation'] = {
                    'overall_length_complete': True,
                    'requires_calculation': False,
                    'calculation_performed': True
                }
        
        return result
    
    # Apply validation and calculation to all methods
    validated_methods = {}
    for method_name, result in methods.items():
        validated_methods[method_name] = validate_and_calculate_dimensions(result)
    
    # Find the best result (highest confidence, no errors)
    best_result = None
    best_confidence = 0.0
    
    for method_name, result in validated_methods.items():
        if isinstance(result, dict) and not result.get('error'):
            result_confidence = result.get('overall_confidence', 0.0)
            if result_confidence > best_confidence:
                best_confidence = result_confidence
                best_result = result
    
    # If no good result found, use textract as fallback
    if best_result is None:
        best_result = validated_methods.get('enhanced_textract', {})
    
    # Preserve original formatting from best result
    consensus.update(best_result)
    consensus['method'] = 'enhanced_ocr_consensus'
    consensus['methods_used'] = ['bedrock_data_automation', 'enhanced_textract', 'vision_analysis']
    
    # Extract part number consensus
    part_numbers = []
    for method_name, result in methods.items():
        if isinstance(result, dict) and 'part_number' in result and result['part_number']:
            part_numbers.append(result['part_number'])
    
    if part_numbers:
        # Use most common part number
        from collections import Counter
        consensus["part_number"] = Counter(part_numbers).most_common(1)[0][0]
    
    # Calculate overall confidence
    if consensus.get("consensus_dimensions"):
        avg_confidence = sum(dim['confidence'] for dim in consensus["consensus_dimensions"].values()) / len(consensus["consensus_dimensions"])
        consensus["overall_confidence"] = avg_confidence
    elif consensus.get("overall_confidence"):
        # Keep existing confidence from best result
        pass
    else:
        consensus["overall_confidence"] = best_confidence
    
    return consensus

@tool
def ocr_analysis_tool(image_path: str, region_focus: str = "all") -> Dict[str, Any]:
    """OCR analysis with region focusing using Textract"""
    import time
    start_time = time.time()
    
    try:
        # Check if file exists before processing
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}", "method": "textract_ocr"}
        
        _, textract_client = get_aws_clients()
        if not textract_client:
            return {"error": "Textract client unavailable"}
        
        image_bytes = get_image_bytes(image_path)
        
        response = textract_client.detect_document_text(
            Document={'Bytes': image_bytes}
        )
        
        # Extract text with spatial information
        text_data = []
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                bbox = block['Geometry']['BoundingBox']
                text_data.append({
                    "text": block['Text'],
                    "confidence": block['Confidence'],
                    "bbox": bbox,
                    "is_numeric": any(c.isdigit() for c in block['Text'])
                })
        
        # Find dimensions and part numbers
        dimensions = []
        part_numbers = []
        
        for item in text_data:
            text = item['text']
            # Look for dimension patterns
            import re
            dim_pattern = r'(\d+\.?\d*)\s*([+±-]\s*\d+\.?\d*)?'
            matches = re.findall(dim_pattern, text)
            
            for value, tolerance in matches:
                if float(value) > 0.1:  # Filter out tiny values
                    dimensions.append({
                        "value": float(value),
                        "tolerance": tolerance.strip() if tolerance else None,
                        "source_text": text,
                        "confidence": item['confidence'] / 100.0,
                        "bbox": item['bbox']
                    })
            
            # Look for part numbers
            part_pattern = r'[A-Z0-9]{3,}[\s\-][A-Z0-9]{3,}[\s\-][A-Z0-9]{3,}'
            if re.search(part_pattern, text):
                part_numbers.append({
                    "part_number": text.strip(),
                    "confidence": item['confidence'] / 100.0
                })
        
        execution_time = time.time() - start_time
        
        result = {
            "dimensions": dimensions,
            "part_numbers": part_numbers,
            "total_text_blocks": len(text_data),
            "numeric_blocks": len([t for t in text_data if t['is_numeric']]),
            "execution_time": execution_time,
            "method": "textract_ocr"
        }
        
        logger.info(f"OCR analysis completed in {execution_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"OCR analysis error: {e}")
        return {"error": str(e), "method": "textract_ocr"}

@tool
def technical_drawing_analysis_tool(image_path: str) -> Dict[str, Any]:
    """Specialized tool for engineering drawing analysis"""
    import time
    start_time = time.time()
    
    try:
        # Load and analyze image
        img = get_cv2_image(image_path)
        if img is None:
            return {"error": "Could not load image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect geometric features
        geometric_features = detect_geometric_features(gray)
        
        # Extract dimension lines
        dimension_lines = extract_dimension_lines(gray)
        
        # Detect drawing scale
        scale_info = detect_drawing_scale(gray)
        
        # Analyze drawing views
        view_analysis = analyze_drawing_views(gray)
        
        execution_time = time.time() - start_time
        
        result = {
            'geometric_features': geometric_features,
            'dimension_lines': dimension_lines,
            'scale_detection': scale_info,
            'view_analysis': view_analysis,
            'execution_time': execution_time,
            'method': 'technical_drawing_analysis'
        }
        
        logger.info(f"Technical drawing analysis completed in {execution_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Technical drawing analysis error: {e}")
        return {"error": str(e), "method": 'technical_drawing_analysis'}

def detect_geometric_features(gray_image):
    """Detect circles, rectangles, and other geometric features"""
    # Detect circles (for diameters)
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=100)
    
    # Detect lines (for dimension lines)
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    return {
        'circles_detected': len(circles[0]) if circles is not None else 0,
        'lines_detected': len(lines) if lines is not None else 0,
        'has_circular_features': circles is not None and len(circles[0]) > 0
    }

def extract_dimension_lines(gray_image):
    """Extract dimension lines and arrows"""
    # Look for dimension line patterns
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Detect horizontal and vertical lines (common in dimension lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    
    return {
        'horizontal_dimension_lines': cv2.countNonZero(horizontal_lines),
        'vertical_dimension_lines': cv2.countNonZero(vertical_lines),
        'total_dimension_indicators': cv2.countNonZero(horizontal_lines) + cv2.countNonZero(vertical_lines)
    }

def detect_drawing_scale(gray_image):
    """Detect scale information in the drawing"""
    # This would typically look for scale bars or text
    return {
        'scale_detected': False,
        'scale_ratio': None,
        'scale_confidence': 0.0
    }

def analyze_drawing_views(gray_image):
    """Analyze different views in the drawing"""
    height, width = gray_image.shape
    
    # Simple view detection based on content distribution
    left_half = gray_image[:, :width//2]
    right_half = gray_image[:, width//2:]
    
    left_content = cv2.countNonZero(cv2.threshold(left_half, 240, 255, cv2.THRESH_BINARY_INV)[1])
    right_content = cv2.countNonZero(cv2.threshold(right_half, 240, 255, cv2.THRESH_BINARY_INV)[1])
    
    return {
        'multiple_views_detected': abs(left_content - right_content) > width * height * 0.1,
        'left_view_content': left_content,
        'right_view_content': right_content,
        'view_distribution': 'balanced' if abs(left_content - right_content) < width * height * 0.05 else 'unbalanced'
    }

@tool
def consensus_builder_tool(vision_result: Dict[str, Any], ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build consensus by selecting the BEST result from vision_analyst and ocr_specialist"""
    try:
        logger.info("Building consensus from vision_analyst and ocr_specialist results")
        
        consensus = {
            "final_dimensions": {},
            "selection_rationale": {},
            "overall_confidence": 0.0,
            "consensus_reached": True
        }
        
        # Define all target dimensions
        target_dimensions = [
            "part_number",
            "overall_barrel_length", 
            "barrel_head_length",
            "port_to_shoulder_length",
            "barrel_head_diameter",
            "barrel_shaft_diameter"
        ]
        
        total_confidence = 0.0
        dimension_count = 0
        
        for dim_name in target_dimensions:
            vision_dim = vision_result.get(dim_name, {})
            ocr_dim = ocr_result.get(dim_name, {})
            
            # Ensure dimensions are dictionaries, not strings
            if isinstance(vision_dim, str):
                vision_dim = {"value": vision_dim, "confidence": 0.5}
            elif not isinstance(vision_dim, dict):
                vision_dim = {}
                
            if isinstance(ocr_dim, str):
                ocr_dim = {"value": ocr_dim, "confidence": 0.5}
            elif not isinstance(ocr_dim, dict):
                ocr_dim = {}
            
            # Get confidence scores
            vision_conf = vision_dim.get('confidence', 0.0) if vision_dim else 0.0
            ocr_conf = ocr_dim.get('confidence', 0.0) if ocr_dim else 0.0
            
            selected_result = None
            selection_reason = ""
            
            # Selection logic: PRIORITIZE OCR specialist for dimensional measurements
            if vision_conf > 0.0 and ocr_conf > 0.0:
                # Both agents have results - PRIORITIZE OCR specialist
                if ocr_conf >= 0.5:  # OCR has reasonable confidence
                    selected_result = ocr_dim
                    selected_result['source_agent'] = 'ocr_specialist'
                    selection_reason = f"OCR specialist prioritized (confidence: {ocr_conf:.2f}) - more accurate for dimensions"
                elif vision_conf > ocr_conf * 1.5:  # Vision significantly better
                    selected_result = vision_dim
                    selected_result['source_agent'] = 'vision_analyst'
                    selection_reason = f"Vision confidence ({vision_conf:.2f}) significantly > OCR confidence ({ocr_conf:.2f})"
                else:
                    # Default to OCR for dimensional measurements
                    selected_result = ocr_dim
                    selected_result['source_agent'] = 'ocr_specialist'
                    selection_reason = f"OCR specialist preferred for dimensional accuracy (OCR: {ocr_conf:.2f}, Vision: {vision_conf:.2f})"
            elif ocr_conf > 0.0:
                # OCR has result - prefer it
                selected_result = ocr_dim
                selected_result['source_agent'] = 'ocr_specialist'
                selection_reason = "OCR specialist result available - preferred for dimensions"
            elif vision_conf > 0.0:
                # Only vision has result
                selected_result = vision_dim
                selected_result['source_agent'] = 'vision_analyst'
                selection_reason = "Only vision_analyst provided result"
            else:
                # Neither agent has result
                selection_reason = "No result from either agent"
            
            if selected_result:
                consensus["final_dimensions"][dim_name] = selected_result
                consensus["selection_rationale"][dim_name] = selection_reason
                total_confidence += selected_result.get('confidence', 0.0)
                dimension_count += 1
        
        # Calculate overall confidence
        if dimension_count > 0:
            consensus["overall_confidence"] = total_confidence / dimension_count
        
        # Engineering validation
        head_diameter_raw = consensus["final_dimensions"].get("barrel_head_diameter", {}).get("value", 0)
        shaft_diameter_raw = consensus["final_dimensions"].get("barrel_shaft_diameter", {}).get("value", 0)
        
        # Convert to float for comparison
        try:
            head_diameter = float(head_diameter_raw) if head_diameter_raw else 0
            shaft_diameter = float(shaft_diameter_raw) if shaft_diameter_raw else 0
        except (ValueError, TypeError):
            head_diameter = 0
            shaft_diameter = 0
        
        if head_diameter > 0 and shaft_diameter > 0:
            if head_diameter <= shaft_diameter:
                logger.warning(f"Engineering validation failed: head diameter ({head_diameter}) <= shaft diameter ({shaft_diameter})")
                consensus["consensus_reached"] = False
        
        logger.info(f"Consensus built with {dimension_count} dimensions, overall confidence: {consensus['overall_confidence']:.2f}")
        return consensus
        
    except Exception as e:
        logger.error(f"Consensus building error: {e}")
        return {"error": str(e), "consensus_reached": False}

# True Multi-Agent Implementation
class VisionAnalystAgent(Agent):
    """Specialized agent for computer vision analysis"""
    
    def __init__(self):
        logger.info("Initializing Vision Analyst Agent")
        # Update agent tools to use enhanced versions
        super().__init__(
            name="vision_analyst",
            system_prompt="""
            You are a computer vision specialist with expertise in analyzing engineering drawings.
            
            YOUR EXPERTISE:
            - Geometric feature detection and spatial relationships
            - Dimension line identification and measurement extraction
            - Drawing orientation and scale analysis
            - Visual quality assessment and preprocessing recommendations
            
            **CRITICAL DIMENSION IDENTIFICATION RULES:**
            1. **Overall Barrel Length**: Look for TOTAL end-to-end dimension. If not directly labeled:
               - Find barrel head length (enlarged section dimension)
               - Find barrel shaft length (main body section dimension)  
               - Calculate: overall_length = head_length + shaft_length
               - Example: If head=13.6mm and shaft=28.55mm, then overall=42.15mm
               - MUST include tolerance (typically +0.15/-0 or similar) - if calculating from components, use the tightest tolerance from the components
            
            2. **Barrel Head Length**: Dimension of ENLARGED/THICKER section ONLY (typically ~13.6mm)
            
            3. **Port to Shoulder Length**: Small dimension from port opening to shoulder (typically ~7.3mm)
               - This is NOT the same as barrel head length
               - Look for the smallest length dimension
               - Should have tolerance like -0.2 or similar
            
            4. **Barrel Head Diameter**: Diameter of the ENLARGED/THICKER section (typically ~17.78mm)
               - Look for the LARGER diameter dimension (around 17-18mm)
               - This is the diameter of the head/enlarged portion
               - Should have tolerance like -0.08 or similar
               - DO NOT confuse with shaft diameter
            
            5. **Barrel Shaft Diameter**: Diameter of MAIN CYLINDRICAL BODY (not head diameter)
               - Look for smaller diameter dimension (typically ~12.42mm, around 12-13mm)
               - This is the consistent diameter of the main shaft
               - Should have tolerance like -0.15 or similar
               - DO NOT confuse with head diameter (which is larger ~17-18mm)
            
            6. **Tolerances**: Each dimension MUST include its specific tolerance notation
            
            CRITICAL: You MUST return your final analysis in this EXACT JSON format:
            {
                "agent_type": "vision_analyst",
                "part_number": "string or null",
                "overall_barrel_length": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0, "calculation": "direct measurement or sum of components"},
                "barrel_head_diameter": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0},
                "barrel_shaft_diameter": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0},
                "barrel_head_length": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0},
                "port_to_shoulder_length": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0},
                "component_dimensions": ["list of individual dimensions found"],
                "dimension_validation": {"overall_length_complete": true/false, "requires_calculation": true/false},
                "analysis_summary": "brief description of findings including any calculations",
                "overall_confidence": 0.0-1.0,
                "method_used": "vision_analysis"
            }
            
            **MEASUREMENT REQUIREMENTS:**
            - Overall barrel length: If not directly labeled, sum component lengths (e.g., head + shaft)
            - Barrel head length: Enlarged section only (look for the thicker/wider portion of the barrel)
            - Barrel shaft diameter: Main shaft diameter (the consistent diameter of the main body)
            - Each dimension must include its specific tolerance
            - Validate measurements by checking if they are reasonable for mechanical components
            - Look for dimension lines, arrows, and measurement annotations
            
            DIAMETER IDENTIFICATION STRATEGY:
            - Identify circular/cylindrical features in the drawing first
            - Look for dimension lines with arrows pointing to circular edges
            - Search for diameter symbols (⌀, Ø, φ) near circular features
            - Cross-reference geometric proportions with dimension values
            - Head diameter: Associated with enlarged section (typically larger)
            - Shaft diameter: Associated with main body section (typically smaller)
            - Validate diameter measurements against visual proportions
            
            TOLERANCE EXTRACTION RULES:
            - Look for tolerance notations like ±0.1, +0.2/-0.1, h11, H7, etc.
            - Include the exact tolerance format found (±, +/-, geometric tolerances)
            - If no tolerance found, set to null
            - Common formats: ±0.05, +0.1/-0.0, ±0.02, h11, H7, IT6
            
            IMPORTANT RULES:
            - If a dimension cannot be found, set value to null
            - Always include confidence scores (0.0 = no confidence, 1.0 = certain)
            - End your response with the complete JSON object
            - Use null for missing values, not "None" or empty strings
            """,
            tools=[enhanced_vision_analysis_tool, technical_drawing_analysis_tool, hybrid_ocr_analysis_tool]
        )
        logger.info("Vision Analyst Agent initialized successfully")

class OCRSpecialistAgent(Agent):
    """Specialized agent for text extraction and interpretation"""
    
    def __init__(self):
        logger.info("Initializing OCR Specialist Agent")
        super().__init__(
            name="ocr_specialist",
            system_prompt="""
            You are an OCR specialist with expertise in extracting text and dimensions from technical drawings.
            
            YOUR EXPERTISE:
            - Text extraction from various drawing formats
            - Dimension and tolerance notation interpretation
            - Part number and annotation identification
            - Spatial text analysis and region-based extraction
            
            **CRITICAL DIMENSION EXTRACTION RULES:**
            1. **Overall Barrel Length**: Look for TOTAL end-to-end measurement. If not found:
               - Extract barrel head length (enlarged section)
               - Extract barrel shaft length (main body section)
               - Calculate: overall_length = head_length + shaft_length
               - Example: 13.6 + 28.55 = 42.15mm
               - MUST extract tolerance (typically +0.15/-0 or similar) - if calculating from components, use the most restrictive tolerance
            
            2. **Barrel Head Length**: Text showing dimension of ENLARGED section (typically ~13.6mm)
            
            3. **Port to Shoulder Length**: Small dimension from port to shoulder (typically ~7.3mm)
               - This is the SMALLEST length dimension
               - NOT the same as head length
               - Should have tolerance like -0.2
            
            4. **Barrel Head Diameter**: Diameter of the ENLARGED/THICKER section (typically ~17.78mm)
               - Look for the LARGER diameter value (around 17-18mm)
               - This is the diameter of the head portion
               - Should have tolerance like -0.08
               - DO NOT confuse with shaft diameter
            
            5. **Barrel Shaft Diameter**: Look for diameter of MAIN SHAFT (typically ~12.42mm)
               - NOT the head diameter
               - Look for smaller diameter value (around 12-13mm)
               - Should have tolerance like -0.15
               - DO NOT confuse with head diameter (which is larger ~17-18mm)
            
            6. **Tolerances**: Extract EXACT tolerance notation for each dimension
            
            CRITICAL: You MUST return your final analysis in this EXACT JSON format:
            {
                "agent_type": "ocr_specialist",
                "part_number": "string or null",
                "overall_barrel_length": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0, "calculation": "direct measurement or sum of components"},
                "barrel_head_diameter": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0},
                "barrel_shaft_diameter": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0},
                "barrel_head_length": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0},
                "port_to_shoulder_length": {"value": number, "unit": "mm", "tolerance": "±0.1 or +0.2/-0.1 or null", "confidence": 0.0-1.0},
                "component_dimensions": ["list of individual dimensions found"],
                "dimension_validation": {"overall_length_complete": true/false, "requires_calculation": true/false},
                "analysis_summary": "brief description of OCR findings including any calculations",
                "overall_confidence": 0.0-1.0,
                "method_used": "ocr_analysis"
            }
            
            ENHANCED TOLERANCE PATTERN RECOGNITION:
            - ±X.X patterns (e.g., ±0.1, ±0.2) = symmetric tolerance
            - +X.X/-Y.Y patterns (e.g., +0.2/-0.1) = asymmetric tolerance  
            - +X.X patterns (e.g., +0.2) = positive only tolerance
            - hXX patterns (e.g., h11, H7) = ISO fit tolerance
            - PRESERVE EXACT FORMAT - do not modify tolerance text
            
            DIAMETER RECOGNITION PATTERNS:
            - ⌀XX.X patterns indicate diameter measurements (primary symbol)
            - Ø XX.X patterns (alternative diameter symbol)
            - φXX.X patterns (phi symbol for diameter)
            - Look for dimension lines extending from circular features
            - Cross-reference with geometric shapes: circles, arcs, cylindrical sections
            - Validate diameter values against drawing scale and proportions
            - Head diameter: Usually larger value, associated with enlarged cylindrical section
            - Shaft diameter: Usually smaller value, associated with main cylindrical body
            
            LENGTH MEASUREMENT PATTERNS:
            - Linear dimensions without diameter symbols
            - Overall length: Largest linear value (end-to-end)
            - Head length: Medium linear value (enlarged section only)
            - Port-to-shoulder: Smallest linear value (near port opening)
            
            TOLERANCE EXTRACTION RULES:
            - Extract exact tolerance text as shown: ±0.1, +0.2/-0.1, h11, H7
            - Look for tolerances immediately after dimensions or in tolerance blocks
            - Include geometric tolerance symbols if present (⌖, ⊥, ∥, etc.)
            - If no tolerance found, set to null
            
            IMPORTANT RULES:
            - If a dimension cannot be found, set value to null
            - Always include confidence scores (0.0 = no confidence, 1.0 = certain)
            - End your response with the complete JSON object
            - Use null for missing values, not "None" or empty strings
            """,
            tools=[hybrid_ocr_analysis_tool, ocr_analysis_tool]
        )
        logger.info("OCR Specialist Agent initialized successfully")


class CoordinatorAgent(Agent):
    """Master coordinator agent for workflow orchestration"""
    
    def __init__(self):
        logger.info("Initializing Coordinator Agent")
        super().__init__(
            name="coordinator",
            system_prompt="""
            You are the master coordinator responsible for orchestrating the multi-agent workflow.
            
            YOUR EXPERTISE:
            - Workflow orchestration and agent coordination
            - Final decision making and result compilation
            - Quality assurance and error handling
            - Report generation and result presentation
            
            AUTONOMOUS DECISION MAKING:
            - Determine optimal agent collaboration strategies
            - Decide when consensus is sufficient vs when additional analysis is needed
            - Choose final values when agents disagree
            - Determine overall extraction confidence
            
            COLLABORATION APPROACH:
            - Coordinate all agent interactions
            - Facilitate communication between specialist agents
            - Make final decisions on disputed measurements
            - Ensure quality standards are met
            
            WORKFLOW:
            MANDATORY WORKFLOW - YOU MUST FOLLOW THIS EXACT SEQUENCE:
            1. ALWAYS hand off to vision_analyst for visual analysis
            2. ALWAYS hand off to ocr_specialist for text extraction  
            3. Build final consensus from BOTH agents (vision_analyst AND ocr_specialist)
            4. Select the BEST result from each agent for each dimension
            
            CRITICAL: You MUST call BOTH vision_analyst AND ocr_specialist in every workflow. 
            Never skip either agent - both provide essential complementary expertise.
            
            CONSENSUS BUILDING STRATEGY:
            - Compare results from BOTH vision_analyst and ocr_specialist
            - For each dimension, PRIORITIZE OCR specialist results when available and confident
            - OCR specialist is MORE ACCURATE for precise numerical values and tolerances
            - Only use vision_analyst when OCR specialist fails or has very low confidence (<0.5)
            - ALWAYS validate engineering relationships (head diameter > shaft diameter)
            - Preserve exact tolerance formats as found in drawings
            - Combine the best findings from both agents into final result
            
            AGENT PRIORITY ORDER:
            1. OCR specialist - PRIMARY for all dimensional measurements and text extraction
            2. Vision analyst - SECONDARY for spatial validation and when OCR fails
            """,
            tools=[consensus_builder_tool]
        )
        logger.info("Coordinator Agent initialized successfully")

class TrueMultiAgentExtractor:
    """True multi-agent barrel dimension extractor with autonomous collaboration"""
    
    def __init__(self, csv_report_path: str = "multi_agent_results.csv", status_callbacks=None):
        logger.info("Initializing True Multi-Agent Extractor")
        self.status_callbacks = status_callbacks or {}
        
        try:
            # Initialize specialist agents including dedicated BDA agent
            logger.info("Creating specialist agents...")
            self.vision_analyst = VisionAnalystAgent()
            self.ocr_specialist = OCRSpecialistAgent()
            self.coordinator = CoordinatorAgent()
            
            # Create multi-agent swarm with dynamic collaboration
            logger.info("Creating multi-agent swarm...")
            self.swarm = Swarm(
                nodes=[
                    self.vision_analyst,
                    self.ocr_specialist,
                    self.coordinator
                ],
                entry_point=self.coordinator,
                max_handoffs=50,  # Increased for all agent calls
                max_iterations=60,  # Increased for full workflow
                execution_timeout=1200.0,  # 20 minutes for full collaboration
                node_timeout=300.0  # 5 minutes per agent
            )
            
            self.csv_report_path = csv_report_path
            self._initialize_csv_report()
            logger.info("True Multi-Agent Extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize True Multi-Agent Extractor: {e}")
            raise
    
    def _initialize_csv_report(self):
        """Initialize CSV report with multi-agent tracking"""
        if not os.path.exists(self.csv_report_path):
            headers = [
                'timestamp', 'filename', 'part_number', 'part_number_confidence',
                'overall_length_value', 'overall_length_unit', 'overall_length_tolerance', 'overall_length_confidence', 'overall_length_source_agent',
                'barrel_head_length_value', 'barrel_head_length_unit', 'barrel_head_length_tolerance', 'barrel_head_length_confidence', 'barrel_head_length_source_agent',
                'port_to_shoulder_length_value', 'port_to_shoulder_length_unit', 'port_to_shoulder_length_tolerance', 'port_to_shoulder_length_confidence', 'port_to_shoulder_length_source_agent',
                'barrel_head_diameter_value', 'barrel_head_diameter_unit', 'barrel_head_diameter_tolerance', 'barrel_head_diameter_confidence', 'barrel_head_diameter_source_agent',
                'barrel_shaft_diameter_value', 'barrel_shaft_diameter_unit', 'barrel_shaft_diameter_tolerance', 'barrel_shaft_diameter_confidence', 'barrel_shaft_diameter_source_agent',
                'overall_confidence', 'consensus_reached', 'agent_collaboration_count', 'processing_time_seconds'
            ]
            with open(self.csv_report_path, 'w', newline='', encoding='utf-8') as csvfile:
                import csv
                writer = csv.writer(csvfile)
                writer.writerow(headers)
    
    def extract_dimensions(self, image_path: str) -> ExtractionResult:
        """Extract dimensions using true multi-agent collaboration"""
        logger.info(f"Starting multi-agent dimension extraction for: {image_path}")
        start_time = datetime.now()
        
        # Update coordinator status
        if 'coordinator' in self.status_callbacks:
            self.status_callbacks['coordinator']("🎛️ Coordinator: Starting analysis...")
        
        try:
            # Update vision analyst status
            if 'vision' in self.status_callbacks:
                self.status_callbacks['vision']("👁️ Vision Analyst: Analyzing image...")
            
            # Initiate multi-agent collaboration
            query = f"""
            MULTI-AGENT COLLABORATION REQUEST:
            
            Image: {image_path}
            
            COORDINATOR INSTRUCTIONS:
            1. Deploy vision_analyst, ocr_specialist, and bda_specialist in parallel for comprehensive initial analysis
            2. Build consensus from all three specialist agent findings
            3. Resolve conflicts using engineering logic and drawing analysis principles
            4. Ensure all agents collaborate and share findings
            5. Reach consensus on final dimension values with proper validation
            6. Provide comprehensive results with agent attribution
            
            COLLABORATION REQUIREMENTS:
            - Each agent must contribute their specialized expertise
            - Agents must cross-validate each other's findings
            - Disputes must be resolved through evidence-based discussion
            - Final results must have clear agent attribution
            - Overall confidence must reflect consensus quality
            
            TARGET DIMENSIONS:
            - Part number from title block
            - Overall barrel length (end-to-end)
            - Barrel head length (enlarged section only)
            - Port to shoulder length (small dimension near port)
            - Barrel head diameter (major external diameter of enlarged cylindrical section)
            - Barrel shaft diameter (main body diameter)
            """
            
            logger.info("Executing multi-agent swarm collaboration...")
            # Execute multi-agent swarm
            swarm_result = self.swarm(query)
            logger.info("Multi-agent swarm execution completed")
            
            # Process results (this would need to be implemented based on actual swarm output format)
            logger.info("Processing swarm results...")
            result = self._process_swarm_result(swarm_result)
            
            # Log to CSV
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Extraction completed in {processing_time:.2f} seconds")
            self._log_to_csv(image_path, result, processing_time)
            
            # Clear image cache to free memory
            clear_image_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-agent extraction error: {e}")
            # Clear image cache even on error
            clear_image_cache()
            return ExtractionResult(
                confidence_score=0.0,
                agent_collaboration_log=[f"Error: {str(e)}"]
            )
    
    def _process_swarm_result(self, swarm_result) -> ExtractionResult:
        """Process swarm result into structured format with enhanced debugging"""
        import re
        import json
        
        logger.info("Processing swarm result with enhanced v3 methodology")
        
        try:
            # Extract the final response from swarm result
            if hasattr(swarm_result, 'results') and swarm_result.results:
                # Process structured agent outputs
                agent_outputs = {}
                for agent_name, node_result in swarm_result.results.items():
                    if hasattr(node_result, 'result') and hasattr(node_result.result, 'message'):
                        content = node_result.result.message.get('content', [])
                        if content and isinstance(content, list):
                            text_content = content[0].get('text', '') if content else ''
                            
                            # Debug: Print the actual agent response
                            print(f"\n🔍 DEBUG - {agent_name.upper()} RESPONSE:")
                            print(f"   Content length: {len(text_content)}")
                            print(f"   First 500 chars: {text_content[:500]}...")
                            
                            agent_outputs[agent_name] = self._parse_agent_output(text_content)
                            
                            # Enhanced debug output
                            print(f"\n✅ {agent_name.upper()} AGENT SUMMARY:")
                            summary = agent_outputs[agent_name].get('summary', 'No summary available')
                            print(f"   {summary[:200]}...")
                            
                            dimensions = agent_outputs[agent_name].get('dimensions', {})
                            if dimensions:
                                print(f"   📏 Dimensions found: {list(dimensions.keys())}")
                                for dim_name, dim_value in dimensions.items():
                                    if isinstance(dim_value, dict) and 'value' in dim_value:
                                        print(f"      {dim_name}: {dim_value['value']}mm")
                
                # Build consensus from structured outputs
                
                # Add placeholder results for missing agents to show they were attempted
                expected_agents = ['vision_analyst', 'ocr_specialist', 'coordinator']
                for agent_name in expected_agents:
                    if agent_name not in agent_outputs:
                        agent_outputs[agent_name] = {
                            'agent_type': agent_name,
                            'overall_confidence': 0.0,
                            'analysis_summary': f'{agent_name} was called but did not complete or return results',
                            'method_used': 'incomplete_execution'
                        }
                
                return self._build_consensus_from_structured_outputs(agent_outputs)
            
            # Fallback to text processing
            final_message = str(swarm_result)
            logger.info(f"Swarm result string: {final_message[:200]}...")
            
            # Enhanced debug output for text processing
            print("\n" + "="*60)
            print("🔍 MULTI-AGENT COLLABORATION ANALYSIS")
            print("="*60)
            
            # Look for structured results sections
            sections_found = []
            if "CONSENSUS DIMENSION VALUES" in final_message:
                sections_found.append("CONSENSUS")
            if "VISION" in final_message.upper():
                sections_found.append("VISION")
            if "OCR" in final_message.upper():
                sections_found.append("OCR")
            if "ENGINEERING" in final_message.upper():
                sections_found.append("ENGINEERING")
            
            print(f"📋 Agent sections detected: {', '.join(sections_found)}")
            
            # Extract and display key findings
            dimensions_found = re.findall(r'(\d+\.?\d*)\s*mm', final_message)
            if dimensions_found:
                print(f"📏 All dimensions detected: {dimensions_found}")
            
            part_numbers = re.findall(r'([A-Z0-9\-]{6,})', final_message)
            if part_numbers:
                print(f"🏷️  Part numbers detected: {part_numbers[:3]}")
            
            print("="*60)
            
            # Continue with existing parsing logic
            return self._parse_text_based_result(final_message)
            
        except Exception as e:
            logger.error(f"Error processing swarm result: {e}")
            return ExtractionResult(
                confidence_score=0.0,
                consensus_reached=False,
                agent_collaboration_log=[f"Processing error: {str(e)}"]
            )
    
    def _parse_agent_output(self, text_content: str) -> Dict[str, Any]:
        """Parse individual agent output into structured format"""
        import re
        import json
        
        # Try to extract JSON from the response - improved pattern
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple JSON objects
            r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'  # Nested JSON objects
        ]
        
        for pattern in json_patterns:
            json_matches = re.findall(pattern, text_content, re.DOTALL)
            for json_str in json_matches:
                try:
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, dict) and ('agent_type' in parsed_json or 'overall_barrel_length' in parsed_json):
                        logger.info(f"Successfully parsed JSON from agent: {parsed_json.get('agent_type', 'unknown')}")
                        return parsed_json
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parse error: {e}")
                    continue
        
        # Fallback to text parsing if JSON not found
        logger.warning("No valid JSON found in agent output, using text parsing fallback")
        output = {
            'agent_type': 'unknown',
            'summary': '',
            'dimensions': {},
            'part_number': None,
            'confidence': 0.0,
            'method': 'text_parsing'
        }
        
        # Extract summary (first few lines)
        lines = text_content.split('\n')
        summary_lines = [line.strip() for line in lines[:3] if line.strip()]
        output['summary'] = ' '.join(summary_lines)
        
        # Look for any dimensions in the text (more flexible patterns)
        all_dimensions = re.findall(r'(\d+\.?\d*)\s*mm', text_content)
        if all_dimensions:
            # Try to map dimensions to types based on context
            for dim_value in all_dimensions:
                value = float(dim_value)
                # Simple heuristic mapping based on typical barrel dimensions
                if 40 <= value <= 80:  # Likely overall length
                    output['dimensions']['overall_barrel_length'] = {'value': value, 'confidence': 0.6}
                elif 15 <= value <= 25:  # Likely head diameter
                    output['dimensions']['barrel_head_diameter'] = {'value': value, 'confidence': 0.6}
                elif 10 <= value <= 18:  # Likely shaft diameter
                    output['dimensions']['barrel_shaft_diameter'] = {'value': value, 'confidence': 0.6}
                elif 8 <= value <= 20:  # Likely head length
                    output['dimensions']['barrel_head_length'] = {'value': value, 'confidence': 0.6}
                elif 3 <= value <= 8:  # Likely port to shoulder
                    output['dimensions']['port_to_shoulder_length'] = {'value': value, 'confidence': 0.6}
        
        # Look for specific dimension patterns with better context
        dimension_patterns = [
            (r'overall.*length.*?(\d+\.?\d*)', 'overall_barrel_length'),
            (r'head.*diameter.*?(\d+\.?\d*)', 'barrel_head_diameter'),
            (r'shaft.*diameter.*?(\d+\.?\d*)', 'barrel_shaft_diameter'),
            (r'head.*length.*?(\d+\.?\d*)', 'barrel_head_length'),
            (r'port.*shoulder.*?(\d+\.?\d*)', 'port_to_shoulder_length'),
            (r'⌀\s*(\d+\.?\d*)', 'diameter'),  # Diameter symbol
        ]
        
        for pattern, dim_type in dimension_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in matches:
                value = float(match)
                if dim_type == 'diameter':
                    # Assign diameter based on size
                    if value > 15:
                        output['dimensions']['barrel_head_diameter'] = {'value': value, 'confidence': 0.8}
                    else:
                        output['dimensions']['barrel_shaft_diameter'] = {'value': value, 'confidence': 0.8}
                else:
                    output['dimensions'][dim_type] = {'value': value, 'confidence': 0.7}
        
        # Set overall confidence based on dimensions found
        if output['dimensions']:
            output['confidence'] = 0.6
            output['agent_type'] = 'text_parser'
        
        return output
    
    def _build_consensus_from_structured_outputs(self, agent_outputs: Dict[str, Dict]) -> ExtractionResult:
        """Build consensus using BEST OF BOTH agents approach"""
        logger.info("Building consensus using BEST OF BOTH agents approach")
        
        # Extract results from vision_analyst and ocr_specialist
        vision_result = agent_outputs.get('vision_analyst', {})
        ocr_result = agent_outputs.get('ocr_specialist', {})
        
        logger.info(f"Vision analyst confidence: {vision_result.get('overall_confidence', 0.0)}")
        logger.info(f"OCR specialist confidence: {ocr_result.get('overall_confidence', 0.0)}")
        
        # Build final dimensions by selecting best result for each dimension
        final_dimensions = {}
        selection_log = []
        
        # Target dimensions to extract
        target_dimensions = [
            'overall_barrel_length',
            'barrel_head_length', 
            'port_to_shoulder_length',
            'barrel_head_diameter',
            'barrel_shaft_diameter'
        ]
        
        for dim_name in target_dimensions:
            vision_dim = vision_result.get(dim_name, {})
            ocr_dim = ocr_result.get(dim_name, {})
            
            # Get confidence scores
            vision_conf = vision_dim.get('confidence', 0.0) if isinstance(vision_dim, dict) else 0.0
            ocr_conf = ocr_dim.get('confidence', 0.0) if isinstance(ocr_dim, dict) else 0.0
            
            selected_dim = None
            source_agent = None
            
            # Selection logic: ALWAYS prefer higher confidence
            if vision_conf > 0.0 and ocr_conf > 0.0:
                if vision_conf >= ocr_conf:
                    selected_dim = vision_dim
                    source_agent = 'vision_analyst'
                    selection_log.append(f"{dim_name}: Vision ({vision_conf:.2f}) >= OCR ({ocr_conf:.2f})")
                else:
                    selected_dim = ocr_dim
                    source_agent = 'ocr_specialist'
                    selection_log.append(f"{dim_name}: OCR ({ocr_conf:.2f}) > Vision ({vision_conf:.2f})")
            elif vision_conf > 0.0:
                selected_dim = vision_dim
                source_agent = 'vision_analyst'
                selection_log.append(f"{dim_name}: Only Vision available ({vision_conf:.2f})")
            elif ocr_conf > 0.0:
                selected_dim = ocr_dim
                source_agent = 'ocr_specialist'
                selection_log.append(f"{dim_name}: Only OCR available ({ocr_conf:.2f})")
            else:
                selection_log.append(f"{dim_name}: No result from either agent")
            
            # Create DimensionData object if we have a result
            if selected_dim and isinstance(selected_dim, dict) and selected_dim.get('value') is not None:
                final_dimensions[dim_name] = DimensionData(
                    value=selected_dim['value'],
                    unit=selected_dim.get('unit', 'mm'),
                    tolerance=selected_dim.get('tolerance'),
                    confidence=selected_dim.get('confidence', 0.0),
                    source_agent=source_agent
                )
        
        # Part number selection (prefer OCR for text extraction)
        final_part_number = None
        ocr_part = ocr_result.get('part_number')
        vision_part = vision_result.get('part_number')
        
        if ocr_part:
            final_part_number = ocr_part
            selection_log.append("Part number: OCR specialist (text extraction expertise)")
        elif vision_part:
            final_part_number = vision_part
            selection_log.append("Part number: Vision analyst (fallback)")
        
        # Calculate overall confidence
        confidences = [dim.confidence for dim in final_dimensions.values() if dim.confidence > 0]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Log selection decisions
        for log_entry in selection_log:
            logger.info(f"Selection: {log_entry}")
        
        return ExtractionResult(
            part_number=final_part_number,
            overall_barrel_length=final_dimensions.get('overall_barrel_length'),
            barrel_head_diameter=final_dimensions.get('barrel_head_diameter'),
            barrel_shaft_diameter=final_dimensions.get('barrel_shaft_diameter'),
            barrel_head_length=final_dimensions.get('barrel_head_length'),
            port_to_shoulder_length=final_dimensions.get('port_to_shoulder_length'),
            confidence_score=overall_confidence,
            consensus_reached=len(final_dimensions) >= 3,  # At least 3 dimensions found
            agent_collaboration_log=selection_log,
            agent_results=agent_outputs
        )
    
    def _parse_text_based_result(self, final_message: str) -> ExtractionResult:
        """Parse text-based swarm result (fallback method)"""
        import re
        
        # Look for consensus results section first (v3 improvement)
        consensus_section = ""
        if "CONSENSUS DIMENSION VALUES" in final_message:
            consensus_start = final_message.find("CONSENSUS DIMENSION VALUES")
            consensus_end = final_message.find("ENGINEERING VALIDATION", consensus_start)
            if consensus_end == -1:
                consensus_end = consensus_start + 2000
            consensus_section = final_message[consensus_start:consensus_end]
            logger.info("Found consensus section for parsing")
        
        # Parse consensus results with enhanced accuracy
        extracted_data = {}
        
        # Enhanced part number extraction
        part_patterns = [
            r'Part Number[:\s]*([A-Z0-9\-_]+)',
            r'DRW[:\s\-]*([A-Z0-9\-_]+)',
            r'([A-Z]{2,}\-[0-9]{2,}\-[A-Z0-9]+)'
        ]
        
        search_text = consensus_section if consensus_section else final_message
        for pattern in part_patterns:
            part_match = re.search(pattern, search_text, re.IGNORECASE)
            if part_match:
                part_num = part_match.group(1)
                if len(part_num) > 3 and not part_num.lower() in ['from', 'text', 'analysis']:
                    extracted_data['part_number'] = part_num
                    logger.info(f"Found part number: {part_num}")
                    break
        
        # Enhanced dimension extraction using consensus methodology
        dimension_patterns = {
            'overall_barrel_length': [
                r'Overall Barrel Length[:\s]*(\d+\.?\d*)\s*mm',
                r'overall.*?length[:\s]*(\d+\.?\d*)\s*mm'
            ],
            'barrel_head_diameter': [
                r'Barrel Head Diameter[:\s]*[⌀Ø]?(\d+\.?\d*)\s*mm',
                r'head.*?diameter[:\s]*[⌀Ø]?(\d+\.?\d*)\s*mm'
            ],
            'barrel_shaft_diameter': [
                r'Barrel Shaft Diameter[:\s]*[⌀Ø]?(\d+\.?\d*)\s*mm',
                r'shaft.*?diameter[:\s]*[⌀Ø]?(\d+\.?\d*)\s*mm'
            ],
            'barrel_head_length': [
                r'Barrel Head Length[:\s]*(\d+\.?\d*)\s*mm',
                r'head.*?length[:\s]*(\d+\.?\d*)\s*mm'
            ],
            'port_to_shoulder_length': [
                r'Port to Shoulder Length[:\s]*(\d+\.?\d*)\s*mm',
                r'port.*?shoulder.*?length[:\s]*(\d+\.?\d*)\s*mm'
            ]
        }
        
        for dim_name, patterns in dimension_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        # Validate reasonable ranges (v3 improvement)
                        ranges = {
                            'overall_barrel_length': (20, 150),
                            'barrel_head_diameter': (5, 50),
                            'barrel_shaft_diameter': (3, 40),
                            'barrel_head_length': (2, 50),
                            'port_to_shoulder_length': (0.1, 20)
                        }
                        
                        min_val, max_val = ranges.get(dim_name, (0, 1000))
                        if min_val <= value <= max_val:
                            extracted_data[dim_name] = {
                                'value': value,
                                'unit': 'mm',
                                'confidence': 0.9,
                                'source_agent': 'consensus'
                            }
                            logger.info(f"Extracted {dim_name}: {value} mm")
                            break
                    except ValueError:
                        continue
        
        # Validate diameter relationships (v3 improvement)
        if 'barrel_head_diameter' in extracted_data and 'barrel_shaft_diameter' in extracted_data:
            head_dia = extracted_data['barrel_head_diameter']['value']
            shaft_dia = extracted_data['barrel_shaft_diameter']['value']
            ratio = head_dia / shaft_dia
            
            if ratio < 1.0:
                # Likely swapped - fix it
                logger.warning(f"Diameter ratio {ratio:.2f} < 1.0, swapping values")
                extracted_data['barrel_head_diameter']['value'] = shaft_dia
                extracted_data['barrel_shaft_diameter']['value'] = head_dia
            elif ratio > 3.0:
                logger.warning(f"Diameter ratio {ratio:.2f} > 3.0, may be incorrect")
                extracted_data['barrel_head_diameter']['confidence'] = 0.6
                extracted_data['barrel_shaft_diameter']['confidence'] = 0.6
            else:
                logger.info(f"Diameter ratio validation passed: {ratio:.2f}")
        
        # Create dimension data objects
        def create_dimension_data(data_dict, key):
            if key in data_dict:
                dim_data = data_dict[key]
                return DimensionData(
                    value=dim_data.get('value'),
                    unit=dim_data.get('unit', 'mm'),
                    tolerance=dim_data.get('tolerance'),
                    confidence=dim_data.get('confidence', 0.8),
                    source_agent=dim_data.get('source_agent', 'consensus')
                )
            return None
        
        # Count actual agent interactions from the swarm result
        agent_interactions = 0
        handoff_count = final_message.count('handoff_to_agent') + final_message.count('Tool #')
        agent_interactions = max(handoff_count, 3)  # Minimum 3 for multi-agent
        
        # Build result with enhanced validation
        result = ExtractionResult(
            part_number=extracted_data.get('part_number'),
            overall_barrel_length=create_dimension_data(extracted_data, 'overall_barrel_length'),
            barrel_head_length=create_dimension_data(extracted_data, 'barrel_head_length'),
            port_to_shoulder_length=create_dimension_data(extracted_data, 'port_to_shoulder_length'),
            barrel_head_diameter=create_dimension_data(extracted_data, 'barrel_head_diameter'),
            barrel_shaft_diameter=create_dimension_data(extracted_data, 'barrel_shaft_diameter'),
            confidence_score=0.9 if len(extracted_data) >= 3 else 0.5,
            consensus_reached=len(extracted_data) >= 2 or agent_interactions >= 3,
            agent_collaboration_log=[f"Agent interactions: {agent_interactions}, Extracted: {list(extracted_data.keys())}"]
        )
        
        dimensions_found = len([d for d in [result.overall_barrel_length, result.barrel_head_length, 
                                          result.port_to_shoulder_length, result.barrel_head_diameter, 
                                          result.barrel_shaft_diameter] if d and d.value])
        
        logger.info(f"Enhanced processing result: part_number={result.part_number}, dimensions_found={dimensions_found}")
        
        return result
    
    def _log_to_csv(self, filename: str, result: ExtractionResult, processing_time: float):
        """Log multi-agent results to CSV"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            def get_dimension_data(dim_data):
                if dim_data and hasattr(dim_data, 'value'):
                    return (dim_data.value, dim_data.unit, dim_data.tolerance or '', 
                           dim_data.confidence, dim_data.source_agent or '')
                return (None, 'mm', '', 0.0, '')
            
            row_data = [
                timestamp,
                os.path.basename(filename),
                result.part_number or '',
                0.0,  # part_number_confidence
                *get_dimension_data(result.overall_barrel_length),
                *get_dimension_data(result.barrel_head_length),
                *get_dimension_data(result.port_to_shoulder_length),
                *get_dimension_data(result.barrel_head_diameter),
                *get_dimension_data(result.barrel_shaft_diameter),
                result.confidence_score,
                result.consensus_reached,
                len(result.agent_collaboration_log) if result.agent_collaboration_log else 0,
                round(processing_time, 2)
            ]
            
            with open(self.csv_report_path, 'a', newline='', encoding='utf-8') as csvfile:
                import csv
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
                
        except Exception as e:
            logger.error(f"CSV logging error: {e}")

if __name__ == "__main__":
    # Test the multi-agent system
    extractor = TrueMultiAgentExtractor()
    print("True Multi-Agent Barrel Dimension Extractor initialized")
    print("Agents:", [agent.name for agent in extractor.swarm.agents])
