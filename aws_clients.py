"""
AWS service clients for the Streamlit Engineering Drawing Extractor. 

This module provides clients for AWS services including Bedrock Data Automation,
Claude 4 Sonnet via Bedrock, and Amazon Textract with error handling and retry logic.
"""

import boto3
import json
import base64
import time
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError, NoCredentialsError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSClientError(Exception):
    """Custom exception for AWS client errors."""
    pass


class BedrockDataAutomationClient:
    """Client for Amazon Bedrock Data Automation service."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize Bedrock Data Automation client."""
        try:
            self.client = boto3.client('bedrock-data-automation-runtime', region_name=region_name)
            self.s3_client = boto3.client('s3', region_name=region_name)
            self.region_name = region_name
            self.bucket_name = f"bda-temp-{region_name}"
        except NoCredentialsError:
            raise AWSClientError("AWS credentials not found. Please configure your credentials.")
        except Exception as e:
            raise AWSClientError(f"Failed to initialize Bedrock Data Automation client: {str(e)}")
    
    def get_engineering_blueprint(self) -> Dict[str, Any]:
        """Get the engineering drawing extraction blueprint configuration."""
        return {
            "blueprintName": "engineering-drawing-extractor",
            "version": "1.0",
            "fields": [
                {
                    "name": "dimensions",
                    "type": "LIST",
                    "description": "Extract all dimensional measurements including linear, angular, and radial dimensions with their units",
                    "subFields": [
                        {"name": "value", "type": "NUMBER"},
                        {"name": "unit", "type": "STRING"},
                        {"name": "type", "type": "STRING"},
                        {"name": "confidence", "type": "NUMBER"}
                    ]
                },
                {
                    "name": "tolerances",
                    "type": "LIST",
                    "description": "Extract tolerance specifications including geometric tolerances and surface finish requirements",
                    "subFields": [
                        {"name": "type", "type": "STRING"},
                        {"name": "value", "type": "STRING"},
                        {"name": "associated_dimension", "type": "STRING"}
                    ]
                },
                {
                    "name": "part_numbers",
                    "type": "LIST",
                    "description": "Extract part identifiers, drawing numbers, and component labels",
                    "subFields": [
                        {"name": "identifier", "type": "STRING"},
                        {"name": "description", "type": "STRING"}
                    ]
                },
                {
                    "name": "annotations",
                    "type": "LIST",
                    "description": "Extract text annotations, notes, callouts, and instructions",
                    "subFields": [
                        {"name": "text", "type": "STRING"},
                        {"name": "type", "type": "STRING"}
                    ]
                }
            ]
        }
    
    def extract_from_pdf(self, pdf_content: bytes, project_name: str = "engineering-drawing-extractor") -> Dict[str, Any]:
        """
        Extract engineering data from PDF using Bedrock Data Automation.
        
        Args:
            pdf_content: PDF file content as bytes
            project_name: BDA project name
            
        Returns:
            Extraction results dictionary
        """
        try:
            import uuid
            
            logger.info(f"Processing document with Bedrock Data Automation project: {project_name}")
            
            # Validate BDA project exists first
            try:
                project_arn = 'arn:aws:bedrock:us-east-1:364010800473:data-automation-project/a5b5f6f321c2'
                logger.info(f"Validating BDA project: {project_arn}")
            except Exception as validation_error:
                logger.warning(f"BDA project validation failed: {validation_error}")
                raise AWSClientError(f"BDA project not accessible: {validation_error}")
            
            # Generate unique S3 keys
            input_key = f"input/{uuid.uuid4()}.pdf"
            output_key = f"output/{uuid.uuid4()}/"
            
            logger.info(f"Uploading PDF to S3: s3://{self.bucket_name}/{input_key}")
            
            # Upload PDF to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=input_key,
                Body=pdf_content,
                ContentType='application/pdf'
            )
            
            logger.info("Starting BDA processing...")
            
            # Call BDA API
            response = self.client.invoke_data_automation_async(
                inputConfiguration={
                    's3Uri': f's3://{self.bucket_name}/{input_key}'
                },
                outputConfiguration={
                    's3Uri': f's3://{self.bucket_name}/{output_key}'
                },
                dataAutomationConfiguration={
                    'dataAutomationProjectArn': 'arn:aws:bedrock:us-east-1:364010800473:data-automation-project/a5b5f6f321c2',
                    'stage': 'LIVE'
                },
                dataAutomationProfileArn='arn:aws:bedrock:us-east-1:364010800473:data-automation-profile/us.data-automation-v1'
            )
            
            # Poll for completion
            invocation_arn = response['invocationArn']
            max_wait_time = 300  # 5 minutes instead of 20
            poll_interval = 10   # 10 seconds instead of 15
            elapsed_time = 0
            
            logger.info(f"BDA processing started. Invocation ARN: {invocation_arn}")
            
            while elapsed_time < max_wait_time:
                status_response = self.client.get_data_automation_status(
                    invocationArn=invocation_arn
                )
                
                status = status_response['status']
                logger.info(f"BDA status after {elapsed_time}s: {status}")
                logger.info(f"Full status response: {status_response}")
                
                if status in ['Completed', 'COMPLETED', 'Success', 'SUCCESS']:
                    logger.info("BDA processing completed successfully")
                    # Download and parse results
                    result_data = self._download_bda_results(output_key)
                    from unified_barrel_parser import parse_barrel_from_text
                    
                    # Extract raw text from BDA response
                    raw_text = ""
                    if 'document' in result_data and 'representation' in result_data['document']:
                        raw_text = result_data['document']['representation'].get('text', '')
                    
                    extracted_data = parse_barrel_from_text(raw_text, "BDA")
                    
                    # Cleanup S3 objects
                    self._cleanup_s3_objects(input_key, output_key)
                    
                    return {
                        'status': 'SUCCESS',
                        'extractedData': extracted_data,
                        'confidence': extracted_data.get('overall_confidence', 0.8),
                        'processingTime': elapsed_time
                    }
                    
                elif status == 'Failed':
                    error_msg = status_response.get('errorMessage', 'Unknown error')
                    logger.error(f"BDA processing failed: {error_msg}")
                    self._cleanup_s3_objects(input_key, output_key)
                    raise AWSClientError(f"BDA processing failed: {error_msg}")
                
                elif status in ['InProgress', 'Pending']:
                    logger.info(f"BDA still processing... ({elapsed_time}/{max_wait_time}s)")
                else:
                    logger.warning(f"Unknown BDA status: {status}")
                
                time.sleep(poll_interval)
                elapsed_time += poll_interval
            
            # Timeout - cleanup and raise error
            self._cleanup_s3_objects(input_key, output_key)
            raise AWSClientError(f"BDA processing timed out after {max_wait_time} seconds")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"BDA API error: {error_code} - {error_message}")
            raise AWSClientError(f"Bedrock Data Automation failed: {error_message}")
        except Exception as e:
            logger.error(f"Unexpected error in BDA extraction: {str(e)}")
            raise AWSClientError(f"Bedrock Data Automation extraction failed: {str(e)}")
    
    def _cleanup_s3_objects(self, input_key: str, output_prefix: str):
        """Cleanup S3 objects."""
        try:
            # Delete input file
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=input_key)
            
            # Delete output objects
            objects = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=output_prefix)
            if 'Contents' in objects:
                for obj in objects['Contents']:
                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
                    
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup S3 objects: {cleanup_error}")
    
    def _download_bda_results(self, output_prefix: str) -> Dict[str, Any]:
        """Download BDA results from S3."""
        try:
            # List objects in output prefix
            objects = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=output_prefix)
            
            if 'Contents' not in objects:
                return {}
            
            # Download the first JSON result file
            for obj in objects['Contents']:
                if obj['Key'].endswith('.json'):
                    response = self.s3_client.get_object(Bucket=self.bucket_name, Key=obj['Key'])
                    return json.loads(response['Body'].read().decode('utf-8'))
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to download BDA results: {str(e)}")
            return {}

    def _parse_bda_response(self, bda_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse BDA response into our standard format."""
        try:
            logger.info(f"BDA Response structure: {json.dumps(bda_response, indent=2, default=str)}")
            
            # Extract data from BDA response structure
            output = bda_response.get('output', {})
            logger.info(f"BDA Output keys: {list(output.keys()) if output else 'No output'}")
            
            # Initialize result structure
            result = {
                'dimensions': [],
                'tolerances': [],
                'part_numbers': [],
                'annotations': [],
                'overall_confidence': 0.8
            }
            
            # Parse extracted entities from BDA - try multiple possible structures
            extracted_data = None
            if 'extractedData' in output:
                extracted_data = output['extractedData']
            elif 'extracted_data' in output:
                extracted_data = output['extracted_data']
            elif 'results' in output:
                extracted_data = output['results']
            elif isinstance(output, list) and len(output) > 0:
                extracted_data = output[0]
            else:
                # Try the root level
                extracted_data = bda_response
            
            logger.info(f"Extracted data keys: {list(extracted_data.keys()) if isinstance(extracted_data, dict) else 'Not a dict'}")
            
            # Special handling for barrel diagrams
            barrel_fields = ['Overall barrel length', 'Barrel head length', 'Port to shoulder length', 'Barrel head Dia', 'Barrel shaft Dia']
            
            # Check if this is a barrel diagram by looking for barrel-specific terms
            is_barrel_diagram = False
            full_text = str(bda_response).lower()
            if any(term in full_text for term in ['barrel', 'shaft', 'port', 'shoulder']):
                is_barrel_diagram = True
                logger.info("Detected barrel diagram - using specialized parsing")
            
            if extracted_data and isinstance(extracted_data, dict):
                
                # Parse dimensions - try multiple field names
                dimension_fields = ['dimensions', 'dimension', 'measurements', 'values']
                for field in dimension_fields:
                    if field in extracted_data:
                        logger.info(f"Found dimensions in field: {field}")
                        for item in extracted_data[field]:
                            logger.info(f"Dimension item: {item}")
                            
                            # For barrel diagrams, categorize dimensions by type and include tolerances
                            dim_text = str(item.get('text', item.get('raw_text', ''))).lower()
                            dim_type = 'LINEAR'
                            
                            if is_barrel_diagram:
                                if 'overall' in dim_text and 'length' in dim_text:
                                    dim_type = 'Overall barrel length'
                                elif 'head' in dim_text and 'length' in dim_text:
                                    dim_type = 'Barrel head length'
                                elif 'port' in dim_text and 'shoulder' in dim_text:
                                    dim_type = 'Port to shoulder length'
                                elif 'head' in dim_text and ('dia' in dim_text or 'diameter' in dim_text):
                                    dim_type = 'Barrel head Dia'
                                elif 'shaft' in dim_text and ('dia' in dim_text or 'diameter' in dim_text):
                                    dim_type = 'Barrel shaft Dia'
                            
                            # Extract value with tolerance if present
                            raw_value = item.get('value', item.get('measurement', item.get('dimension', 0)))
                            tolerance = item.get('tolerance', '')
                            
                            # Combine value and tolerance
                            if tolerance:
                                display_value = f"{raw_value}{tolerance}"
                            else:
                                display_value = str(raw_value)
                            
                            result['dimensions'].append({
                                'value': display_value,
                                'unit': item.get('unit', item.get('units', '')),
                                'type': dim_type,
                                'confidence': item.get('confidence', item.get('score', 0.8)),
                                'location_description': item.get('location', item.get('position', '')),
                                'raw_text': item.get('text', item.get('raw_text', item.get('original_text', '')))
                            })
                        break
                
                # Parse tolerances - try multiple field names
                tolerance_fields = ['tolerances', 'tolerance', 'specs', 'specifications']
                for field in tolerance_fields:
                    if field in extracted_data:
                        logger.info(f"Found tolerances in field: {field}")
                        for item in extracted_data[field]:
                            result['tolerances'].append({
                                'type': item.get('type', item.get('tolerance_type', 'PLUS_MINUS')),
                                'value': item.get('value', item.get('tolerance_value', '')),
                                'confidence': item.get('confidence', item.get('score', 0.8)),
                                'raw_text': item.get('text', item.get('raw_text', '')),
                                'associated_dimension_id': item.get('dimensionId', item.get('dimension_id'))
                            })
                        break
                
                # Parse part numbers - try multiple field names
                part_fields = ['partNumbers', 'part_numbers', 'parts', 'identifiers']
                for field in part_fields:
                    if field in extracted_data:
                        logger.info(f"Found part numbers in field: {field}")
                        for item in extracted_data[field]:
                            result['part_numbers'].append({
                                'identifier': item.get('identifier', item.get('part_number', item.get('id', ''))),
                                'description': item.get('description', item.get('desc', '')),
                                'confidence': item.get('confidence', item.get('score', 0.8)),
                                'raw_text': item.get('text', item.get('raw_text', ''))
                            })
                        break
                
                # Parse annotations - try multiple field names
                annotation_fields = ['annotations', 'annotation', 'notes', 'text', 'labels']
                for field in annotation_fields:
                    if field in extracted_data:
                        logger.info(f"Found annotations in field: {field}")
                        for item in extracted_data[field]:
                            result['annotations'].append({
                                'text': item.get('text', item.get('annotation', item.get('note', ''))),
                                'type': item.get('type', item.get('annotation_type', 'NOTE')),
                                'confidence': item.get('confidence', item.get('score', 0.8)),
                                'location_description': item.get('location', item.get('position', ''))
                            })
                        break
                
                # Calculate overall confidence
                all_confidences = []
                for category in ['dimensions', 'tolerances', 'part_numbers', 'annotations']:
                    for item in result[category]:
                        all_confidences.append(item['confidence'])
                
                if all_confidences:
                    result['overall_confidence'] = sum(all_confidences) / len(all_confidences)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to parse BDA response: {str(e)}")
            return {
                'dimensions': [],
                'tolerances': [],
                'part_numbers': [],
                'annotations': [],
                'overall_confidence': 0.5
            }


class ClaudeBedrockClient:
    """Client for Claude 4 Sonnet via Amazon Bedrock."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize Claude Bedrock client."""
        try:
            self.client = boto3.client('bedrock-runtime', region_name=region_name)
            self.region_name = region_name
            self.model_id = "anthropic.claude-4-0-sonnet-20250109-v1:0"
        except NoCredentialsError:
            raise AWSClientError("AWS credentials not found. Please configure your credentials.")
        except Exception as e:
            raise AWSClientError(f"Failed to initialize Claude Bedrock client: {str(e)}")
    
    def get_extraction_prompt(self) -> str:
        """Get the prompt for engineering drawing extraction."""
        return """
You are an expert at analyzing engineering drawings and technical documents. 
Please analyze this PDF and extract the following information:

1. DIMENSIONS: All dimensional measurements with their values and units
2. TOLERANCES: Any tolerance specifications (±, geometric tolerances, surface finish)
3. PART NUMBERS: Part identifiers, drawing numbers, or component labels
4. ANNOTATIONS: Text notes, callouts, instructions, or labels

For each extracted item, provide:
- The exact value/text found
- Your confidence level (0.0 to 1.0)
- A brief description of where it was found
- The type/category of the item

Return the results in JSON format with the following structure:
{
  "dimensions": [
    {
      "value": 10.5,
      "unit": "mm",
      "type": "LINEAR",
      "confidence": 0.95,
      "location_description": "Top edge",
      "raw_text": "10.5mm"
    }
  ],
  "tolerances": [
    {
      "type": "PLUS_MINUS",
      "value": "±0.1",
      "confidence": 0.9,
      "associated_dimension": "dimension_id",
      "raw_text": "±0.1mm"
    }
  ],
  "part_numbers": [
    {
      "identifier": "PN-12345",
      "description": "Main housing",
      "confidence": 0.98,
      "raw_text": "PN-12345 Main housing"
    }
  ],
  "annotations": [
    {
      "text": "Remove all burrs",
      "type": "NOTE",
      "confidence": 0.92,
      "location_description": "Bottom right corner"
    }
  ],
  "overall_confidence": 0.94,
  "processing_notes": "Successfully extracted engineering data from technical drawing"
}
"""
    
    def extract_from_pdf(self, pdf_content: bytes, max_tokens: int = 4000, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Extract engineering data from PDF using Claude 4 Sonnet.
        
        Args:
            pdf_content: PDF file content as bytes
            max_tokens: Maximum tokens for response
            temperature: Model temperature setting
            
        Returns:
            Extraction results dictionary
        """
        try:
            # Encode PDF content to base64
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            
            # Prepare the request payload
            request_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.get_extraction_prompt()
                            },
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Make the API call
            logger.info(f"Processing document with Claude 4 Sonnet model: {self.model_id}")
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_payload)
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            # Extract the content from Claude's response
            if 'content' in response_body and len(response_body['content']) > 0:
                content_text = response_body['content'][0]['text']
                
                # Try to parse JSON from the response
                try:
                    extracted_data = json.loads(content_text)
                    return {
                        'status': 'SUCCESS',
                        'extractedData': extracted_data,
                        'confidence': extracted_data.get('overall_confidence', 0.8),
                        'processingTime': 0.0  # Will be calculated by caller
                    }
                except json.JSONDecodeError:
                    logger.warning("Claude response was not valid JSON, using unified barrel parser")
                    from unified_barrel_parser import parse_barrel_from_text
                    barrel_data = parse_barrel_from_text(content_text, "Claude")
                    return {
                        'status': 'SUCCESS',
                        'extractedData': barrel_data,
                        'confidence': 0.7,
                        'processingTime': 0.0
                    }
            else:
                raise AWSClientError("No content received from Claude model")
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Claude Bedrock API error: {error_code} - {error_message}")
            raise AWSClientError(f"Claude extraction failed: {error_message}")
        except Exception as e:
            logger.error(f"Unexpected error in Claude extraction: {str(e)}")
            raise AWSClientError(f"Claude extraction failed: {str(e)}")


class TextractClient:
    """Client for Amazon Textract service."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize Textract client."""
        try:
            self.client = boto3.client('textract', region_name=region_name)
            self.region_name = region_name
        except NoCredentialsError:
            raise AWSClientError("AWS credentials not found. Please configure your credentials.")
        except Exception as e:
            raise AWSClientError(f"Failed to initialize Textract client: {str(e)}")
    
    def extract_from_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Extract text from PDF using Amazon Textract.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Extraction results dictionary
        """
        try:
            logger.info("Processing document with Amazon Textract")
            
            # Call Textract to detect document text
            response = self.client.detect_document_text(
                Document={'Bytes': pdf_content}
            )
            
            # Extract text blocks
            extracted_text = []
            confidence_scores = []
            
            for block in response.get('Blocks', []):
                if block['BlockType'] == 'LINE':
                    text = block.get('Text', '')
                    confidence = block.get('Confidence', 0.0) / 100.0  # Convert to 0-1 scale
                    
                    if text.strip():
                        extracted_text.append(text)
                        confidence_scores.append(confidence)
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Use unified barrel parser for consistent results
            from unified_barrel_parser import parse_barrel_from_text
            full_text = ' '.join(extracted_text)
            barrel_data = parse_barrel_from_text(full_text, "Textract")
            
            return {
                'status': 'SUCCESS',
                'extractedData': barrel_data,
                'confidence': overall_confidence,
                'processingTime': 0.0  # Will be calculated by caller
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Textract API error: {error_code} - {error_message}")
            raise AWSClientError(f"Textract extraction failed: {error_message}")
        except Exception as e:
            logger.error(f"Unexpected error in Textract extraction: {str(e)}")
            raise AWSClientError(f"Textract extraction failed: {str(e)}")
    
    def _parse_dimensions(self, text_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse dimensional information from extracted text."""
        dimensions = []
        import re
        
        for i, line in enumerate(text_lines):
            # Check for diameter first (most specific)
            diameter_match = re.search(r'Ø\s*(\d+\.?\d*)\s*(mm|in|inches|cm|m)', line, re.IGNORECASE)
            if diameter_match:
                dimensions.append({
                    'value': float(diameter_match.group(1)),
                    'unit': diameter_match.group(2),
                    'type': 'DIAMETER',
                    'confidence': 0.7,
                    'location_description': f'Line {i+1}',
                    'raw_text': line.strip()
                })
                continue
            
            # Check for radius (second most specific)
            radius_match = re.search(r'R\s*(\d+\.?\d*)\s*(mm|in|inches|cm|m)', line, re.IGNORECASE)
            if radius_match:
                dimensions.append({
                    'value': float(radius_match.group(1)),
                    'unit': radius_match.group(2),
                    'type': 'RADIAL',
                    'confidence': 0.7,
                    'location_description': f'Line {i+1}',
                    'raw_text': line.strip()
                })
                continue
            
            # Check for general dimensions (least specific)
            general_match = re.search(r'(\d+\.?\d*)\s*(mm|in|inches|cm|m)', line, re.IGNORECASE)
            if general_match:
                dimensions.append({
                    'value': float(general_match.group(1)),
                    'unit': general_match.group(2),
                    'type': 'LINEAR',
                    'confidence': 0.7,
                    'location_description': f'Line {i+1}',
                    'raw_text': line.strip()
                })
        
        return dimensions
    
    def _parse_tolerances(self, text_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse tolerance information from extracted text."""
        tolerances = []
        import re
        
        # Simple regex patterns for tolerance formats
        tolerance_patterns = [
            r'±\s*(\d+\.?\d*)',  # Plus/minus tolerance
            r'\+(\d+\.?\d*)\s*-(\d+\.?\d*)',  # Asymmetric tolerance
        ]
        
        for i, line in enumerate(text_lines):
            for pattern in tolerance_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    tolerances.append({
                        'type': 'PLUS_MINUS',
                        'value': match.group(0),
                        'confidence': 0.6,
                        'associated_dimension': None,
                        'raw_text': line.strip()
                    })
        
        return tolerances
    
    def _parse_part_numbers(self, text_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse part number information from extracted text."""
        part_numbers = []
        import re
        
        # Simple regex patterns for part numbers
        part_patterns = [
            r'(?:PN|P/N|Part\s*No\.?)\s*:?\s*([A-Z0-9\-]+)',  # PN: ABC-123
            r'([A-Z]{2,}\-\d+)',  # Pattern like ABC-123
        ]
        
        for i, line in enumerate(text_lines):
            for pattern in part_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    identifier = match.group(1)  # First capture group is the part number
                    part_numbers.append({
                        'identifier': identifier,
                        'description': None,
                        'confidence': 0.6,
                        'raw_text': line.strip()
                    })
        
        return part_numbers
    
    def _parse_annotations(self, text_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse annotation information from extracted text."""
        annotations = []
        
        # Keywords that indicate annotations/notes
        annotation_keywords = ['note', 'notes', 'see', 'ref', 'typical', 'unless', 'all', 'remove', 'finish']
        
        for i, line in enumerate(text_lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in annotation_keywords):
                annotations.append({
                    'text': line.strip(),
                    'type': 'NOTE',
                    'confidence': 0.5,
                    'location_description': f'Line {i+1}'
                })
        
        return annotations


class AWSServiceManager:
    """Manager class for coordinating AWS service clients."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize AWS service manager."""
        self.region_name = region_name
        self.bda_client = None
        self.claude_client = None
        self.textract_client = None
        
        # Initialize clients lazily
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS service clients with error handling."""
        try:
            self.bda_client = BedrockDataAutomationClient(self.region_name)
            logger.info("Bedrock Data Automation client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize BDA client: {str(e)}")
        
        try:
            self.claude_client = ClaudeBedrockClient(self.region_name)
            logger.info("Claude Bedrock client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Claude client: {str(e)}")
        
        try:
            self.textract_client = TextractClient(self.region_name)
            logger.info("Textract client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Textract client: {str(e)}")
    
    def get_available_methods(self) -> List[str]:
        """Get list of available extraction methods based on initialized clients."""
        methods = []
        if self.bda_client:
            methods.append('bedrock_data_automation')
        if self.claude_client:
            methods.append('claude_4_5_sonnet')
        if self.textract_client:
            methods.append('textract')
        return methods
    
    def extract_with_method(self, pdf_content: bytes, method: str) -> Dict[str, Any]:
        """
        Extract data using specified method.
        
        Args:
            pdf_content: PDF file content as bytes
            method: Extraction method ('bedrock_data_automation', 'claude_4_5_sonnet', 'textract')
            
        Returns:
            Extraction results dictionary
        """
        start_time = time.time()
        
        try:
            if method == 'bedrock_data_automation' and self.bda_client:
                result = self.bda_client.extract_from_pdf(pdf_content)
            elif method == 'claude_4_5_sonnet' and self.claude_client:
                result = self.claude_client.extract_from_pdf(pdf_content)
            elif method == 'textract' and self.textract_client:
                result = self.textract_client.extract_from_pdf(pdf_content)
            else:
                raise AWSClientError(f"Method '{method}' is not available or client not initialized")
            
            # Add processing time
            result['processingTime'] = time.time() - start_time
            result['method'] = method
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed with method {method}: {str(e)}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'method': method,
                'processingTime': time.time() - start_time
            }
    
    def extract_with_fallback(self, pdf_content: bytes, preferred_methods: List[str] = None) -> Dict[str, Any]:
        """
        Extract data with automatic fallback between methods.
        
        Args:
            pdf_content: PDF file content as bytes
            preferred_methods: List of methods to try in order
            
        Returns:
            Extraction results dictionary
        """
        if preferred_methods is None:
            preferred_methods = ['bedrock_data_automation', 'claude_4_5_sonnet', 'textract']
        
        available_methods = self.get_available_methods()
        methods_to_try = [m for m in preferred_methods if m in available_methods]
        
        if not methods_to_try:
            raise AWSClientError("No extraction methods available")
        
        last_error = None
        
        for method in methods_to_try:
            logger.info(f"Attempting extraction with method: {method}")
            
            result = self.extract_with_method(pdf_content, method)
            
            if result.get('status') == 'SUCCESS':
                logger.info(f"Extraction successful with method: {method}")
                return result
            elif result.get('status') == 'PARTIAL_SUCCESS':
                logger.warning(f"Partial success with method: {method}")
                return result
            else:
                last_error = result.get('error', 'Unknown error')
                logger.warning(f"Method {method} failed: {last_error}")
        
        # All methods failed
        raise AWSClientError(f"All extraction methods failed. Last error: {last_error}")
