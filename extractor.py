"""
Core extraction engine for the Streamlit Engineering Drawing Extractor. 

This module provides the main extraction functionality with PDF validation,
preprocessing, multi-service support, and result processing.
"""

import io
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import PyPDF2
from PIL import Image
import fitz  # PyMuPDF for better PDF handling

from models import (
    ExtractionResult, Dimension, Tolerance, PartNumber, Annotation,
    create_dimension, create_tolerance, create_part_number, create_annotation,
    generate_unique_id
)
from aws_clients import AWSServiceManager, AWSClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFValidationError(Exception):
    """Exception raised for PDF validation errors."""
    pass


class ExtractionEngine:
    """Core extraction engine with multi-service support."""
    
    def __init__(self, aws_region: str = "us-east-1", max_file_size_mb: int = 100):
        """
        Initialize the extraction engine.
        
        Args:
            aws_region: AWS region for service clients
            max_file_size_mb: Maximum allowed file size in MB
        """
        self.aws_region = aws_region
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.aws_manager = AWSServiceManager(aws_region)
        
        # Configuration
        self.confidence_threshold = 0.5  # Lowered to see Textract results
        self.processing_timeout = 300  # 5 minutes
        
        logger.info(f"Extraction engine initialized with region: {aws_region}")
    
    def validate_pdf(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Validate PDF file for processing.
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Original filename
            
        Returns:
            Validation result dictionary
            
        Raises:
            PDFValidationError: If validation fails
        """
        try:
            # Check file size
            if len(pdf_content) > self.max_file_size_bytes:
                raise PDFValidationError(
                    f"File size ({len(pdf_content) / 1024 / 1024:.1f}MB) exceeds "
                    f"maximum allowed size ({self.max_file_size_bytes / 1024 / 1024}MB)"
                )
            
            # Check file extension
            if not filename.lower().endswith('.pdf'):
                raise PDFValidationError("File must be a PDF document")
            
            # Validate PDF structure using PyPDF2
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
                num_pages = len(pdf_reader.pages)
                
                if num_pages == 0:
                    raise PDFValidationError("PDF file contains no pages")
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    raise PDFValidationError("Encrypted PDFs are not supported")
                
            except Exception as e:
                if isinstance(e, PDFValidationError):
                    raise
                raise PDFValidationError(f"Invalid PDF structure: {str(e)}")
            
            # Additional validation using PyMuPDF for better error detection
            try:
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                
                # Check if document can be opened
                if doc.page_count == 0:
                    raise PDFValidationError("PDF document has no readable pages")
                
                # Check first page for basic content
                first_page = doc[0]
                text_content = first_page.get_text()
                
                doc.close()
                
                return {
                    'valid': True,
                    'num_pages': num_pages,
                    'file_size_mb': len(pdf_content) / 1024 / 1024,
                    'has_text': len(text_content.strip()) > 0,
                    'filename': filename
                }
                
            except Exception as e:
                raise PDFValidationError(f"PDF processing error: {str(e)}")
            
        except PDFValidationError:
            raise
        except Exception as e:
            raise PDFValidationError(f"Unexpected validation error: {str(e)}")
    
    def preprocess_pdf(self, pdf_content: bytes) -> bytes:
        """
        Preprocess PDF for optimal extraction.
        
        Args:
            pdf_content: Original PDF content
            
        Returns:
            Preprocessed PDF content
        """
        try:
            # For now, return original content
            # In a production system, you might:
            # - Optimize image quality
            # - Remove unnecessary elements
            # - Standardize page orientation
            # - Enhance text clarity
            
            logger.info("PDF preprocessing completed")
            return pdf_content
            
        except Exception as e:
            logger.warning(f"PDF preprocessing failed, using original: {str(e)}")
            return pdf_content
    
    def calculate_confidence_score(self, extracted_data: Dict[str, Any], method: str) -> float:
        """
        Calculate overall confidence score for extraction results.
        
        Args:
            extracted_data: Raw extraction data from AWS service
            method: Extraction method used
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        try:
            if method == 'bedrock_data_automation':
                # BDA typically provides structured confidence
                return extracted_data.get('confidence', 0.8)
            
            elif method == 'claude_4_5_sonnet':
                # Claude provides overall confidence in response
                if 'overall_confidence' in extracted_data:
                    return extracted_data['overall_confidence']
                
                # Calculate from individual item confidences
                all_confidences = []
                for category in ['dimensions', 'tolerances', 'part_numbers', 'annotations']:
                    items = extracted_data.get(category, [])
                    for item in items:
                        if isinstance(item, dict) and 'confidence' in item:
                            all_confidences.append(item['confidence'])
                
                return sum(all_confidences) / len(all_confidences) if all_confidences else 0.5
            
            elif method == 'textract':
                # Textract confidence is calculated during parsing
                return extracted_data.get('confidence', 0.6)
            
            else:
                return 0.5  # Default confidence for unknown methods
                
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    def parse_extraction_results(self, raw_data: Dict[str, Any], method: str) -> Tuple[List, List, List, List]:
        """
        Parse raw extraction results into structured data models.
        
        Args:
            raw_data: Raw extraction data from AWS service
            method: Extraction method used
            
        Returns:
            Tuple of (dimensions, tolerances, part_numbers, annotations) lists
        """
        try:
            extracted_data = raw_data.get('extractedData', {})
            
            # Parse dimensions
            dimensions = []
            for dim_data in extracted_data.get('dimensions', []):
                try:
                    dimension = create_dimension(
                        value=dim_data.get('value', 0) if not isinstance(dim_data.get('value', 0), str) or not any(c in str(dim_data.get('value', 0)) for c in ['+', '±', '-']) else dim_data.get('value', 0),
                        unit=dim_data.get('unit', ''),
                        type=dim_data.get('type', 'LINEAR'),
                        confidence=float(dim_data.get('confidence', 0.5)),
                        location_description=dim_data.get('location_description', ''),
                        raw_text=dim_data.get('raw_text', '')
                    )
                    dimensions.append(dimension)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse dimension: {dim_data}, error: {str(e)}")
            
            # Parse tolerances
            tolerances = []
            for tol_data in extracted_data.get('tolerances', []):
                try:
                    tolerance = create_tolerance(
                        type=tol_data.get('type', 'PLUS_MINUS'),
                        value=str(tol_data.get('value', '')),
                        confidence=float(tol_data.get('confidence', 0.5)),
                        raw_text=tol_data.get('raw_text', ''),
                        associated_dimension_id=tol_data.get('associated_dimension_id')
                    )
                    tolerances.append(tolerance)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse tolerance: {tol_data}, error: {str(e)}")
            
            # Parse part numbers
            part_numbers = []
            for part_data in extracted_data.get('part_numbers', []):
                try:
                    part_number = create_part_number(
                        identifier=str(part_data.get('identifier', '')),
                        confidence=float(part_data.get('confidence', 0.5)),
                        raw_text=part_data.get('raw_text', ''),
                        description=part_data.get('description')
                    )
                    part_numbers.append(part_number)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse part number: {part_data}, error: {str(e)}")
            
            # Parse annotations
            annotations = []
            for ann_data in extracted_data.get('annotations', []):
                try:
                    annotation = create_annotation(
                        text=str(ann_data.get('text', '')),
                        type=ann_data.get('type', 'NOTE'),
                        confidence=float(ann_data.get('confidence', 0.5)),
                        location_description=ann_data.get('location_description', '')
                    )
                    annotations.append(annotation)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse annotation: {ann_data}, error: {str(e)}")
            
            logger.info(f"Parsed {len(dimensions)} dimensions, {len(tolerances)} tolerances, "
                       f"{len(part_numbers)} part numbers, {len(annotations)} annotations")
            
            return dimensions, tolerances, part_numbers, annotations
            
        except Exception as e:
            logger.error(f"Failed to parse extraction results: {str(e)}")
            return [], [], [], []
    
    def extract_from_pdf(self, pdf_content: bytes, filename: str, 
                        method: str = "auto", 
                        confidence_filter: bool = True) -> ExtractionResult:
        """
        Extract engineering data from PDF using specified method.
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Original filename
            method: Extraction method ('auto', 'bedrock_data_automation', 'claude_4_5_sonnet', 'textract')
            confidence_filter: Whether to filter low-confidence results
            
        Returns:
            ExtractionResult object with all extracted data
            
        Raises:
            PDFValidationError: If PDF validation fails
            AWSClientError: If extraction fails
        """
        start_time = time.time()
        errors = []
        
        try:
            # Validate PDF
            logger.info(f"Starting extraction for file: {filename}")
            validation_result = self.validate_pdf(pdf_content, filename)
            logger.info(f"PDF validation successful: {validation_result}")
            
            # Preprocess PDF
            processed_content = self.preprocess_pdf(pdf_content)
            
            # Perform extraction based on method
            if method == "auto":
                # Try methods in order of preference
                preferred_methods = ['bedrock_data_automation', 'claude_4_5_sonnet', 'textract']
                raw_result = self.aws_manager.extract_with_fallback(processed_content, preferred_methods)
            else:
                # Use specific method
                raw_result = self.aws_manager.extract_with_method(processed_content, method)
            
            # Check if extraction was successful
            if raw_result.get('status') not in ['SUCCESS', 'PARTIAL_SUCCESS']:
                error_msg = raw_result.get('error', 'Unknown extraction error')
                raise AWSClientError(f"Extraction failed: {error_msg}")
            
            # Parse results into structured data
            dimensions, tolerances, part_numbers, annotations = self.parse_extraction_results(
                raw_result, raw_result.get('method', method)
            )
            
            # Apply confidence filtering if requested
            if confidence_filter:
                dimensions = [d for d in dimensions if d.confidence >= self.confidence_threshold]
                tolerances = [t for t in tolerances if t.confidence >= self.confidence_threshold]
                part_numbers = [p for p in part_numbers if p.confidence >= self.confidence_threshold]
                annotations = [a for a in annotations if a.confidence >= self.confidence_threshold]
                
                logger.info(f"Applied confidence filter (>= {self.confidence_threshold})")
            
            # Calculate overall confidence
            overall_confidence = self.calculate_confidence_score(
                raw_result.get('extractedData', {}), 
                raw_result.get('method', method)
            )
            
            # Add any processing errors
            if raw_result.get('status') == 'PARTIAL_SUCCESS':
                errors.append("Partial extraction - some data may be missing or have low confidence")
            
            # Create final result
            processing_time = time.time() - start_time
            
            result = ExtractionResult(
                filename=filename,
                processing_time=processing_time,
                overall_confidence=overall_confidence,
                dimensions=dimensions,
                tolerances=tolerances,
                part_numbers=part_numbers,
                annotations=annotations,
                extraction_method=raw_result.get('method', method),
                timestamp=datetime.now(),
                errors=errors
            )
            
            logger.info(f"Extraction completed successfully in {processing_time:.2f}s using {result.extraction_method}")
            logger.info(f"Results: {result.get_summary_stats()}")
            
            return result
            
        except PDFValidationError:
            raise
        except AWSClientError:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected extraction error: {str(e)}"
            logger.error(error_msg)
            
            # Return empty result with error
            return ExtractionResult(
                filename=filename,
                processing_time=processing_time,
                overall_confidence=0.0,
                dimensions=[],
                tolerances=[],
                part_numbers=[],
                annotations=[],
                extraction_method=method,
                timestamp=datetime.now(),
                errors=[error_msg]
            )
    
    def get_available_methods(self) -> List[str]:
        """Get list of available extraction methods."""
        methods = self.aws_manager.get_available_methods()
        if methods:
            methods.insert(0, 'auto')  # Add auto method if any services are available
        return methods
    
    def get_extraction_stats(self, result: ExtractionResult) -> Dict[str, Any]:
        """
        Get detailed statistics about extraction results.
        
        Args:
            result: ExtractionResult object
            
        Returns:
            Dictionary with detailed statistics
        """
        stats = result.get_summary_stats()
        
        # Add confidence statistics
        all_items = result.dimensions + result.tolerances + result.part_numbers + result.annotations
        confidences = [item.confidence for item in all_items]
        
        if confidences:
            stats.update({
                'confidence_stats': {
                    'min': min(confidences),
                    'max': max(confidences),
                    'avg': sum(confidences) / len(confidences),
                    'high_confidence_items': len([c for c in confidences if c >= 0.8]),
                    'medium_confidence_items': len([c for c in confidences if 0.5 <= c < 0.8]),
                    'low_confidence_items': len([c for c in confidences if c < 0.5])
                }
            })
        else:
            stats['confidence_stats'] = {
                'min': 0, 'max': 0, 'avg': 0,
                'high_confidence_items': 0,
                'medium_confidence_items': 0,
                'low_confidence_items': 0
            }
        
        # Add processing information
        stats.update({
            'processing_info': {
                'method': result.extraction_method,
                'processing_time': result.processing_time,
                'overall_confidence': result.overall_confidence,
                'has_errors': len(result.errors) > 0,
                'error_count': len(result.errors)
            }
        })
        
        return stats


def create_extraction_engine(aws_region: str = "us-east-1", 
                           max_file_size_mb: int = 100) -> ExtractionEngine:
    """
    Factory function to create an extraction engine instance.
    
    Args:
        aws_region: AWS region for service clients
        max_file_size_mb: Maximum allowed file size in MB
        
    Returns:
        Configured ExtractionEngine instance
    """

    return ExtractionEngine(aws_region, max_file_size_mb)
