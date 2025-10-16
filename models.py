"""
Core data models for the Streamlit Engineering Drawing Extractor. 

This module defines the data structures used throughout the application
for representing extracted engineering drawing information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import json
import uuid


@dataclass
class Dimension:
    """Represents a dimensional measurement extracted from an engineering drawing."""
    
    id: str
    value: Union[float, str]
    unit: str
    type: str  # 'LINEAR', 'ANGULAR', 'RADIAL', 'DIAMETER'
    confidence: float
    location_description: str
    raw_text: str
    
    def __post_init__(self):
        """Validate dimension data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        valid_types = ['LINEAR', 'ANGULAR', 'RADIAL', 'DIAMETER']
        if self.type not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dimension to dictionary for serialization."""
        return {
            'id': self.id,
            'value': self.value,
            'unit': self.unit,
            'type': self.type,
            'confidence': self.confidence,
            'location_description': self.location_description,
            'raw_text': self.raw_text
        }


@dataclass
class Tolerance:
    """Represents a tolerance specification extracted from an engineering drawing."""
    
    id: str
    type: str  # 'PLUS_MINUS', 'GEOMETRIC', 'SURFACE_FINISH'
    value: str
    confidence: float
    associated_dimension_id: Optional[str]
    raw_text: str
    
    def __post_init__(self):
        """Validate tolerance data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        valid_types = ['PLUS_MINUS', 'GEOMETRIC', 'SURFACE_FINISH']
        if self.type not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tolerance to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type,
            'value': self.value,
            'confidence': self.confidence,
            'associated_dimension_id': self.associated_dimension_id,
            'raw_text': self.raw_text
        }


@dataclass
class PartNumber:
    """Represents a part number or identifier extracted from an engineering drawing."""
    
    id: str
    identifier: str
    description: Optional[str]
    confidence: float
    raw_text: str
    
    def __post_init__(self):
        """Validate part number data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert part number to dictionary for serialization."""
        return {
            'id': self.id,
            'identifier': self.identifier,
            'description': self.description,
            'confidence': self.confidence,
            'raw_text': self.raw_text
        }


@dataclass
class Annotation:
    """Represents a text annotation extracted from an engineering drawing."""
    
    id: str
    text: str
    type: str  # 'NOTE', 'CALLOUT', 'LABEL', 'INSTRUCTION'
    confidence: float
    location_description: str
    
    def __post_init__(self):
        """Validate annotation data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        valid_types = ['NOTE', 'CALLOUT', 'LABEL', 'INSTRUCTION']
        if self.type not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert annotation to dictionary for serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'type': self.type,
            'confidence': self.confidence,
            'location_description': self.location_description
        }


@dataclass
class ExtractionResult:
    """Complete result of engineering drawing extraction process."""
    
    filename: str
    processing_time: float
    overall_confidence: float
    dimensions: List[Dimension]
    tolerances: List[Tolerance]
    part_numbers: List[PartNumber]
    annotations: List[Annotation]
    extraction_method: str
    timestamp: datetime
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate extraction result data after initialization."""
        if not 0.0 <= self.overall_confidence <= 1.0:
            raise ValueError("Overall confidence must be between 0.0 and 1.0")
        
        valid_methods = ['bedrock_data_automation', 'claude_3_5_sonnet', 'textract', 'auto']
        if self.extraction_method not in valid_methods:
            raise ValueError(f"Extraction method must be one of {valid_methods}")
    
    def get_total_items(self) -> int:
        """Get total number of extracted items."""
        return len(self.dimensions) + len(self.tolerances) + len(self.part_numbers) + len(self.annotations)
    
    def get_summary_stats(self) -> Dict[str, int]:
        """Get summary statistics of extracted items."""
        return {
            'dimensions': len(self.dimensions),
            'tolerances': len(self.tolerances),
            'part_numbers': len(self.part_numbers),
            'annotations': len(self.annotations),
            'total_items': self.get_total_items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert extraction result to dictionary for serialization."""
        return {
            'filename': self.filename,
            'processing_time': self.processing_time,
            'overall_confidence': self.overall_confidence,
            'dimensions': [dim.to_dict() for dim in self.dimensions],
            'tolerances': [tol.to_dict() for tol in self.tolerances],
            'part_numbers': [part.to_dict() for part in self.part_numbers],
            'annotations': [ann.to_dict() for ann in self.annotations],
            'extraction_method': self.extraction_method,
            'timestamp': self.timestamp.isoformat(),
            'errors': self.errors,
            'summary_stats': self.get_summary_stats()
        }
    
    def to_json(self) -> str:
        """Convert extraction result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def generate_unique_id() -> str:
    """Generate a unique identifier for data objects."""
    return str(uuid.uuid4())


def create_dimension(value: float, unit: str, type: str, confidence: float, 
                    location_description: str, raw_text: str) -> Dimension:
    """Factory function to create a Dimension with auto-generated ID."""
    return Dimension(
        id=generate_unique_id(),
        value=value,
        unit=unit,
        type=type,
        confidence=confidence,
        location_description=location_description,
        raw_text=raw_text
    )


def create_tolerance(type: str, value: str, confidence: float, raw_text: str,
                    associated_dimension_id: Optional[str] = None) -> Tolerance:
    """Factory function to create a Tolerance with auto-generated ID."""
    return Tolerance(
        id=generate_unique_id(),
        type=type,
        value=value,
        confidence=confidence,
        associated_dimension_id=associated_dimension_id,
        raw_text=raw_text
    )


def create_part_number(identifier: str, confidence: float, raw_text: str,
                      description: Optional[str] = None) -> PartNumber:
    """Factory function to create a PartNumber with auto-generated ID."""
    return PartNumber(
        id=generate_unique_id(),
        identifier=identifier,
        description=description,
        confidence=confidence,
        raw_text=raw_text
    )


def create_annotation(text: str, type: str, confidence: float, 
                     location_description: str) -> Annotation:
    """Factory function to create an Annotation with auto-generated ID."""
    return Annotation(
        id=generate_unique_id(),
        text=text,
        type=type,
        confidence=confidence,
        location_description=location_description

    )
