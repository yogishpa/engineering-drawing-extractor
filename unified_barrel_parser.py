import re
import logging
from datetime import datetime

# Configure file logging only
log_filename = f"barrel_parser_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

def parse_barrel_from_text(raw_text, method_name="unknown"):
    """Parse barrel measurements from any text source - EXACT EXTRACTION ONLY."""
    try:
        logger.info(f"Parsing barrel measurements using {method_name} - text length: {len(raw_text)}")
        logger.info(f"Raw text sample: {raw_text[:500]}...")  # Log first 500 chars for debugging
        
        # Initialize result structure
        result = {
            'dimensions': [],
            'tolerances': [],
            'part_numbers': [],
            'annotations': [],
            'overall_confidence': 0.8
        }
        
        # Extract part number from text - ONLY exact matches
        part_patterns = [
            r'(\d{10})',  # 10-digit number like 1413124003 or 9401081110
        ]
        
        for pattern in part_patterns:
            part_matches = re.findall(pattern, raw_text)
            for match in part_matches:
                part_num = str(match)
                if len(part_num) == 10 and part_num.isdigit():
                    result['part_numbers'].append({
                        'identifier': part_num,
                        'description': 'PUMP BARREL',
                        'confidence': 0.9,
                        'raw_text': part_num
                    })
                    logger.info(f"Found part number: {part_num}")
                    break
            if result['part_numbers']:
                break
        
        # DO NOT extract dimensions unless they are clearly labeled
        # Only return what we can definitively identify
        logger.info(f"{method_name} extracted {len(result['dimensions'])} dimensions, {len(result['part_numbers'])} part numbers")
        return result
        
    except Exception as e:
        logger.error(f"Failed to parse barrel measurements with {method_name}: {str(e)}")
        return {
            'dimensions': [],
            'tolerances': [],
            'part_numbers': [],
            'annotations': [],
            'overall_confidence': 0.5
        }
