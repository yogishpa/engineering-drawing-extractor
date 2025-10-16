"""
Error handling and user feedback system for the Streamlit Engineering Drawing Extractor. 

This module provides comprehensive error handling, user-friendly error messages,
and retry mechanisms for the extraction process.
"""

import logging
import time
import functools
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
import streamlit as st

from aws_clients import AWSClientError
from extractor import PDFValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification."""
    FILE_VALIDATION = "file_validation"
    AWS_SERVICE = "aws_service"
    NETWORK = "network"
    PROCESSING = "processing"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorHandler:
    """Centralized error handling and user feedback system."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_history = []
        self.retry_counts = {}
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds
    
    def classify_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify error by category and severity.
        
        Args:
            error: Exception to classify
            
        Returns:
            Tuple of (category, severity)
        """
        if isinstance(error, PDFValidationError):
            return ErrorCategory.FILE_VALIDATION, ErrorSeverity.ERROR
        
        elif isinstance(error, AWSClientError):
            error_msg = str(error).lower()
            
            # Check for specific AWS error patterns
            if any(term in error_msg for term in ['credentials', 'access denied', 'unauthorized']):
                return ErrorCategory.CONFIGURATION, ErrorSeverity.ERROR
            elif any(term in error_msg for term in ['timeout', 'connection', 'network']):
                return ErrorCategory.NETWORK, ErrorSeverity.WARNING
            elif any(term in error_msg for term in ['throttling', 'rate limit', 'quota']):
                return ErrorCategory.AWS_SERVICE, ErrorSeverity.WARNING
            else:
                return ErrorCategory.AWS_SERVICE, ErrorSeverity.ERROR
        
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK, ErrorSeverity.WARNING
        
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.PROCESSING, ErrorSeverity.ERROR
        
        else:
            return ErrorCategory.UNKNOWN, ErrorSeverity.ERROR
    
    def get_error_suggestions(self, error: Exception, category: ErrorCategory) -> List[str]:
        """
        Get user-friendly suggestions based on error type.
        
        Args:
            error: Exception that occurred
            category: Error category
            
        Returns:
            List of suggestion strings
        """
        error_msg = str(error).lower()
        
        if category == ErrorCategory.FILE_VALIDATION:
            if 'size' in error_msg:
                return [
                    "Try compressing the PDF file using online tools",
                    "Split large drawings into multiple smaller files",
                    "Remove unnecessary pages or high-resolution images",
                    "Use PDF optimization tools to reduce file size"
                ]
            elif 'encrypted' in error_msg or 'password' in error_msg:
                return [
                    "Remove password protection from the PDF",
                    "Use 'Print to PDF' to create an unprotected copy",
                    "Contact the document owner for an unprotected version"
                ]
            elif 'corrupted' in error_msg or 'invalid' in error_msg:
                return [
                    "Try opening the file in a PDF viewer to verify it's valid",
                    "Re-download or re-create the PDF file",
                    "Convert the file to PDF using a different tool",
                    "Check if the file was completely uploaded"
                ]
            else:
                return [
                    "Ensure the file is a valid PDF document",
                    "Try uploading the file again",
                    "Check the file extension is .pdf"
                ]
        
        elif category == ErrorCategory.AWS_SERVICE:
            if 'credentials' in error_msg or 'access' in error_msg:
                return [
                    "Check your AWS credentials configuration",
                    "Verify your AWS account has the necessary permissions",
                    "Ensure AWS CLI is properly configured",
                    "Contact your AWS administrator for access"
                ]
            elif 'region' in error_msg:
                return [
                    "Try selecting a different AWS region",
                    "Verify the service is available in your selected region",
                    "Check AWS service health dashboard"
                ]
            elif 'quota' in error_msg or 'limit' in error_msg:
                return [
                    "Wait a few minutes before trying again",
                    "Try using a different extraction method",
                    "Contact AWS support to increase service limits",
                    "Reduce the frequency of requests"
                ]
            else:
                return [
                    "Try a different extraction method",
                    "Wait a few minutes and try again",
                    "Check AWS service status",
                    "Contact support if the issue persists"
                ]
        
        elif category == ErrorCategory.NETWORK:
            return [
                "Check your internet connection",
                "Try again in a few minutes",
                "Verify firewall settings allow AWS connections",
                "Contact your network administrator if issues persist"
            ]
        
        elif category == ErrorCategory.PROCESSING:
            return [
                "Try uploading the file again",
                "Use a different extraction method",
                "Check if the PDF contains extractable content",
                "Verify the document is a technical drawing"
            ]
        
        elif category == ErrorCategory.CONFIGURATION:
            return [
                "Check your AWS configuration settings",
                "Verify environment variables are set correctly",
                "Ensure all required services are enabled",
                "Contact your system administrator"
            ]
        
        else:
            return [
                "Try the operation again",
                "Check your internet connection",
                "Contact support if the issue continues",
                "Try using a different browser or device"
            ]
    
    def display_error(self, error: Exception, context: str = ""):
        """
        Display user-friendly error message in Streamlit.
        
        Args:
            error: Exception to display
            context: Additional context about when the error occurred
        """
        category, severity = self.classify_error(error)
        suggestions = self.get_error_suggestions(error, category)
        
        # Log the error
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        
        # Record error in history
        self.error_history.append({
            'timestamp': time.time(),
            'error': str(error),
            'category': category.value,
            'severity': severity.value,
            'context': context
        })
        
        # Display appropriate Streamlit message
        error_title = f"{category.value.replace('_', ' ').title()} Error"
        
        if severity == ErrorSeverity.CRITICAL:
            st.error(f"🚨 **{error_title}**: {str(error)}")
        elif severity == ErrorSeverity.ERROR:
            st.error(f"❌ **{error_title}**: {str(error)}")
        elif severity == ErrorSeverity.WARNING:
            st.warning(f"⚠️ **{error_title}**: {str(error)}")
        else:
            st.info(f"ℹ️ **{error_title}**: {str(error)}")
        
        # Display suggestions
        if suggestions:
            st.info("**💡 Suggestions:**")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
    
    def with_retry(self, max_retries: int = None, delay: float = None):
        """
        Decorator for adding retry logic to functions.
        
        Args:
            max_retries: Maximum number of retry attempts
            delay: Delay between retries in seconds
        """
        if max_retries is None:
            max_retries = self.max_retries
        if delay is None:
            delay = self.retry_delay
        
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    
                    except Exception as e:
                        category, severity = self.classify_error(e)
                        
                        # Don't retry certain types of errors
                        if category in [ErrorCategory.FILE_VALIDATION, ErrorCategory.CONFIGURATION]:
                            raise
                        
                        if attempt < max_retries:
                            logger.warning(f"Attempt {attempt + 1} failed for {func_name}: {str(e)}")
                            
                            # Show retry message to user
                            st.warning(f"⏳ Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                            time.sleep(delay)
                            
                            # Exponential backoff
                            delay *= 1.5
                        else:
                            # Final attempt failed
                            logger.error(f"All {max_retries + 1} attempts failed for {func_name}")
                            raise
                
            return wrapper
        return decorator
    
    def handle_timeout(self, timeout_seconds: int = 300):
        """
        Decorator for adding timeout handling to functions.
        
        Args:
            timeout_seconds: Timeout in seconds
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
                
                # Set up timeout
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel timeout
                    return result
                except TimeoutError as e:
                    st.error(f"⏰ **Timeout Error**: {str(e)}")
                    st.info("**💡 Suggestions:**")
                    st.write("• Try with a smaller PDF file")
                    st.write("• Use a different extraction method")
                    st.write("• Check your internet connection")
                    st.write("• Try again during off-peak hours")
                    raise
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
                
            return wrapper
        return decorator
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about errors that have occurred.
        
        Returns:
            Dictionary with error statistics
        """
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        
        for error_record in self.error_history:
            category = error_record['category']
            severity = error_record['severity']
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_category": category_counts,
            "by_severity": severity_counts,
            "recent_errors": self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
        }
    
    def clear_error_history(self):
        """Clear the error history."""
        self.error_history.clear()
        self.retry_counts.clear()


# Global error handler instance
error_handler = ErrorHandler()


def handle_extraction_errors(func: Callable):
    """
    Decorator for handling extraction-related errors.
    
    Args:
        func: Function to wrap with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler.display_error(e, func.__name__)
            raise
    
    return wrapper


def validate_extraction_prerequisites() -> Tuple[bool, List[str]]:
    """
    Validate that all prerequisites for extraction are met.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check if running in Streamlit
    try:
        import streamlit as st
        if not hasattr(st, 'session_state'):
            errors.append("Application must be run in Streamlit environment")
    except ImportError:
        errors.append("Streamlit is not installed")
    
    # Check AWS dependencies
    try:
        import boto3
    except ImportError:
        errors.append("boto3 is not installed - AWS services will not be available")
    
    # Check PDF processing dependencies
    try:
        import PyPDF2
        import fitz
    except ImportError:
        errors.append("PDF processing libraries are not installed")
    
    return len(errors) == 0, errors


def highlight_low_confidence_results(extraction_result, confidence_threshold: float = 0.7):
    """
    Highlight extraction results with low confidence scores.
    
    Args:
        extraction_result: ExtractionResult object
        confidence_threshold: Threshold below which to highlight items
    """
    low_confidence_items = []
    
    # Check dimensions
    for dim in extraction_result.dimensions:
        if dim.confidence < confidence_threshold:
            low_confidence_items.append(f"Dimension: {dim.raw_text} (confidence: {dim.confidence:.1%})")
    
    # Check tolerances
    for tol in extraction_result.tolerances:
        if tol.confidence < confidence_threshold:
            low_confidence_items.append(f"Tolerance: {tol.raw_text} (confidence: {tol.confidence:.1%})")
    
    # Check part numbers
    for part in extraction_result.part_numbers:
        if part.confidence < confidence_threshold:
            low_confidence_items.append(f"Part Number: {part.raw_text} (confidence: {part.confidence:.1%})")
    
    # Check annotations
    for ann in extraction_result.annotations:
        if ann.confidence < confidence_threshold:
            low_confidence_items.append(f"Annotation: {ann.text} (confidence: {ann.confidence:.1%})")
    
    if low_confidence_items:
        st.warning("⚠️ **Low Confidence Results Detected**")
        st.info("The following items may need manual verification:")
        for item in low_confidence_items:
            st.write(f"• {item}")
        
        st.info("**💡 Tips to improve confidence:**")
        st.write("• Ensure the PDF has high image quality")
        st.write("• Try a different extraction method")
        st.write("• Verify the document contains clear technical drawings")
        st.write("• Check that text and dimensions are clearly visible")


def display_processing_feedback(status: str, progress: float = 0.0, message: str = ""):
    """
    Display processing feedback with progress indicators.
    
    Args:
        status: Processing status ('starting', 'processing', 'complete', 'error')
        progress: Progress value between 0.0 and 1.0
        message: Additional status message
    """
    if status == 'starting':
        st.info("🚀 **Starting extraction process...**")
        if message:
            st.write(f"📋 {message}")
        progress_bar = st.progress(0)
        return progress_bar
    
    elif status == 'processing':
        if message:
            st.info(f"⚙️ **Processing**: {message}")
        if 'progress_bar' in st.session_state:
            st.session_state.progress_bar.progress(progress)
    
    elif status == 'complete':
        st.success("✅ **Extraction completed successfully!**")
        if message:
            st.write(f"📊 {message}")
        if 'progress_bar' in st.session_state:
            st.session_state.progress_bar.progress(1.0)
    
    elif status == 'error':
        st.error("❌ **Extraction failed**")
        if message:
            st.write(f"💥 {message}")


def validate_file_upload(uploaded_file, max_size_mb: int = 100) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file before processing.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file extension
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, f"Invalid file type. Expected PDF, got {uploaded_file.name.split('.')[-1].upper()}"
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
    
    # Check if file is empty
    if uploaded_file.size == 0:
        return False, "File appears to be empty"
    
    return True, None


def display_extraction_tips():
    """Display tips for better extraction results."""
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        **For optimal extraction results:**
        
        📄 **Document Quality:**
        • Use high-resolution PDF files (300 DPI or higher)
        • Ensure text and dimensions are clearly visible
        • Avoid heavily compressed or low-quality scans
        
        🎯 **Drawing Content:**
        • Include technical drawings with clear dimensions
        • Ensure tolerance specifications are legible
        • Make sure part numbers and annotations are visible
        
        ⚙️ **Extraction Methods:**
        • **Auto**: Best for most cases - tries multiple methods
        • **Claude 4 Sonnet**: Best for complex drawings with mixed content
        • **Bedrock Data Automation**: Optimized for structured engineering data
        • **Textract**: Good fallback for basic text extraction
        
        🔧 **Troubleshooting:**
        • If results are poor, try a different extraction method
        • For large files, consider splitting into smaller sections
        • Ensure your AWS credentials are properly configured
        """)


def display_system_health():
    """Display system health information in Streamlit sidebar."""
    with st.sidebar:
        st.subheader("🏥 System Health")
        
        # Check prerequisites
        is_healthy, health_errors = validate_extraction_prerequisites()
        
        if is_healthy:
            st.success("✅ All systems operational")
        else:
            st.error("❌ System issues detected")
            for error in health_errors:
                st.write(f"• {error}")
        
        # Error statistics
        stats = error_handler.get_error_statistics()
        if stats["total_errors"] > 0:
            st.warning(f"⚠️ {stats['total_errors']} errors recorded")
            
            if st.button("Clear Error History"):
                error_handler.clear_error_history()
                st.success("Error history cleared")
                st.experimental_rerun()
        else:
            st.info("📊 No errors recorded")
        
        # Display extraction tips

        display_extraction_tips()
