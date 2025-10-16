"""
Configuration management for the Streamlit Engineering Drawing Extractor.

This module handles environment variables, Streamlit secrets, and application settings.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AWSConfig:
    """AWS service configuration."""
    region: str
    access_key_id: Optional[str]
    secret_access_key: Optional[str]


@dataclass
class ModelConfig:
    """AI model configuration."""
    claude_model_id: str
    max_tokens: int
    temperature: float


@dataclass
class BDAConfig:
    """Bedrock Data Automation configuration."""
    project_name: str
    blueprint_name: str


@dataclass
class AppConfig:
    """Application configuration."""
    max_file_size_mb: int
    confidence_threshold: float
    processing_timeout: int
    default_method: str
    enable_method_selection: bool
    log_level: str


class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self._config_cache = {}
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from various sources."""
        logger.info("Loading application configuration...")
        
        # Load from environment variables first
        self._load_from_env()
        
        # Override with Streamlit secrets if available
        if STREAMLIT_AVAILABLE:
            self._load_from_streamlit_secrets()
        
        logger.info("Configuration loaded successfully")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        self._config_cache.update({
            # AWS Configuration
            'aws_region': os.getenv('AWS_REGION', 'us-east-1'),
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            
            # Application Settings
            'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', '100')),
            'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.7')),
            'processing_timeout': int(os.getenv('PROCESSING_TIMEOUT', '300')),
            'default_method': os.getenv('DEFAULT_METHOD', 'auto'),
            'enable_method_selection': os.getenv('ENABLE_METHOD_SELECTION', 'true').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            
            # Model Configuration
            'claude_model_id': os.getenv('CLAUDE_MODEL_ID', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
            'max_tokens': int(os.getenv('MAX_TOKENS', '4000')),
            'temperature': float(os.getenv('TEMPERATURE', '0.1')),
            
            # BDA Configuration
            'bda_project_name': os.getenv('BDA_PROJECT_NAME', 'engineering-drawing-extractor'),
            'bda_blueprint_name': os.getenv('BDA_BLUEPRINT_NAME', 'engineering-drawing-blueprint'),
        })
    
    def _load_from_streamlit_secrets(self):
        """Load configuration from Streamlit secrets."""
        try:
            if hasattr(st, 'secrets'):
                # AWS secrets
                if 'aws' in st.secrets:
                    aws_secrets = st.secrets['aws']
                    self._config_cache.update({
                        'aws_region': aws_secrets.get('AWS_DEFAULT_REGION', self._config_cache['aws_region']),
                        'aws_access_key_id': aws_secrets.get('AWS_ACCESS_KEY_ID', self._config_cache['aws_access_key_id']),
                        'aws_secret_access_key': aws_secrets.get('AWS_SECRET_ACCESS_KEY', self._config_cache['aws_secret_access_key']),
                    })
                
                # App secrets
                if 'app' in st.secrets:
                    app_secrets = st.secrets['app']
                    self._config_cache.update({
                        'max_file_size_mb': int(app_secrets.get('MAX_FILE_SIZE_MB', self._config_cache['max_file_size_mb'])),
                        'confidence_threshold': float(app_secrets.get('CONFIDENCE_THRESHOLD', self._config_cache['confidence_threshold'])),
                        'default_method': app_secrets.get('DEFAULT_METHOD', self._config_cache['default_method']),
                    })
                
                # Model secrets
                if 'models' in st.secrets:
                    model_secrets = st.secrets['models']
                    self._config_cache.update({
                        'claude_model_id': model_secrets.get('CLAUDE_MODEL_ID', self._config_cache['claude_model_id']),
                        'max_tokens': int(model_secrets.get('MAX_TOKENS', self._config_cache['max_tokens'])),
                        'temperature': float(model_secrets.get('TEMPERATURE', self._config_cache['temperature'])),
                    })
                
                # BDA secrets
                if 'bda' in st.secrets:
                    bda_secrets = st.secrets['bda']
                    self._config_cache.update({
                        'bda_project_name': bda_secrets.get('PROJECT_NAME', self._config_cache['bda_project_name']),
                        'bda_blueprint_name': bda_secrets.get('BLUEPRINT_NAME', self._config_cache['bda_blueprint_name']),
                    })
                
                logger.info("Streamlit secrets loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Streamlit secrets: {str(e)}")
    
    def get_aws_config(self) -> AWSConfig:
        """Get AWS configuration."""
        return AWSConfig(
            region=self._config_cache['aws_region'],
            access_key_id=self._config_cache['aws_access_key_id'],
            secret_access_key=self._config_cache['aws_secret_access_key']
        )
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(
            claude_model_id=self._config_cache['claude_model_id'],
            max_tokens=self._config_cache['max_tokens'],
            temperature=self._config_cache['temperature']
        )
    
    def get_bda_config(self) -> BDAConfig:
        """Get BDA configuration."""
        return BDAConfig(
            project_name=self._config_cache['bda_project_name'],
            blueprint_name=self._config_cache['bda_blueprint_name']
        )
    
    def get_app_config(self) -> AppConfig:
        """Get application configuration."""
        return AppConfig(
            max_file_size_mb=self._config_cache['max_file_size_mb'],
            confidence_threshold=self._config_cache['confidence_threshold'],
            processing_timeout=self._config_cache['processing_timeout'],
            default_method=self._config_cache['default_method'],
            enable_method_selection=self._config_cache['enable_method_selection'],
            log_level=self._config_cache['log_level']
        )
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging."""
        return {
            'aws_region': self._config_cache['aws_region'],
            'aws_credentials_configured': bool(self._config_cache['aws_access_key_id']),
            'max_file_size_mb': self._config_cache['max_file_size_mb'],
            'confidence_threshold': self._config_cache['confidence_threshold'],
            'default_method': self._config_cache['default_method'],
            'claude_model_id': self._config_cache['claude_model_id'],
            'bda_project_name': self._config_cache['bda_project_name'],
            'dotenv_available': DOTENV_AVAILABLE,
            'streamlit_available': STREAMLIT_AVAILABLE
        }
    
    def validate_configuration(self) -> tuple[bool, list[str]]:
        """
        Validate configuration and return any issues.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check AWS configuration
        if not self._config_cache['aws_region']:
            errors.append("AWS region is not configured")
        
        # Check file size limits
        if self._config_cache['max_file_size_mb'] <= 0:
            errors.append("Maximum file size must be greater than 0")
        
        # Check confidence threshold
        if not (0.0 <= self._config_cache['confidence_threshold'] <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        # Check model configuration
        if not self._config_cache['claude_model_id']:
            errors.append("Claude model ID is not configured")
        
        if self._config_cache['max_tokens'] <= 0:
            errors.append("Max tokens must be greater than 0")
        
        if not (0.0 <= self._config_cache['temperature'] <= 2.0):
            errors.append("Temperature must be between 0.0 and 2.0")
        
        # Check method configuration
        valid_methods = ['auto', 'claude_3_5_sonnet', 'textract', 'bedrock_data_automation']
        if self._config_cache['default_method'] not in valid_methods:
            errors.append(f"Default method must be one of: {', '.join(valid_methods)}")
        
        return len(errors) == 0, errors


# Global configuration manager instance
config_manager = ConfigManager()


def get_aws_config() -> AWSConfig:
    """Get AWS configuration."""
    return config_manager.get_aws_config()


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return config_manager.get_model_config()


def get_bda_config() -> BDAConfig:
    """Get BDA configuration."""
    return config_manager.get_bda_config()


def get_app_config() -> AppConfig:
    """Get application configuration."""
    return config_manager.get_app_config()


def validate_config() -> tuple[bool, list[str]]:
    """Validate current configuration."""
    return config_manager.validate_configuration()


def display_config_info():
    """Display configuration information (for debugging)."""
    if STREAMLIT_AVAILABLE:
        import streamlit as st
        
        with st.expander("🔧 Configuration Info"):
            config_summary = config_manager.get_config_summary()
            
            st.write("**Configuration Summary:**")
            for key, value in config_summary.items():
                if key == 'aws_credentials_configured':
                    status = "✅ Configured" if value else "❌ Not configured"
                    st.write(f"• **AWS Credentials**: {status}")
                else:
                    st.write(f"• **{key.replace('_', ' ').title()}**: {value}")
            
            # Validation
            is_valid, errors = validate_config()
            if is_valid:
                st.success("✅ Configuration is valid")
            else:
                st.error("❌ Configuration issues detected:")
                for error in errors:
                    st.write(f"• {error}")