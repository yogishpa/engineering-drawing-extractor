# Configuration Template for Strands Barrel Extractor
# Copy this file to config.py and update the values

# PLACEHOLDER: AWS Configuration
AWS_CONFIG = {
    # Replace with your AWS region
    'region': 'YOUR_AWS_REGION',  # e.g., 'us-west-2', 'us-east-1', 'eu-west-1'
    
    # AWS Credentials (choose one method)
    'credentials_method': 'aws_cli',  # Options: 'aws_cli', 'environment', 'iam_role'
    
    # If using environment variables, set these:
    'access_key_id': 'YOUR_ACCESS_KEY_ID',
    'secret_access_key': 'YOUR_SECRET_ACCESS_KEY',
}

# PLACEHOLDER: Model Configuration
MODEL_CONFIG = {
    # Primary models (Claude 4.5)
    'primary_sonnet': 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
    'primary_opus': 'global.anthropic.claude-opus-4-5-20251101-v1:0',
    
    # Fallback models (Claude 3)
    'fallback_sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
    'fallback_haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
    
    # Model parameters
    'max_tokens': 2000,
    'temperature': 0.1,
}

# Application Configuration
APP_CONFIG = {
    'log_level': 'INFO',  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'max_file_size_mb': 50,  # Maximum PDF file size
    'image_dpi': 300,  # PDF to image conversion DPI
    'confidence_threshold': 0.7,  # Minimum confidence for results
}

# PLACEHOLDER: Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'Strands Swarm Barrel Extractor',
    'layout': 'wide',
    'port': 8501,
    'host': 'localhost',
}

# Agent Configuration
AGENT_CONFIG = {
    'timeout_seconds': 120,  # Timeout for each agent
    'retry_attempts': 3,  # Number of retry attempts
    'enable_fallback': True,  # Enable fallback models
}

# PLACEHOLDER: Custom Prompts (Optional)
CUSTOM_PROMPTS = {
    'enable_custom': False,  # Set to True to use custom prompts
    'extraction_prompt': """
    # Your custom extraction prompt here
    # This will override the default prompt
    """,
    'evaluation_prompt': """
    # Your custom evaluation prompt here
    # This will override the default evaluation prompt
    """
}
