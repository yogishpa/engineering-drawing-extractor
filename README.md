# Streamlit Engineering Drawing Extractor

A powerful web application for extracting dimensions, tolerances, part numbers, and annotations from engineering drawings using AWS AI services.

## Features

- 📄 **PDF Upload**: Support for engineering drawings up to 100MB
- 🤖 **Multiple AI Methods**: Claude 4 Sonnet, Bedrock Data Automation, Amazon Textract
- 📊 **Structured Results**: Organized extraction of dimensions, tolerances, part numbers, and annotations
- 🎯 **Confidence Scoring**: Quality assessment for all extracted data
- 📥 **Export Options**: Download results in JSON, CSV, or summary report formats
- ⚡ **Real-time Processing**: Live status updates and progress indicators
- 🛡️ **Error Handling**: Comprehensive error management with user-friendly messages
- 🔧 **Configurable**: Flexible configuration via environment variables or Streamlit secrets

## Quick Start

### Prerequisites

- Python 3.9 or higher
- AWS account with appropriate permissions
- AWS CLI configured (optional but recommended)

### Installation

1. **Clone and setup the application:**
   ```bash
   git clone <repository-url>
   cd streamlit-drawing-extractor
   ./deploy.sh setup
   ```

2. **Configure AWS credentials:**
   Edit the `.env` file with your AWS credentials:
   ```bash
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   ```

3. **Start the application:**
   ```bash
   ./deploy.sh local
   ```

4. **Open your browser:**
   Navigate to `http://localhost:8501`

## Configuration

### Environment Variables

The application can be configured using environment variables or a `.env` file:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Application Settings
MAX_FILE_SIZE_MB=100
CONFIDENCE_THRESHOLD=0.7
DEFAULT_METHOD=auto

# Model Configuration
CLAUDE_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
MAX_TOKENS=4000
TEMPERATURE=0.1
```

### Streamlit Secrets

For Streamlit Cloud deployment, use `.streamlit/secrets.toml`:

```toml
[aws]
AWS_ACCESS_KEY_ID = "your-access-key"
AWS_SECRET_ACCESS_KEY = "your-secret-key"
AWS_DEFAULT_REGION = "us-east-1"

[app]
MAX_FILE_SIZE_MB = 100
CONFIDENCE_THRESHOLD = 0.7
DEFAULT_METHOD = "auto"
```

## Deployment Options

### Local Development

```bash
# Setup and start local server
./deploy.sh setup
./deploy.sh local
```

### Docker Deployment

```bash
# Build and start with Docker
./deploy.sh docker

# Stop Docker container
./deploy.sh stop
```

### Streamlit Cloud

1. Fork this repository
2. Connect to Streamlit Cloud
3. Configure secrets in the Streamlit Cloud dashboard
4. Deploy automatically

### AWS EC2

```bash
# On EC2 instance
sudo yum update -y
sudo yum install -y python3 python3-pip git docker
sudo systemctl start docker
sudo usermod -a -G docker ec2-user

# Clone and deploy
git clone <repository-url>
cd streamlit-drawing-extractor
./deploy.sh setup
./deploy.sh docker
```

## Usage

### Supported File Types

- **PDF files** up to 100MB
- Engineering drawings with clear text and dimensions
- Technical documents with structured content

### Extraction Methods

1. **Auto (Recommended)**: Automatically selects the best available method
2. **Claude 3.5 Sonnet**: Advanced AI vision analysis for complex drawings
3. **Bedrock Data Automation**: Structured extraction with engineering-specific blueprints
4. **Amazon Textract**: Reliable text extraction with basic parsing

### Workflow

1. **Upload**: Select and upload your PDF engineering drawing
2. **Configure**: Choose extraction method and confidence settings
3. **Process**: Click "Extract Data" to begin processing
4. **Review**: Examine results with confidence indicators
5. **Export**: Download results in your preferred format

## API Reference

### Core Components

- **ExtractionEngine**: Main processing engine
- **AWSServiceManager**: AWS service coordination
- **ErrorHandler**: Comprehensive error management
- **ConfigManager**: Configuration management

### Data Models

- **ExtractionResult**: Complete extraction results
- **Dimension**: Dimensional measurements
- **Tolerance**: Tolerance specifications
- **PartNumber**: Part identifiers
- **Annotation**: Text annotations

## Development

### Project Structure

```
streamlit-drawing-extractor/
├── app.py                 # Main Streamlit application
├── extractor.py          # Core extraction engine
├── aws_clients.py        # AWS service clients
├── models.py             # Data models
├── error_handler.py      # Error handling system
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-container setup
├── deploy.sh           # Deployment script
├── .env.example        # Environment template
├── .streamlit/         # Streamlit configuration
│   ├── config.toml
│   └── secrets.toml.example
└── tests/              # Test files
    ├── test_*.py
    └── ...
```

### Running Tests

```bash
# Run all tests
./deploy.sh test

# Run specific test file
pytest test_extractor.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Implement changes**: Add code and tests
3. **Run tests**: `./deploy.sh test`
4. **Update documentation**: Update README and docstrings
5. **Submit PR**: Create pull request for review

## Troubleshooting

### Common Issues

**AWS Credentials Not Found**
```bash
# Check AWS configuration
aws configure list

# Set environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
```

**File Upload Fails**
- Check file size (max 100MB)
- Ensure file is a valid PDF
- Verify file is not encrypted

**Low Extraction Quality**
- Use high-resolution PDFs (300 DPI+)
- Ensure text and dimensions are clearly visible
- Try different extraction methods
- Check document contains technical drawings

**Service Unavailable**
- Verify AWS credentials and permissions
- Check AWS service availability in your region
- Ensure internet connectivity
- Try different AWS region

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
./deploy.sh local
```

### Performance Optimization

- Use smaller PDF files when possible
- Enable confidence filtering for faster processing
- Choose appropriate extraction method for your content
- Consider using Docker for consistent performance

## Security

### Best Practices

- Never commit AWS credentials to version control
- Use IAM roles with minimal required permissions
- Enable AWS CloudTrail for audit logging
- Regularly rotate access keys
- Use HTTPS in production deployments

### Required AWS Permissions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock-runtime:InvokeModel",
                "textract:DetectDocumentText",
                "bedrock-data-automation:*"
            ],
            "Resource": "*"
        }
    ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions for questions and ideas

## Changelog

### v1.0.0
- Initial release
- Support for Claude 3.5 Sonnet, Textract, and Bedrock Data Automation
- Comprehensive error handling and user feedback
- Docker deployment support
- Configuration management system
- Export functionality (JSON, CSV, reports)

---


**Built with ❤️ using Streamlit and AWS AI Services**# engineering-drawing-extractor
