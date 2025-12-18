#!/bin/bash

# Strands Barrel Extractor Setup Script
# This script automates the installation process

set -e  # Exit on any error

echo "🚀 Starting Strands Barrel Extractor Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.9+ is installed
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.9"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)"; then
            print_success "Python $PYTHON_VERSION found (>= $REQUIRED_VERSION required)"
        else
            print_error "Python $PYTHON_VERSION found, but >= $REQUIRED_VERSION required"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9 or higher."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip..."
    
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 not found. Please install pip."
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            print_status "Detected Ubuntu/Debian, installing poppler-utils..."
            sudo apt-get update
            sudo apt-get install -y poppler-utils
        elif command -v yum &> /dev/null; then
            print_status "Detected RHEL/CentOS, installing poppler-utils..."
            sudo yum install -y poppler-utils
        else
            print_warning "Unknown Linux distribution. Please install poppler-utils manually."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            print_status "Detected macOS, installing poppler via Homebrew..."
            brew install poppler
        else
            print_warning "Homebrew not found. Please install poppler manually or install Homebrew first."
        fi
    else
        print_warning "Unknown OS. Please install poppler manually."
    fi
    
    print_success "System dependencies installation completed"
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment and install Python packages
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Check AWS CLI
check_aws_cli() {
    print_status "Checking AWS CLI..."
    
    if command -v aws &> /dev/null; then
        AWS_VERSION=$(aws --version 2>&1 | cut -d/ -f2 | cut -d' ' -f1)
        print_success "AWS CLI $AWS_VERSION found"
        
        # Check if credentials are configured
        if aws sts get-caller-identity &> /dev/null; then
            print_success "AWS credentials are configured"
        else
            print_warning "AWS credentials not configured. Run 'aws configure' to set them up."
        fi
    else
        print_warning "AWS CLI not found. Install it for easier credential management."
        print_status "You can install it with: pip install awscli"
    fi
}

# Create configuration file
create_config() {
    print_status "Creating configuration file..."
    
    if [ -f "config_template.py" ]; then
        if [ ! -f "config.py" ]; then
            cp config_template.py config.py
            print_success "Configuration file created from template"
            print_warning "Please edit config.py to set your AWS region and credentials"
        else
            print_warning "config.py already exists. Skipping creation."
        fi
    else
        print_warning "config_template.py not found. Skipping config creation."
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test imports
    python3 -c "
import streamlit
import boto3
import cv2
import numpy
import pandas
import pdf2image
from PIL import Image
print('✅ All imports successful')
" 2>/dev/null && print_success "All Python packages imported successfully" || {
        print_error "Some packages failed to import"
        exit 1
    }
}

# Main setup function
main() {
    echo "=============================================="
    echo "  Strands Barrel Extractor Setup Script"
    echo "=============================================="
    echo ""
    
    # Run setup steps
    check_python
    check_pip
    install_system_deps
    create_venv
    install_python_deps
    check_aws_cli
    create_config
    test_installation
    
    echo ""
    echo "=============================================="
    print_success "Setup completed successfully!"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "1. Configure AWS credentials:"
    echo "   - Run: aws configure"
    echo "   - Or set environment variables"
    echo "   - Or use IAM roles (if on EC2)"
    echo ""
    echo "2. Edit config.py to set your AWS region"
    echo ""
    echo "3. Request model access in AWS Bedrock console:"
    echo "   - Claude 4.5 Sonnet"
    echo "   - Claude 4.5 Opus"
    echo "   - Claude 3 Sonnet (fallback)"
    echo "   - Claude 3 Haiku (fallback)"
    echo ""
    echo "4. Start the application:"
    echo "   source venv/bin/activate"
    echo "   streamlit run strands_barrel_extractor.py"
    echo ""
    echo "For detailed instructions, see README.md"
}

# Run main function
main
