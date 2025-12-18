@echo off
REM Strands Barrel Extractor Setup Script for Windows
REM This script automates the installation process

echo ===============================================
echo   Strands Barrel Extractor Setup Script
echo ===============================================
echo.

REM Check if Python is installed
echo [INFO] Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% found

REM Check if pip is installed
echo [INFO] Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found. Please install pip.
    pause
    exit /b 1
)
echo [SUCCESS] pip found

REM Create virtual environment
echo [INFO] Creating virtual environment...
if exist venv (
    echo [WARNING] Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment created

REM Activate virtual environment and install dependencies
echo [INFO] Installing Python dependencies...
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
if exist requirements.txt (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install Python dependencies
        pause
        exit /b 1
    )
    echo [SUCCESS] Python dependencies installed
) else (
    echo [ERROR] requirements.txt not found
    pause
    exit /b 1
)

REM Check AWS CLI
echo [INFO] Checking AWS CLI...
aws --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] AWS CLI not found. Install it for easier credential management.
    echo You can install it from: https://aws.amazon.com/cli/
) else (
    echo [SUCCESS] AWS CLI found
    
    REM Check if credentials are configured
    aws sts get-caller-identity >nul 2>&1
    if %errorlevel% neq 0 (
        echo [WARNING] AWS credentials not configured. Run 'aws configure' to set them up.
    ) else (
        echo [SUCCESS] AWS credentials are configured
    )
)

REM Create configuration file
echo [INFO] Creating configuration file...
if exist config_template.py (
    if not exist config.py (
        copy config_template.py config.py >nul
        echo [SUCCESS] Configuration file created from template
        echo [WARNING] Please edit config.py to set your AWS region and credentials
    ) else (
        echo [WARNING] config.py already exists. Skipping creation.
    )
) else (
    echo [WARNING] config_template.py not found. Skipping config creation.
)

REM Test installation
echo [INFO] Testing installation...
python -c "import streamlit; import boto3; import cv2; import numpy; import pandas; import pdf2image; from PIL import Image; print('All imports successful')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Some packages failed to import
    pause
    exit /b 1
)
echo [SUCCESS] All Python packages imported successfully

echo.
echo ===============================================
echo [SUCCESS] Setup completed successfully!
echo ===============================================
echo.
echo Next steps:
echo 1. Configure AWS credentials:
echo    - Run: aws configure
echo    - Or set environment variables:
echo      set AWS_ACCESS_KEY_ID=your_key_id
echo      set AWS_SECRET_ACCESS_KEY=your_secret_key
echo      set AWS_DEFAULT_REGION=us-west-2
echo    - Or use IAM roles (if on EC2)
echo.
echo 2. Edit config.py to set your AWS region
echo.
echo 3. Install poppler for PDF processing:
echo    - Download from: https://github.com/oschwartz10612/poppler-windows/releases
echo    - Extract and add bin folder to PATH
echo.
echo 4. Request model access in AWS Bedrock console:
echo    - Claude 4.5 Sonnet
echo    - Claude 4.5 Opus
echo    - Claude 3 Sonnet (fallback)
echo    - Claude 3 Haiku (fallback)
echo.
echo 5. Start the application:
echo    venv\Scripts\activate.bat
echo    streamlit run strands_barrel_extractor.py
echo.
echo For detailed instructions, see README.md
echo.
pause
