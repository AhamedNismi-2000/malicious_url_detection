#!/usr/bin/env pwsh

Write-Host "🔧 Setting up Malicious URL Detection environment..."
Write-Host ""

# ----------------------------------------
# 1. Create virtual environment
# ----------------------------------------
Write-Host "📌 Creating virtual environment: venv"
python -m venv venv

# Activate venv (Windows)
Write-Host "📌 Activating virtual environment..."
. .\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "✅ Virtual environment activated."
Write-Host ""

# ----------------------------------------
# 2. Upgrade pip
# ----------------------------------------
Write-Host "⬆️ Upgrading pip..."
pip install --upgrade pip

# ----------------------------------------
# 3. Install required packages
# ----------------------------------------
Write-Host "📦 Installing required dependencies..."

pip install `
    numpy `
    pandas `
    requests `
    tldextract `
    python-whois `
    dnspython `
    ipaddress `
    tqdm `
    scipy `
    python-dateutil `
    urllib3 `
    joblib `
    scikit-learn `
    nltk `
    regex `
    matplotlib `
    validators `
    beautifulsoup4 `
    lxml
    seaborn
Write-Host ""
Write-Host "🎉 Packages installed successfully."
Write-Host ""

# ----------------------------------------
# 4. Download NLTK resources (fixed for PowerShell)
# ----------------------------------------
Write-Host "⬇️ Downloading NLTK data..."

python -c "
import nltk;
nltk.download('punkt');
nltk.download('stopwords');
"

Write-Host "✨ NLTK resources downloaded."
Write-Host ""

Write-Host "🚀 Setup complete!"
Write-Host "Activate environment with:"
Write-Host "  .\venv\Scripts\Activate.ps1"


# to run this                 Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
#                             .\setup.ps1
