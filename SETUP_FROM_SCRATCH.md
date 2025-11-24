# 🚀 Complete Setup Guide - From Scratch

This guide will help you set up the Topic Classification App on a **brand new computer** with nothing installed.

---

## 📋 Prerequisites Checklist

Before starting, you'll need:
- ✅ Internet connection
- ✅ Administrator/sudo access
- ✅ ~2GB free disk space
- ✅ Azure OpenAI API credentials

---

## 🖥️ Setup Instructions by Operating System

# macOS Setup (Fresh Computer)

## Step 1: Install Homebrew (Package Manager)

1. Open **Terminal** (Applications → Utilities → Terminal)
2. Install Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Follow the on-screen instructions
4. Close and reopen Terminal

## Step 2: Install Python

```bash
# Install Python 3.11 (recommended)
brew install python@3.11

# Verify installation
python3 --version
# Should show: Python 3.11.x
```

## Step 3: Install Git

```bash
# Install Git
brew install git

# Verify installation
git --version
# Should show: git version 2.x.x
```

## Step 4: Clone the Repository

```bash
# Navigate to your desired location
cd ~/Desktop

# Clone the repository
git clone https://github.com/gavrielhan/open-questions.git

# Enter the project directory
cd open-questions
```

## Step 5: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

## Step 6: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# This will take 2-3 minutes
```

## Step 7: Configure Environment

```bash
# Create .env file
nano .env
```

Paste this configuration (replace with your actual credentials):

```bash
# Azure OpenAI Configuration
API_KEY=your_azure_api_key_here
API_BASE_URL=https://sni-ai-foundry.cognitiveservices.azure.com
MODEL=gpt-5.1-gavriel
AZURE_API_VERSION=2025-04-01-preview

# DeepSeek for JSON repair (uses same API key by default)
REPAIR_MODEL=DeepSeek-V3.1-gavriel
REPAIR_ENDPOINT=https://sni-ai-foundry.services.ai.azure.com/openai/v1/
REPAIR_API_VERSION=2025-04-01-preview
```

Press `Ctrl+O` to save, `Enter` to confirm, `Ctrl+X` to exit.

## Step 8: Test the Setup

```bash
# Test CLI classification (on 5 rows)
python3 classify_topics.py --limit 5

# If successful, you should see:
# ✔ Batch 1/1 processed by model
# ✅ Done. Saved to [filename]_classified.csv
```

## Step 9: Create Desktop App

```bash
# Make script executable
chmod +x create_desktop_app.sh

# Run the script
./create_desktop_app.sh

# The app will appear on your Desktop
```

## Step 10: Launch the App

1. Go to your **Desktop**
2. Double-click **"Topic Classifier.app"**
3. If you see a security warning:
   - Go to **System Preferences → Security & Privacy**
   - Click **"Open Anyway"**
4. Browser opens automatically at `http://127.0.0.1:5000`

---

# Windows Setup (Fresh Computer)

## Step 1: Install Python

1. Go to https://www.python.org/downloads/
2. Download **Python 3.11.x** (recommended)
3. Run the installer
4. **⚠️ IMPORTANT:** Check ✅ "Add Python to PATH"
5. Click "Install Now"
6. Wait for installation to complete

## Step 2: Verify Python Installation

1. Open **Command Prompt** (Win+R, type `cmd`, press Enter)
2. Check Python version:
```cmd
python --version
```
Should show: `Python 3.11.x`

## Step 3: Install Git

1. Go to https://git-scm.com/download/win
2. Download Git for Windows
3. Run the installer
4. Use default settings (click "Next" through all options)
5. Finish installation

## Step 4: Clone the Repository

1. Open **Command Prompt**
2. Navigate to your desired location:
```cmd
cd %USERPROFILE%\Desktop
```
3. Clone the repository:
```cmd
git clone https://github.com/gavrielhan/open-questions.git
cd open-questions
```

## Step 5: Create Virtual Environment

```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your prompt
```

## Step 6: Install Dependencies

```cmd
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# This will take 2-3 minutes
```

## Step 7: Configure Environment

1. Create a file named `.env` (use Notepad)
```cmd
notepad .env
```

2. Paste this configuration (replace with your actual credentials):

```
API_KEY=your_azure_api_key_here
API_BASE_URL=https://sni-ai-foundry.cognitiveservices.azure.com
MODEL=gpt-5.1-gavriel
AZURE_API_VERSION=2025-04-01-preview

REPAIR_MODEL=DeepSeek-V3.1-gavriel
REPAIR_ENDPOINT=https://sni-ai-foundry.services.ai.azure.com/openai/v1/
REPAIR_API_VERSION=2025-04-01-preview
```

3. Save and close (File → Save, then close Notepad)

## Step 8: Test the Setup

```cmd
# Test CLI classification
python classify_topics.py --limit 5

# If successful, you should see:
# ✔ Batch 1/1 processed by model
# ✅ Done. Saved to [filename]_classified.csv
```

## Step 9: Create Desktop Shortcut

```cmd
# Run PowerShell script
powershell -ExecutionPolicy Bypass -File create_windows_shortcut.ps1
```

## Step 10: Launch the App

1. Go to your **Desktop**
2. Double-click **"Topic Classifier"** shortcut
3. Command window opens and starts the server
4. Browser opens automatically at `http://127.0.0.1:5000`

---

## 🔧 Troubleshooting

### Issue: "python: command not found" (Mac)

**Solution:**
```bash
# Install Python via Homebrew
brew install python@3.11

# Or download from python.org
```

### Issue: "python is not recognized" (Windows)

**Solution:**
1. Reinstall Python
2. **Check** ✅ "Add Python to PATH" during installation
3. Restart Command Prompt

### Issue: "pip install fails with SSL error"

**Solution:**
```bash
# Mac
brew install openssl
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Windows
python -m pip install --upgrade pip
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue: "Port 5000 is in use"

**Solution (Mac):**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

**Solution (Windows):**
```cmd
# Find and kill process
netstat -ano | findstr :5000
taskkill /PID [PID_NUMBER] /F
```

### Issue: "Module not found" errors

**Solution:**
```bash
# Make sure virtual environment is activated
# You should see (venv) in your prompt

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Azure API errors (404, 401)

**Solution:**
1. Verify `API_KEY` is correct
2. Check `MODEL` matches your deployment name exactly
3. Verify `API_BASE_URL` is correct
4. Test in Azure portal first

---

## 📦 What Gets Installed

| Package | Purpose | Size |
|---------|---------|------|
| Python 3.11+ | Programming language | ~100MB |
| pandas | Excel/CSV processing | ~50MB |
| flask | Web server | ~10MB |
| langchain | LLM orchestration | ~30MB |
| requests | HTTP requests | ~5MB |
| openpyxl | Excel file handling | ~5MB |
| Total | | ~200MB |

---

## 🔐 Security Notes

- ✅ `.env` file is gitignored (API keys stay private)
- ✅ All processing happens locally
- ✅ No data sent except to your Azure endpoint
- ✅ Virtual environment isolates dependencies

---

## 📚 Next Steps After Setup

1. **Test with sample data:**
   ```bash
   python3 classify_topics.py --limit 10
   ```

2. **Use the web app:**
   - Upload your Excel/CSV file
   - Select columns
   - Run classification
   - Download results

3. **Check the documentation:**
   - `README.md` - Main documentation
   - `README_WEB_APP.md` - Web app guide
   - `README_ENHANCED.md` - Enhanced features

---

## 📞 Getting Help

If you encounter issues:

1. **Check logs:** Look at terminal/command prompt output
2. **Verify credentials:** Double-check `.env` file
3. **Test connection:** Try a simple API call in Azure portal
4. **Check versions:** Ensure Python 3.7+

---

## ⏱️ Estimated Setup Time

| Platform | Time |
|----------|------|
| **Mac** (no Python) | 15-20 minutes |
| **Mac** (Python installed) | 5-10 minutes |
| **Windows** (no Python) | 20-25 minutes |
| **Windows** (Python installed) | 5-10 minutes |

---

## ✅ Setup Verification Checklist

- [ ] Python 3.7+ installed and in PATH
- [ ] Git installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] `.env` file created with correct credentials
- [ ] Test classification runs successfully
- [ ] Desktop app/shortcut created
- [ ] Web app opens in browser

**All checked?** You're ready to go! 🎉

