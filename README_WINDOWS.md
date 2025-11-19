# Windows Setup Guide - Topic Classification Tool

This guide will help you set up the Topic Classification Web App on Windows.

## 🚀 Quick Start

### Option 1: Create Desktop Shortcut (Recommended)

1. **Double-click** `create_windows_shortcut.bat` in the project folder
2. A shortcut named "Topic Classifier" will appear on your Desktop
3. **Double-click** the shortcut to launch the app
4. Your browser will open automatically with the upload interface

### Option 2: Run Directly

1. **Double-click** `launch_app.bat` in the project folder
2. A command window will open and start the web server
3. Your browser will open automatically

## 📋 Prerequisites

- **Python 3.7 or higher** installed
- All dependencies installed (`pip install -r requirements.txt`)
- Valid `.env` file with API configuration

### Installing Python Dependencies

Open Command Prompt or PowerShell in the project folder and run:

```cmd
pip install -r requirements.txt
```

## 🔧 Configuration

Make sure your `.env` file in the project folder contains:

```
API_KEY=your_api_key_here
API_BASE_URL=your_api_base_url_here
MODEL=your_model_name_here
```

## 🎯 Using the App

1. **Launch** the app using one of the methods above
2. **Wait** for the browser to open (may take a few seconds)
3. **Upload** your Excel file (.xlsx or .xls) by:
   - Dragging and dropping it onto the upload area, OR
   - Clicking the upload area to browse for a file
4. Click **"Start Classification"**
5. Wait for processing (progress shown on screen)
6. Click **"Download Classified File"** when complete

## 🐛 Troubleshooting

### "Python is not recognized"

**Solution:** Python is not in your system PATH.
1. Install Python from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Restart your computer
4. Or manually add Python to PATH in System Environment Variables

### "Port 5000 is already in use"

**Solution:** Another instance is running or another app is using port 5000.
1. Close any other instances of the app
2. Close any other apps using port 5000
3. Or edit `web_app.py` and change `port=5000` to a different port (e.g., `port=5001`)

### Browser doesn't open automatically

**Solution:** Manually open your browser and go to:
```
http://127.0.0.1:5000
```

### "Module not found" errors

**Solution:** Install missing dependencies:
```cmd
pip install flask pandas openpyxl python-dotenv requests langchain langchain-openai langchain-community langchain-core
```

### PowerShell execution policy error

If you see "execution of scripts is disabled", run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or simply use `launch_app.bat` directly instead of the shortcut.

### File upload fails

- Maximum file size: 50MB
- Supported formats: .xlsx, .xls only
- Ensure the file has at least 11 columns (0-10)
- Column 8 should contain the main text
- Columns 9+ should be topic headers

## 📁 File Structure

```
open_questions/
├── launch_app.bat              # Windows launcher (double-click this)
├── launch_app_silent.vbs      # Silent launcher (optional)
├── create_windows_shortcut.bat # Creates desktop shortcut
├── create_windows_shortcut.ps1 # PowerShell shortcut creator
├── web_app.py                  # Flask web application
├── classify_topics.py          # Classification logic
├── translate_columns.py        # Translation script
├── templates/
│   └── index.html             # Upload page UI
├── uploads/                    # Temporary uploaded files
├── outputs/                    # Classified result files
└── .env                       # API configuration
```

## 🔒 Security Notes

- The app runs locally on your machine (127.0.0.1)
- Files are processed in local `uploads/` and `outputs/` directories
- API keys are loaded from your local `.env` file
- No data is sent anywhere except to your configured API endpoint

## 🗑️ Uninstalling

To remove the desktop shortcut:
- Right-click "Topic Classifier" on Desktop → Delete

The app files will remain in the project folder.

## 💡 Tips

- **Keep the command window open** while using the app (it shows server status)
- **Close the command window** to stop the server
- The app works offline for file upload, but needs internet for API calls
- You can run multiple classifications without restarting the app

## 📞 Need Help?

Check the main `README_WEB_APP.md` for more details, or:
- Check the command window for error messages
- Verify your `.env` file is configured correctly
- Ensure all Python packages are installed

