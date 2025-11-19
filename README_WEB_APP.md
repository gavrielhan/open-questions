# Topic Classification Web Application

A user-friendly web interface for the AI-powered topic classification tool.

**Supports:** macOS and Windows

## 🚀 Quick Start

### macOS

#### Option 1: Use the Desktop App (Recommended)

1. **Double-click** `Topic Classifier.app` on your Desktop
2. A Terminal window will open and your browser will open automatically
3. Drag and drop your Excel file or click to browse
4. Click "Start Classification" and wait for processing
5. Download your classified file when complete

#### Option 2: Use Terminal

```bash
cd "/Users/gavrielhannuna/Desktop/Samuel Neaman/open_questions"
python3 web_app.py
```

### Windows

#### Option 1: Create Desktop Shortcut (Recommended)

1. **Double-click** `create_windows_shortcut.bat` in the project folder
2. A shortcut named "Topic Classifier" will appear on your Desktop
3. **Double-click** the shortcut to launch the app
4. Your browser will open automatically

#### Option 2: Run Directly

1. **Double-click** `launch_app.bat` in the project folder
2. A command window will open and start the web server
3. Your browser will open automatically

**See `README_WINDOWS.md` for detailed Windows setup instructions.**

## 📋 Requirements

- Python 3.7 or higher
- All dependencies from `requirements.txt` installed
- Valid `.env` file with API configuration

## 🔧 Configuration

Make sure your `.env` file contains:

```
API_KEY=your_api_key_here
API_BASE_URL=your_api_base_url_here
MODEL=your_model_name_here
```

Optional settings:
```
OPENAI_MAX_TOKENS=400
OPENAI_TEMPERATURE=0
OPENAI_BATCH_SIZE=5
OPENAI_PARALLEL_WORKERS=2
OPENAI_REQUEST_DELAY_SECONDS=0.5
GEMINI_AI_KEY=your_gemini_key_for_json_repair
```

## 📊 Features

- **Beautiful UI**: Modern, responsive design with drag-and-drop support
- **Real-time Progress**: Visual feedback during classification
- **Batch Processing**: Efficient handling of large Excel files
- **Error Handling**: Robust error messages and retry logic
- **Automatic Download**: One-click download of classified results
- **File Validation**: Ensures only valid Excel files (.xlsx, .xls) are processed

## 🎨 Customizing the Desktop Icon

The app uses the default Terminal icon by default. To customize:

1. Find or create an icon (PNG, JPG, or ICNS format)
2. Right-click on `Topic Classifier.app` → Get Info
3. Drag your icon onto the small icon in the top-left of the Info window
4. The custom icon will be applied immediately

## 🔒 Security Notes

- The app runs locally on your machine (127.0.0.1)
- Files are processed in local `uploads/` and `outputs/` directories
- API keys are loaded from your local `.env` file
- No data is sent anywhere except to your configured API endpoint

## 📁 File Structure

```
open_questions/
├── web_app.py                  # Flask web application
├── classify_topics.py          # Classification logic
├── translate_columns.py        # Translation script
├── templates/
│   └── index.html             # Upload page UI
├── uploads/                    # Temporary uploaded files
├── outputs/                    # Classified result files
├── launch_app.command         # Terminal launcher (alternative)
└── .env                       # API configuration
```

## 🐛 Troubleshooting

### macOS Security Warning

If you see "Topic Classifier cannot be opened because it is from an unidentified developer":

1. Go to System Preferences → Security & Privacy
2. Click "Open Anyway" at the bottom
3. Confirm by clicking "Open"

### Port Already in Use

If port 5000 is already in use:

1. Edit `web_app.py`
2. Change `app.run(debug=False, port=5000, ...)` to use a different port
3. Save and restart the app

### Configuration Error

If you see "Configuration Error" on the upload page:

1. Verify your `.env` file exists in the same directory as `web_app.py`
2. Check that `API_KEY`, `API_BASE_URL`, and `MODEL` are set correctly
3. Restart the application

### File Upload Fails

- Maximum file size: 50MB
- Supported formats: .xlsx, .xls only
- Ensure the file has at least 11 columns (0-10)
- Column 8 should contain the main text
- Columns 9+ should be topic headers

## 📞 Support

For issues or questions, check:
- Console.app logs (search for "TopicClassifier")
- Terminal output when running `python3 web_app.py` directly

## 🗑️ Uninstalling

To remove the desktop app:
- Drag `Topic Classifier.app` from Desktop to Trash
- Optionally delete the project folder

The web interface and command-line tools will still work.

