# Open Questions - Topic Classification Tool

AI-powered topic classification tool for Excel files with a beautiful web interface. Supports both macOS and Windows.

## 🚀 Features

- **Web-based Interface**: Modern, responsive UI with drag-and-drop file upload
- **AI-Powered Classification**: Uses LLM to classify topics in Hebrew text
- **Cross-Platform**: Works on both macOS and Windows
- **Batch Processing**: Efficient handling of large Excel files
- **Validation Chain**: Two-step classification with validation for accuracy
- **Translation Support**: Optional translation of Hebrew content to English

## 📋 Requirements

- Python 3.7 or higher
- All dependencies from `requirements.txt`
- Valid API configuration in `.env` file

## 🧰 Full Setup on a New Machine

1. **Install prerequisites**
   - Python 3.9+ (recommended) and `pip`
   - Git
   - (Optional) `virtualenv` or `conda`
2. **Clone the repo**
   ```bash
   git clone https://github.com/gavrielhan/open-questions.git
   cd open-questions
   ```
3. **Create and activate a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows
   ```
4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
5. **Create your `.env`**
   ```
   API_KEY=your_api_key_here
   API_BASE_URL=https://your-litellm-endpoint
   MODEL=openai/gpt-4o
   # Optional extras:
   # GEMINI_AI_KEY=...
   # OPENAI_MAX_TOKENS=400
   ```
6. **Run a smoke test**
   ```bash
   python3 web_app_enhanced.py
   ```
   - Browser should open at `http://127.0.0.1:5000`
   - Upload a small Excel/CSV to verify everything works
7. **Create a desktop launcher (optional but recommended)**
   - macOS: `chmod +x create_desktop_app.sh && ./create_desktop_app.sh`
   - Windows: double-click `create_windows_shortcut.bat`

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/gavrielhan/open-questions.git
cd open-questions
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API configuration:
```
API_KEY=your_api_key_here
API_BASE_URL=your_api_base_url_here
MODEL=your_model_name_here
```

## 🎯 Quick Start

### macOS

**Option 1: Desktop App**
1. Run `chmod +x create_desktop_app.sh && ./create_desktop_app.sh`
2. Double-click `Topic Classifier.app` on your Desktop
3. Browser opens automatically with the upload interface

**Option 2: Terminal**
```bash
python3 web_app_enhanced.py
```

### Windows

**Option 1: Desktop Shortcut**
1. Double-click `create_windows_shortcut.bat`
2. Double-click the "Topic Classifier" shortcut on Desktop

**Option 2: Direct Launch**
```cmd
launch_app.bat
```

## 📖 Usage

1. **Launch the app** using one of the methods above
2. **Upload** your Excel file (.xlsx or .xls)
3. **Wait** for classification (progress shown on screen)
4. **Download** your classified file

### Excel File Format

- Must have at least 11 columns (0-10)
- `classify_topics.py` defaults:
  - For Excel: column index 8 = main text, columns 9+ = topics
  - For CSV (Med Students dataset): column index 0 = main text, columns 1-9 = topics
- The tool will classify each row's text against all topics
- To run from CLI:
  ```bash
  python3 classify_topics.py --input "path/to/file.xlsx"
  # optional flags:
  #   --limit 100
  #   --skip-existing
  ```

## 🔧 Configuration

See `.env.example` (if provided) or check `README_WEB_APP.md` for detailed configuration options.

## 📚 Documentation

- **`README_WEB_APP.md`**: General web app documentation
- **`README_WINDOWS.md`**: Windows-specific setup guide
- **`README_ENHANCED.md`**: Detailed walkthrough of the enhanced upload UI

## 🏗️ Project Structure

```
open-questions/
├── web_app.py                  # Flask web application
├── web_app_enhanced.py         # Enhanced upload interface with sheet/column selection
├── classify_topics.py          # Classification logic with LangChain
├── translate_columns.py        # Translation script
├── templates/
│   └── index.html             # Upload page UI
│   └── index_enhanced.html    # Enhanced upload UI
├── launch_app.bat             # Windows launcher
├── launch_app.command         # macOS launcher
├── create_desktop_app.sh      # macOS app creator
├── create_windows_shortcut.bat # Windows shortcut creator
└── requirements.txt           # Python dependencies
```

## 🔒 Security

- All processing happens locally
- API keys stored in `.env` (not committed to git)
- Files processed in local `uploads/` and `outputs/` directories
- No data sent except to your configured API endpoint

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for use.

## 🙏 Acknowledgments

Built with:
- Flask for the web interface
- LangChain for LLM orchestration
- pandas for Excel processing
- OpenAI-compatible APIs for classification

