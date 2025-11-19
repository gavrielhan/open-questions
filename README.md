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
1. Run `./create_desktop_app.sh` to create the desktop app
2. Double-click `Topic Classifier.app` on your Desktop
3. Browser opens automatically with the upload interface

**Option 2: Terminal**
```bash
python3 web_app.py
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
- Column 8: Main text content (Hebrew)
- Columns 9+: Topic headers (Hebrew)
- The tool will classify each row's text against all topics

## 🔧 Configuration

See `.env.example` (if provided) or check `README_WEB_APP.md` for detailed configuration options.

## 📚 Documentation

- **`README_WEB_APP.md`**: General web app documentation
- **`README_WINDOWS.md`**: Windows-specific setup guide

## 🏗️ Project Structure

```
open-questions/
├── web_app.py                  # Flask web application
├── classify_topics.py          # Classification logic with LangChain
├── translate_columns.py        # Translation script
├── templates/
│   └── index.html             # Upload page UI
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

