#!/usr/bin/env python3
"""
Enhanced Flask web application for Excel topic classification.
Provides a web interface with file validation, column selection, and real-time progress.
"""
from __future__ import annotations

import io
import os
import sys
import webbrowser
from pathlib import Path
from threading import Timer
from datetime import datetime
from queue import Queue
import threading

from flask import Flask, render_template, request, send_file, jsonify, session
from werkzeug.utils import secure_filename
import pandas as pd

# Import our classification module
from classify_topics import OpenAIConfig, update_topics


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['OUTPUT_FOLDER'] = Path(os.path.expanduser('~/Desktop'))  # Save to desktop
app.secret_key = os.urandom(24)  # For session management

# Create necessary directories
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

# Store for progress messages per session
progress_queues = {}


def allowed_file(filename: str) -> bool:
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def column_letter_to_index(letter: str) -> int:
    """Convert Excel column letter (A, B, AA, etc.) to 0-based index."""
    letter = letter.upper().strip()
    result = 0
    for char in letter:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1


def column_index_to_letter(index: int) -> str:
    """Convert 0-based index to Excel column letter."""
    letter = ''
    index += 1  # Convert to 1-based
    while index > 0:
        index -= 1
        letter = chr(index % 26 + ord('A')) + letter
        index //= 26
    return letter


def read_file(filepath: Path, sheet_name=None):
    """Read Excel or CSV file and return DataFrame and available sheets."""
    extension = filepath.suffix.lower()
    
    if extension == '.csv':
        df = pd.read_csv(filepath)
        return df, ['Sheet1']  # CSV has only one sheet
    elif extension in ['.xlsx', '.xls']:
        # Get all sheet names
        xl_file = pd.ExcelFile(filepath)
        sheet_names = xl_file.sheet_names
        
        # Read specific sheet or first sheet
        if sheet_name:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            df = pd.read_excel(filepath, sheet_name=0)
        
        return df, sheet_names
    else:
        raise ValueError(f"Unsupported file type: {extension}")


class ProgressCapture:
    """Capture print statements for progress display."""
    def __init__(self, session_id: str, queue: Queue):
        self.session_id = session_id
        self.queue = queue
        self.terminal = sys.stdout
        
    def write(self, message):
        if message.strip():  # Ignore empty messages
            try:
                self.queue.put(message.strip())
            except Exception:
                pass  # Silently fail if queue is closed
        self.terminal.write(message)
    
    def flush(self):
        self.terminal.flush()


@app.route('/')
def index():
    """Main upload page."""
    return render_template('index_enhanced.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and validation."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload .xlsx, .xls, or .csv file'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
        
        input_path = app.config['UPLOAD_FOLDER'] / f"{session_id}_{filename}"
        file.save(input_path)
        
        # Read file and get metadata
        df, sheet_names = read_file(input_path)
        
        # Get column information
        columns = df.columns.tolist()
        column_info = [
            {
                'index': i,
                'letter': column_index_to_letter(i),
                'name': str(col)
            }
            for i, col in enumerate(columns)
        ]
        
        # Store file info in session
        session['uploaded_file'] = str(input_path)
        session['original_filename'] = filename
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': filename,
            'sheets': sheet_names,
            'columns': column_info,
            'num_rows': len(df),
            'num_columns': len(columns)
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/preview', methods=['POST'])
def preview_data():
    """Preview data from selected sheet and columns."""
    try:
        data = request.json
        sheet_name = data.get('sheet_name')
        answer_col = data.get('answer_column')
        topic_start = data.get('topic_start_column')
        topic_end = data.get('topic_end_column')
        
        if 'uploaded_file' not in session:
            return jsonify({'error': 'No file uploaded'}), 400
        
        filepath = Path(session['uploaded_file'])
        if not filepath.exists():
            return jsonify({'error': 'Uploaded file not found'}), 400
        
        # Read the specified sheet
        df, _ = read_file(filepath, sheet_name)
        
        # Convert column letters to indices
        answer_idx = column_letter_to_index(answer_col)
        topic_start_idx = column_letter_to_index(topic_start)
        topic_end_idx = column_letter_to_index(topic_end)
        
        # Validate indices
        if answer_idx >= len(df.columns) or topic_start_idx >= len(df.columns) or topic_end_idx >= len(df.columns):
            return jsonify({'error': 'Invalid column selection'}), 400
        
        if topic_end_idx < topic_start_idx:
            return jsonify({'error': 'End column must be after or equal to start column'}), 400
        
        # Get column names
        answer_column_name = df.columns[answer_idx]
        topic_columns = df.columns[topic_start_idx:topic_end_idx + 1].tolist()
        
        # Get preview data (first 5 rows)
        preview_data = []
        for i in range(min(5, len(df))):
            row_data = {
                'row_num': i + 1,
                'answer': str(df.iloc[i, answer_idx])[:100] + '...' if len(str(df.iloc[i, answer_idx])) > 100 else str(df.iloc[i, answer_idx])
            }
            preview_data.append(row_data)
        
        # Format topic columns as numbered list
        topic_columns_formatted = []
        for idx, topic in enumerate(topic_columns, 1):
            topic_columns_formatted.append(f"{idx}) {topic}")
        
        return jsonify({
            'success': True,
            'answer_column': answer_column_name,
            'topic_columns': topic_columns,
            'topic_columns_formatted': topic_columns_formatted,
            'num_topics': len(topic_columns),
            'preview': preview_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/classify', methods=['POST'])
def classify():
    """Start classification process."""
    try:
        data = request.json
        sheet_name = data.get('sheet_name')
        answer_col = data.get('answer_column')
        topic_start = data.get('topic_start_column')
        topic_end = data.get('topic_end_column')
        
        if 'uploaded_file' not in session:
            return jsonify({'error': 'No file uploaded'}), 400
        
        session_id = session.get('session_id')
        filepath = Path(session['uploaded_file'])
        
        # Initialize progress queue for this session
        progress_queues[session_id] = Queue()
        
        # Read the file
        df, _ = read_file(filepath, sheet_name)
        
        # Strip whitespace from column names
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        
        # Convert column letters to indices
        answer_idx = column_letter_to_index(answer_col)
        topic_start_idx = column_letter_to_index(topic_start)
        topic_end_idx = column_letter_to_index(topic_end)
        
        # Get column names
        main_column = df.columns[answer_idx]
        topic_cols = df.columns[topic_start_idx:topic_end_idx + 1].tolist()
        
        # Load configuration
        config = OpenAIConfig.from_env()
        
        # Get queue and output filename before thread starts
        queue = progress_queues[session_id]
        original_filename = session['original_filename']
        
        # Run classification in a separate thread
        def run_classification():
            # Redirect stdout to capture progress
            old_stdout = sys.stdout
            sys.stdout = ProgressCapture(session_id, queue)
            
            try:
                queue.put(f"Starting classification...")
                queue.put(f"Model: {config.model}")
                queue.put(f"Processing {len(df)} rows with {len(topic_cols)} topics")
                
                # Run classification
                df_classified = update_topics(
                    df,
                    main_column,
                    topic_cols,
                    config,
                    limit=None,
                    skip_existing=False,
                )
                
                # Save to desktop as CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                original_name = Path(original_filename).stem
                output_filename = f"{original_name}_classified_{timestamp}.csv"
                output_path = app.config['OUTPUT_FOLDER'] / output_filename
                
                df_classified.to_csv(output_path, index=False)
                
                queue.put(f"✅ COMPLETE: Saved to {output_path}")
                queue.put("DONE")
                
            except Exception as e:
                queue.put(f"❌ ERROR: {str(e)}")
                queue.put("ERROR")
            finally:
                sys.stdout = old_stdout
        
        # Start classification thread
        thread = threading.Thread(target=run_classification)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Classification started',
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/progress/<session_id>')
def get_progress(session_id):
    """Get progress updates for a session."""
    if session_id not in progress_queues:
        return jsonify({'messages': [], 'done': False})
    
    messages = []
    done = False
    error = False
    
    # Get all messages from queue
    queue = progress_queues[session_id]
    while not queue.empty():
        msg = queue.get()
        if msg == "DONE":
            done = True
        elif msg == "ERROR":
            done = True
            error = True
        else:
            messages.append(msg)
    
    return jsonify({
        'messages': messages,
        'done': done,
        'error': error
    })


@app.route('/status')
def status():
    """Check if the server is running and configuration is valid."""
    try:
        config = OpenAIConfig.from_env()
        return jsonify({
            'status': 'ready',
            'model': config.model,
            'api_base_url': config.api_base_url
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


def open_browser():
    """Open browser after a short delay."""
    webbrowser.open('http://127.0.0.1:5000')


if __name__ == '__main__':
    # Open browser automatically (works on both macOS and Windows)
    Timer(1, open_browser).start()
    
    # Run Flask app
    print("=" * 50)
    print("  Topic Classification Web App (Enhanced)")
    print("=" * 50)
    print("Starting server...")
    print("Opening browser at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    print()
    
    try:
        app.run(debug=False, port=5000, host='127.0.0.1', threaded=True)
    except OSError as e:
        if "Address already in use" in str(e) or "address is already in use" in str(e):
            print("\n❌ ERROR: Port 5000 is already in use!")
            print("   Another instance might be running, or another app is using port 5000.")
            print("   Please close it and try again, or edit web_app_enhanced.py to use a different port.")
            sys.exit(1)
        else:
            raise

