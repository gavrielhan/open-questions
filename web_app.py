#!/usr/bin/env python3
"""
Flask web application for Excel topic classification.
Provides a web interface for uploading and processing Excel files.
"""
from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# Import our classification module
from classify_topics import OpenAIConfig, update_topics
import pandas as pd


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['OUTPUT_FOLDER'] = Path(__file__).parent / 'outputs'

# Create necessary directories
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}


def allowed_file(filename: str) -> bool:
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main upload page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and classification."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload .xlsx or .xls file'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = app.config['UPLOAD_FOLDER'] / filename
        file.save(input_path)
        
        # Load configuration
        config = OpenAIConfig.from_env()
        
        # Read Excel file
        df = pd.read_excel(input_path)
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        
        if len(df.columns) <= 10:
            return jsonify({'error': f'Expected at least 11 columns, got {len(df.columns)}'}), 400
        
        main_column = df.columns[8]
        topics = df.columns[9:].tolist()
        
        # Process classification
        df = update_topics(
            df,
            main_column,
            topics,
            config,
            limit=None,
            skip_existing=False,
        )
        
        # Save output
        output_filename = f"{Path(filename).stem}_classified.xlsx"
        output_path = app.config['OUTPUT_FOLDER'] / output_filename
        df.to_excel(output_path, index=False)
        
        return jsonify({
            'success': True,
            'message': 'Classification completed successfully',
            'output_file': output_filename,
            'rows_processed': len(df),
            'topics_count': len(topics)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download processed file."""
    output_path = app.config['OUTPUT_FOLDER'] / secure_filename(filename)
    if not output_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(output_path, as_attachment=True)


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
    print("  Topic Classification Web App")
    print("=" * 50)
    print("Starting server...")
    print("Opening browser at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    print()
    
    try:
        app.run(debug=False, port=5000, host='127.0.0.1')
    except OSError as e:
        if "Address already in use" in str(e) or "address is already in use" in str(e):
            print("\n❌ ERROR: Port 5000 is already in use!")
            print("   Another instance might be running, or another app is using port 5000.")
            print("   Please close it and try again, or edit web_app.py to use a different port.")
            sys.exit(1)
        else:
            raise

