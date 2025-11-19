# Enhanced Topic Classification Web App

The enhanced version provides a step-by-step interface with full control over file processing and real-time progress monitoring.

## 🎯 New Features

### 1. **File Validation**
- Supports `.xlsx`, `.xls`, and `.csv` files
- Validates file type before processing
- Shows file information (rows, columns)

### 2. **Sheet Selection**
- For Excel files with multiple sheets
- Dropdown menu to select which sheet to analyze
- Auto-selects if only one sheet exists

### 3. **Column Selection by Letter**
- **Answer Column**: Letter of the column containing text to analyze (e.g., `H`)
- **Topics Start**: Letter where topic columns begin (e.g., `I`)
- **Topics End**: Letter where topic columns end (e.g., `Z`)

### 4. **Preview Before Running**
- Shows first 5 rows of selected data
- Displays topic column names
- Confirms selection before processing

### 5. **Real-time Progress Console**
- Terminal-style console showing model progress
- See exactly what the model is doing
- Batch progress tracking
- Error messages displayed immediately

### 6. **Desktop CSV Output**
- Results saved directly to Desktop
- Timestamped filename
- CSV format for easy viewing in Excel

## 🚀 How to Use

### Step-by-Step Process

1. **Upload File**
   - Drag & drop or click to browse
   - Supported: `.xlsx`, `.xls`, `.csv` files
   - Max size: 100MB

2. **Select Sheet** (if Excel with multiple sheets)
   - Choose which sheet to analyze from dropdown
   - Auto-selected if only one sheet

3. **Specify Columns**
   - **Answer Column Letter**: Column with text (e.g., `H` for column H)
   - **Topics Start Letter**: First topic column (e.g., `I`)
   - **Topics End Letter**: Last topic column (e.g., `Z`)
   - Click **"Preview Selection"** to verify

4. **Review Preview**
   - See sample of your data
   - Verify column names are correct
   - Check number of topics

5. **Run Classification**
   - Click **"Run Program"** button
   - Watch real-time progress in console
   - Wait for completion message

6. **Get Results**
   - CSV file automatically saved to Desktop
   - Filename: `[original]_classified_[timestamp].csv`

## 📋 Requirements

Same as main app:
- Python 3.7+
- All dependencies from `requirements.txt`
- Valid `.env` file with API configuration

## 🔧 Configuration

Your `.env` file should contain:

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
GEMINI_AI_KEY=your_gemini_key
```

## 🎨 User Interface

### Progress Console
The black console shows:
- Model initialization
- Batch processing progress
- Row counts and status
- Completion message
- Any errors encountered

Example output:
```
Starting classification...
Model: openai/gpt-4o-mini
Processing 100 rows with 15 topics
Processing 100 rows in 20 batches (batch_size=5, workers=2)...
✔ Batch 1/20 processed by model
✔ Batch 2/20 processed by model
...
✅ COMPLETE: Saved to /Users/yourname/Desktop/data_classified_20241119_143022.csv
```

## 🐛 Troubleshooting

### Invalid Column Letter
- Use capital letters: `A`, `B`, `C`, ... `Z`, `AA`, `AB`, etc.
- Don't include numbers or special characters
- Make sure the column exists in your file

### Sheet Not Showing
- CSV files only have one sheet (auto-selected)
- Excel files should show all sheets in dropdown
- Refresh page if dropdown is empty

### Preview Shows Wrong Data
- Double-check your column letters
- Remember: A=1st column, B=2nd, etc.
- Preview shows first 5 rows only

### Classification Doesn't Start
- Make sure all steps are green (completed)
- Check console for error messages
- Verify `.env` file exists and is correct

### Progress Console Stops
- Check if error message appeared
- Look for red error messages in console
- Verify internet connection for API calls

## 💡 Tips

1. **Column Letters**: Excel column letters work exactly like in spreadsheets
   - A, B, C... Z, AA, AB... AZ, BA, etc.

2. **Testing**: Use a small subset first
   - Try with 10-20 rows to verify setup
   - Then process full file

3. **Multiple Files**: 
   - Upload one file at a time
   - Previous results are saved to Desktop
   - Refresh page to start new file

4. **CSV Output**:
   - Opens directly in Excel/Numbers/Google Sheets
   - Preserves all original data plus classifications
   - Timestamped to prevent overwriting

## 📂 Output File Location

Results are saved to:
- **macOS**: `/Users/[username]/Desktop/`
- **Windows**: `C:\Users\[username]\Desktop\`

Filename format:
```
[original_filename]_classified_[YYYYMMDD]_[HHMMSS].csv
```

Example:
```
survey_data_classified_20241119_143022.csv
```

## 🔄 Running Multiple Classifications

To process another file:
1. Refresh the browser page
2. Upload new file
3. Configure columns
4. Run again

Each output gets a unique timestamp.

## 🆘 Need Help?

Check the console output for specific error messages. Common issues:

- **Port already in use**: Close other instances
- **API key error**: Check `.env` file
- **Column not found**: Verify column letters
- **File too large**: Split file into smaller parts

