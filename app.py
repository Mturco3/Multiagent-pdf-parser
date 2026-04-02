import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from src.orchestrator import Orchestrator
from dotenv import load_dotenv

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'supersecretkey'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

load_dotenv()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    source_code = None
    markdown_content = None
    output_md = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return render_template('upload.html')
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template('upload.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            output_md = filename.rsplit('.', 1)[0] + '.md'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_md)
            api_keys = {"GROQ": os.getenv("GROQ_API_KEY"), "GOOGLE": os.getenv("GOOGLE_API_KEY")}
            orchestrator = Orchestrator(upload_path, output_path, api_keys)
            orchestrator.run()
            # Read source code and markdown
            with open(upload_path, 'rb') as f:
                source_code = f.read()
            with open(output_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            flash(f'Markdown generated: {output_md}')
            return render_template('upload.html', source_code=source_code, markdown_content=markdown_content, output_md=output_md)
    return render_template('upload.html')

@app.route('/outputs/<filename>')
def download_file(filename):
    return redirect(url_for('static', filename=f'outputs/{filename}'))

if __name__ == '__main__':
    app.run(debug=True)
