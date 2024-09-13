
import os
import sqlite3
import nltk
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random
from nltk.tokenize import word_tokenize
from werkzeug.utils import secure_filename

# Ensure NLTK resources are downloaded
nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SQLite database to store student assignments
DATABASE = 'assignments.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            assignment_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Function to store student assignments
def store_assignment(student_name, assignment_text):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO assignments (student_name, assignment_text)
        VALUES (?, ?)
    ''', (student_name, assignment_text))
    conn.commit()
    conn.close()

# Function to retrieve all assignments
def get_all_assignments():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT assignment_text FROM assignments')
    assignments = cursor.fetchall()
    conn.close()
    return [assignment[0] for assignment in assignments]

def detect_plagiarism(submitted_assignment, other_assignments):
    assignments = other_assignments + [submitted_assignment]
    
    vectorizer = TfidfVectorizer().fit_transform(assignments)
    similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    max_similarity = max(similarity_matrix[0]) if similarity_matrix.size else 0
    
    return max_similarity

def detect_ai_generated(text):
    words = word_tokenize(text.lower())
    ai_words = ['algorithm', 'data', 'neural', 'artificial', 'machine', 'learning']
    ai_content_count = sum(1 for word in words if word in ai_words)
    
    ai_percentage = ai_content_count / len(words) if words else 0
    return ai_percentage > 0.2  

def generate_pie_chart(plag_percentage):
    labels = ['Plagiarized', 'Unique']
    sizes = plag_percentage

    # Ensure no negative values are passed
    if any(size < 0 for size in sizes):
        raise ValueError("Wedge sizes for pie chart must be non-negative.")

    colors = ['red', 'green']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    pie_chart_filename = "static/pie_chart.png"
    plt.savefig(pie_chart_filename)
    plt.close()

    return pie_chart_filename


def highlight_plagiarized_words(text, plag_words):
    highlighted_text = text
    for word in plag_words:
        highlighted_text = highlighted_text.replace(word, f'<mark>{word}</mark>')
    return highlighted_text

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    if 'assignment' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['assignment']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        with open(file_path, 'r') as f:
            assignment_text = f.read()

        student_name = request.form.get('student_name', None)
        if not student_name:
            return jsonify({'error': 'Student name is required'}), 400

        store_assignment(student_name, assignment_text)

        other_assignments = get_all_assignments()

        plag_percentage = detect_plagiarism(assignment_text, other_assignments) * 100
        is_ai_generated = detect_ai_generated(assignment_text)

        plag_words = word_tokenize(assignment_text.lower())
        highlighted_text = highlight_plagiarized_words(assignment_text, plag_words)

        pie_chart_filename = generate_pie_chart(plag_percentage)

        return redirect(url_for('results', student_name=student_name, plag_percentage=plag_percentage, 
                                pie_chart_filename=pie_chart_filename, ai_generated=is_ai_generated, 
                                highlighted_text=highlighted_text))

    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/results')
def results():
    student_name = request.args.get('student_name')
    plag_percentage = request.args.get('plag_percentage')
    pie_chart_filename = request.args.get('pie_chart_filename')
    ai_generated = request.args.get('ai_generated') == 'True'
    highlighted_text = request.args.get('highlighted_text')
    
    return render_template('results.html', student_name=student_name, plag_percentage=plag_percentage,
                           pie_chart=f'/static/{pie_chart_filename}', ai_generated=ai_generated,
                           highlighted_text=highlighted_text)

if __name__ == '_main_':
    app.run(debug=True)