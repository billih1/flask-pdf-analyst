import os
import re
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from transformers import pipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- LOAD MODELS ---
print("Loading models... this may take a minute.")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
print("Models loaded!")

current_context = ""

def preprocess_text(text):
    # Clean up weird PDF formatting
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_context
    summary = ""
    answer = ""
    
    if request.method == 'POST':
        # --- User Uploads PDF ---
        if 'pdf_file' in request.files:
            file = request.files['pdf_file']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # 1. Extract & Preprocess
                raw_text = extract_text_from_pdf(filepath)
                clean_text = preprocess_text(raw_text)
                current_context = clean_text

                # 2. Chunking for Summary
                chunk_size = 2000 
                chunks = [clean_text[i:i+chunk_size] for i in range(0, len(clean_text), chunk_size)]
                
                summary_list = []
                for i, chunk in enumerate(chunks[:15]): 
                    # Summarize each chunk
                    try:
                        result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                        summary_list.append(result[0]['summary_text'])
                    except Exception as e:
                        print(f"Skipping chunk {i} due to error: {e}")
                        continue
                
                summary = " ".join(summary_list)

        # --- SCENARIO B: User Asks Question ---
        elif 'question' in request.form:
            question = request.form['question']
            if current_context:
                # --- UPDATE: Increased context to 30,000 chars (Approx 10-12 pages) ---
                # The model will look for the answer in this larger window.
                search_context = current_context[:30000]
                
                try:
                    result = qa_pipeline(question=question, context=search_context)
                    answer = result['answer']
                except Exception as e:
                    answer = "I couldn't find the answer in the first 10 pages, or the text was too complex."
                
                summary = request.form.get('previous_summary', '')
            else:
                answer = "Please upload a PDF first!"

    return render_template('index.html', summary=summary, answer=answer, previous_summary=summary)

if __name__ == '__main__':
    app.run(debug=True)