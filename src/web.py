from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from main import Main
import os

app = Flask(__name__)

UPLOAD_FOLDER = '../data/test'
ALLOWED_EXTENSIONS = set(['txt'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', search=True)
    else:
        return get_result()

def get_result():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        tuples, ratings = analyze(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('index.html', tuples=tuples, ratings=ratings)


def analyze(input_filename):
    m = Main()
    conll_filename = "../data/test/test.conll"
    output_filename = "../data/test/preprocessed_reviews.txt"

    # preproses kalimat
    m.preprocess(input_filename, output_filename)
    # jadiin conll table
    m.get_conll_table(output_filename, conll_filename)
    # ekstraksi aspek
    m.get_aspects(conll_filename)
    # split setiap sentence
    m.split_sentences()
    # ekstraksi kategori
    m.get_categories()
    # ekstraksi sentimen
    m.get_sentiments()
    # generate tuple aspek, kategori, sentimen
    tuples, tuples_unique = m.get_tuples()

    return tuples_unique, m.get_ratings(tuples)

if __name__ == '__main__':
    app.run()
