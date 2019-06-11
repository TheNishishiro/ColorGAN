import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pix2pix_recon as p2p
from forms import GenerationSettings

UPLOAD_FOLDER = './input/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = 'af0045892ec3ef2ce55cb27e9eef8990'

model = "face_v8"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def upload_file():
    form = GenerationSettings()
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            p2p.predictUploaded(model, filename)
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('home.html', text = "Upload picture", form=form)

@app.route('/uploaded_file', methods=['GET', 'POST'])
def uploaded_file():
    hists = os.listdir('./static/output/')
    hists = ['output/' + file for file in hists]
    return render_template('thispc.html', hists = hists)



if __name__ == "__main__":
    app.run(debug=True)
