from Model.pix2pix import Pix2Pix
from flask import Flask, request, send_file, Response
from API.ColorizationEndpoint import Colorize

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

model = Pix2Pix("FaceGan", None)
model.Load(9)

@app.route('/pix2pix', methods=['POST'])
def ColorizePost():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return 'No file specified'
        print(file)
        colorizedFile = Colorize(model, file)
        return colorizedFile

if __name__ == "__main__":
    app.run(debug=True)