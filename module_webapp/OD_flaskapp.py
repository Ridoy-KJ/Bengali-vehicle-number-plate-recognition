from flask import Flask, render_template, request
import os
from deeplearning import OCR

# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
UPLOAD_PATH = os.path.join(BASE_PATH, 'static', 'upload')
os.makedirs(UPLOAD_PATH, exist_ok=True)


@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = os.path.basename(upload_file.filename)
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        text = OCR(path_save,filename)
        return render_template('index.html',upload=True,upload_image=filename,text=text)
        
        
        
    return render_template('index.html',upload=False)

if __name__ == "__main__":
    app.run(debug=True)