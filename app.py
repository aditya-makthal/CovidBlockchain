from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import ipfshttpclient


app = Flask(__name__)
model = load_model('model.h5')
target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):

    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            resize = tf.image.resize(img, (256,256))
            yhat = model.predict(np.expand_dims(resize/255, 0))
            if yhat > 0.5: 
                res = "Positive"
                return render_template('predict1.html', loki = res,user_image = file_path)
            else:
                res = "Negative"
                return render_template('predict.html', loki = res,user_image = file_path)
        else:
            return "The given file is not in supported format. Please use the supported image format"

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=5000)