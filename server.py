from flask import Flask, render_template
import pickle
import requests

app = Flask(__name__)

model = pickle.load(open("RFC.sav","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods = ["POST","GET"])
def predict():
    values = requests.json["data"]
    print(values)
    
    return True
    
    #clf.predict()


if __name__ == "__main__":
    app.run()