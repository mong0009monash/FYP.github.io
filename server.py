from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

model = pickle.load(open("RFC.sav","rb"))

num_checkbox = 55

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods = ["POST","GET"])
def predict():
    if request.method == 'POST':
        lst = getChecks()
        res = model.predict([lst])[0]
        return render_template('index.html', result = res)
    else:
        return render_template('index.html')
    
def getChecks():
    prefix = "model"
    lst = []
    for i in range(1,num_checkbox+1):
        pos = prefix + str(i)
        if pos in request.form:
            lst.append(1)
        else:
            lst.append(0)
    
    print(lst)
    
    return lst


if __name__ == "__main__":
    app.run()