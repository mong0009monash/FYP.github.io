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
        lst = processList(getChecks())
        print(lst)
        print(len(lst))
        res = model.predict([lst])[0]
        return render_template('index.html', result = res)
    else:
        return render_template('index.html')
    

def processList(lst):
    new_lst = []
    for i in lst:
        if i == None:
            new_lst.append(0)
        else:
            new_lst.append(1)
    
    return new_lst

def getChecks():
    prefix = "model"
    lst = []
    for i in range(1,num_checkbox+1):
        pos = prefix + str(i)
        lst.append(request.form.get(pos))
    
    return lst


if __name__ == "__main__":
    app.run()