from flask import Flask, render_template, request, Response
import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import io
import base64


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
        
        if 1 not in lst:
            return render_template('index.html', result = "Please select 1 or more." , txt_color = "red")
        
        
        res = [model.predict([lst])[0],model.predict_proba([lst])[0][1], get_graph(lst)]
        
        color = ""
        res_txt = ""
        
        if res[0] == True:
            color = "4FE34F"
            res_txt = "High Performance"
        else:
            color = "FF7F7F"
            res_txt = "Low Performance"
        
        return render_template('result.html', result = res_txt, color = color, confidence = res[1], graph = res[2])
    else:
        return render_template('index.html')


def get_graph(input_list):
    columns = ['ann', 'anserini', 'rrf', 'bayesian', 'bert', 'bert-base', 'bm',
           'baseline', 'borda', 'cal', 'classifier', 'dense', 'dfo', 'dfr',
           'dirichlet', 'dph', 'electra', 'ensemble', 'f2exp', 'fusion', 'hybrid',
           'idf', 'indri', 'interpolation', 'lambdamart', 'lambdarank', 'lda',
           'learning-to-rank', 'lnu.ltu', 'lucene', 'networks', 'nogueira',
           'non-stopwords', 'nonrel', 'normalization', 'pointwise', 'ponte',
           'pool rank', 'prf', 'ranking', 're-ranking', 'rrf.1', 'scibert',
           'scispacy', 'sofiaml', 'softmax', 'reciprocal-rank', 'regression', 'tf',
           'tf-ranking', 'tf-idf', 'top-k', 'udel', 'vectors', 'weights']
    feature_importance = model.best_estimator_.feature_importances_
    
    filtered_col = []
    filtered_imp = []
    
    for i in range(len(columns)):
        if input_list[i] == 1:
            filtered_col.append(columns[i])
            filtered_imp.append(feature_importance[i])
    
    
    filtered_col = np.array(filtered_col)
    filtered_imp = np.array(filtered_imp)
    
    sorted_idx = filtered_imp.argsort()
    top_columns = filtered_col[sorted_idx]
    top_columns_score = filtered_imp[sorted_idx]

    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.barh(top_columns, top_columns_score)
    axis.set_title("Feature Importance for RF model")
    
    fig.set_size_inches(12, 9)
    
    
    img = io.BytesIO()
    
    fig.savefig(img, format="png")
    img.seek(0)
    
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url


def getChecks():
    prefix = "model"
    lst = []
    for i in range(1,num_checkbox+1):
        pos = prefix + str(i)
        if pos in request.form:
            lst.append(1)
        else:
            lst.append(0)
    
    return lst


if __name__ == "__main__":
    app.run()