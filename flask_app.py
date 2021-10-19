from flask import Flask, render_template, request, Response
import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import io
import base64
import configparser


#Use settings in config.ini
config = configparser.ConfigParser()
config.read("config.ini")


try:
    #Try using settings in config.ini and load into the program
    
    model_address = config.get("DEFAULT","modeladdress")
    features_number = config.getint("DEFAULT","checkboxnumber")
    
    app = Flask(__name__)
    model = pickle.load(open(model_address,"rb"))
    num_checkbox = features_number
except:
    #If config.ini not found, create config.ini file with default settings
    
    app = Flask(__name__)   
    model = pickle.load(open("RFC.sav","rb"))
    num_checkbox = 47
    
    config = configparser.ConfigParser()
    config["DEFAULT"] = {"ModelAddress": "RFC.sav",
                         "CheckboxNumber" : 47}

    with open("config.ini","w") as configfile:
        config.write(configfile)


@app.route("/")
def index():
    """
    Default function to run if running index.html page.
    
    Return:
        Return index.html on url of /.
    
    """
    return render_template("index.html")



@app.route("/predict", methods = ["POST","GET"])
def predict():
    """
    Predict function, runs when user access /predict of html.
    
    Return:
        Return result.html on url of /predict. 

    """
    
    #if it's POST request
    if request.method == 'POST':
        lst = getChecks()
        
        #check if list has 1's
        if 1 not in lst:
            #if no 1's found, return to default page with message.
            return render_template('index.html', result = "Please select 1 or more." , txt_color = "red")
        
        #compile result : [model result, model probability, result's graph]
        res = [model.predict([lst])[0],model.predict_proba([lst])[0][1], get_graph(lst)]
        
        color = ""
        res_txt = ""
        
        #change colour of result background
        if res[0] == True:
            #if model result is true, return green
            color = "4FE34F"
            res_txt = "High Performance"
        else:
            #else red
            color = "FF7F7F"
            res_txt = "Low Performance"
        
        #return all the compiled result to result.html and render it out on url of /predict
        return render_template('result.html', result = res_txt, color = color, confidence = round(res[1],3), graph = res[2])
    else:
        #if it's GET request, return default page
        return render_template('index.html')


def get_graph(input_list):
    """
    Get graph of the predict result's feature importances of input_list
    
    Argument:
        input_list : a list of 1's and 0's to indicate which feature was selected.
        
    
    Return:
        Return graph of the feature importances of the given input_list

    """
    
    columns = ['ann', 'anserini', 'bayesian', 'bm', 'baseline', 'borda', 'cal',
       'classifier', 'dense', 'dfo', 'dfr', 'dirichlet', 'dph', 'electra',
       'ensemble', 'f2exp', 'fusion', 'hybrid', 'indri', 'interpolation',
       'lambdamart', 'lambdarank', 'lda', 'learning-to-rank', 'lnu.ltu',
       'lucene', 'non-stopwords', 'nonrel', 'normalization', 'pointwise',
       'pool rank', 'prf', 'ranking', 're-ranking', 'scibert', 'scispacy',
       'sofiaml', 'softmax', 'regression', 'tf-ranking', 'top-k', 'udel',
       'vectors', 'weights', 'reciprocal', 'tf_idf', 'BERT']
    feature_importance = model.best_estimator_.feature_importances_
    
    filtered_col = []
    filtered_imp = []
    
    #get selected feature and put them into a list
    for i in range(len(columns)):
        if input_list[i] == 1:
            filtered_col.append(columns[i])
            filtered_imp.append(feature_importance[i])
    
    
    filtered_col = np.array(filtered_col)
    filtered_imp = np.array(filtered_imp)
    
    sorted_idx = filtered_imp.argsort()
    top_columns = filtered_col[sorted_idx]
    top_columns_score = filtered_imp[sorted_idx]

    #create graph
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.barh(top_columns, top_columns_score)
    axis.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        top=False,         
        labelbottom=False)
    axis.set_title("Feature Importance for RF model")
    
    fig.set_size_inches(12, 9)
    
    #change graph into format of which the HTML can read
    img = io.BytesIO()
    
    fig.savefig(img, format="png")
    img.seek(0)
    
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url


def getChecks():
    """
    Get the current checked feature selection        
    
    Return:
        Return a list of 1's and 0's of the feature selected.

    """
    
    #Select out names with prefix of model with number x concatenate to it
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