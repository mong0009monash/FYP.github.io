import pickle
from random import randint
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

clf= pickle.load(open("RFC.sav","rb"))

input_list = [randint(0,1) for x in range(55)]

def create_graph_feature_importances(input_list):
    columns = ['ann', 'anserini', 'rrf', 'bayesian', 'bert', 'bert-base', 'bm',
           'baseline', 'borda', 'cal', 'classifier', 'dense', 'dfo', 'dfr',
           'dirichlet', 'dph', 'electra', 'ensemble', 'f2exp', 'fusion', 'hybrid',
           'idf', 'indri', 'interpolation', 'lambdamart', 'lambdarank', 'lda',
           'learning-to-rank', 'lnu.ltu', 'lucene', 'networks', 'nogueira',
           'non-stopwords', 'nonrel', 'normalization', 'pointwise', 'ponte',
           'pool rank', 'prf', 'ranking', 're-ranking', 'rrf.1', 'scibert',
           'scispacy', 'sofiaml', 'softmax', 'reciprocal-rank', 'regression', 'tf',
           'tf-ranking', 'tf-idf', 'top-k', 'udel', 'vectors', 'weights']
    feature_importance = clf.best_estimator_.feature_importances_
    
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
    
    
    return fig