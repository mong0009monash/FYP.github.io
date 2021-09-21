import pickle
from random import randint

clf= pickle.load(open("RFC.sav","rb"))

column_len = 55
column_range = range(column_len)

x = [[randint(0,1) for x in column_range]]
print(len(x[0]))
print(clf.predict(x))