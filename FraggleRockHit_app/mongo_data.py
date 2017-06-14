from pymongo import MongoClient
import urllib.request, json
import numpy as np
import pandas as pd

with urllib.request.urlopen("http://galvanize-case-study-on-fraud.herokuapp.com/data_point") as url:
    data = json.loads(url.read().decode())
    # print(data)
    print("Object Id: {}".format(data['object_id']))


client = MongoClient()
#set database that collection is in
db = client.fraud
#collection for stats variables to go in


current_proba = np.random.uniform(0, 1)
current_fraud = np.random.uniform(0, 1) 

db.review.update(data, data, upsert=True)

db.review.update_one({'object_id' : data['object_id']},
     {'$set' : {'proba': current_proba}})

db.review.update_one({'object_id' : data['object_id']},
     {'$set' : {'fraud_proba': current_fraud}})

df = pd.DataFrame(list(db.review.find()))
