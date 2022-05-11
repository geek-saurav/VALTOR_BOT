#importing  the necessary libraries

import pandas as pd
from sklearn import preprocessing 
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import json
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#importing the csv files or  Importing the dataset

Training  = pd.read_csv('./data/Training.csv')
testing = pd.read_csv('./data/Testing.csv')

# Saving the information of columns

Colums = Training.columns
Colums = Colums[:-1]

# Slicing and Dicing the dataset to separate features from predictions

x = Training[Colums]
y = Training['prognosis']
Y1 = y

# Dimensionality Reduction for removing redundancies

filtered_data = Training.groupby(Training['prognosis']).max()

# This section of code to be run after scraping the data

doc_dataset = pd.read_csv('./data/doctors_dataset.csv', names = ['Name', 'Description'])


diseases = filtered_data.index
diseases = pd.DataFrame(diseases)
doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan
doctors['disease'] = diseases['prognosis']
doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']

# Encoding String values to integer constants

length = preprocessing.LabelEncoder()
length.fit(y)
y = length.transform(y)

# Splitting the dataset into training set and test set

x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size=0.33, random_state=42)
Testx    = testing[Colums]
Testy    = testing['prognosis']  
Testy    = length.transform(Testy)

#making a decision tree variable for the model

tree = DecisionTreeClassifier()
trees = tree.fit(x_Train,y_Train)

# for evaluating the score of the model

scores = cross_val_score(trees, x_Test, y_Test, cv=3)
#print (scores.mean())
model=SVC()
model.fit(x_Train,y_Train)
#print("for svm: ")
#print(model.score(x_Test,y_Test))

# Checking the Important features

importances =tree.feature_importances_
indices = np.argsort(importances)[::-1]
features = Colums

print("I am Volter Bot.")
print("I am your personal Consultant Doctor and advisor.")
# print("Please reply Yes or No for the following symptoms")

#defining all the  dictionaries for symtoms,severity,precaution and descriptions of the diseases 

severityDict=dict()
description_dict = dict()
precautionDict=dict()
symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index



def calc_condition(exp,days):
    sum=0
    for item in exp:
        sum=sum+severityDict[item]
    if((sum*days)/(len(exp)+1)>13):
        print("Please consult a doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")

#defining a function for diease description

def Description():
    global description_dict
    with open('./data/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_dict.update(_description)


#defining a function to get severity description

def getSeverityDict():
    global severityDict
    with open("./data/symptom_severity.csv") as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDict.update(_diction)
        except:
            pass

#defining a function for getting the precaution

def getprecautionDict():
    global precautionDict
    with open('./data/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDict.update(_prec)

#getting the info of the user

def getInfo(userText):
    # name=input("Name:")
    print("Please enter your name: \n\t\t\t\t\t\t",end="\n->")
    name=userText
    print("Hello ",name)

# functions to check the pattern
 
def check_pattern(dis_list,inp):
    import re
    pred_list=[]
    ptr=0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:

        if regexp.search(item):
            pred_list.append(item)
            # return 1,item
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return ptr,item

#defining a function for predicting the scondary diseases

def sec_predict(symptoms_exp):
    df = pd.read_csv('./data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    x_Train, x_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(x_Train, y_Train)

    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1


    return rf_clf.predict([input_vector])

#defining the function for printing dieases

def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = length.inverse_transform(val[0])
    return disease


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

#main engine to configure the diseases and giving out description and precautions 

    while True:

        print("Please enter the symptom you are experiencing:\n\t\t\t\t\t\t",end="\n->")
        disease_input = input("")
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break

        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days=int(input("Please enter the number of days you're experiencing this symptom:\t\n"))
            break
        except:
            print("Please enter number of days:")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])

            red_cols = filtered_data.columns 
            symptoms_given = red_cols[filtered_data.loc[present_disease].values[0].nonzero()]

            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("Please provide the correct answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])

                print(description_dict[present_disease[0]])
            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_dict[present_disease[0]])
                print(description_dict[second_prediction[0]])

#code for precaution steps

            precution_list=precautionDict[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)
#code for names of doctors for consultation
            row = doctors[doctors['disease'] == present_disease[0]]
            print('Consult ', str(row['name'].values))
            print()
            print('Visit ', str(row['link'].values))
            
        
#callback the functions
    recurse(0, 1)
getSeverityDict()
Description()
getprecautionDict()

tree_to_code(tree,Colums)

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return getInfo(userText)

if __name__ == "__main__":
    app.run()
