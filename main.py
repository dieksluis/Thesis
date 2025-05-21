import os
import csv
from openai import OpenAI
from groq import Groq
import dspy
from dspy import LM
import numpy as np
import pandas as pd
from program import csv_naar_lijst, MakeList, MajorityVote
from evaluation import Evaluation, Score
from signature import QuestionAnswer

if __name__ == "__main__":

    #Put the rapports in a list
    with open('Rapports_tabel.csv', mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=';')
        #Skip header row
        next(reader)
        rapports = list(reader) 
    #put the list of answers in a pandas dataframe
    answers = pd.read_csv('Answers_tabel.csv', delimiter=';', header=0, index_col=0)
    #Create a tupled list with questions and possible answers
    questions = csv_naar_lijst('Vragen.csv')
    #Put different models in Dataframe 
    models = pd.read_csv('Models.csv', delimiter=';', header=0)
    #Create the File that will conatain al the answers later in the program
    end_file = answers.copy()
    #Create the DataFrame that will contain all the Chain of Thought of the models
    reasoning = answers.copy() 
    
    
    for i in range(len(models)):
        #Set Model
        model = models.iloc[i]['Model Name'] 
        #Set API Key
        OPENAI_API_KEY = models.iloc[i]['Api_Key']
        #Configure the model with DSPy
        lm = dspy.LM(model, max_tokens=1000, api_key=OPENAI_API_KEY)
        dspy.configure(lm=lm)
        #Make the dataframe that will lead to our main csv file. 
        end_file, reasoning = MakeList(end_file, reasoning, rapports, questions, model)

    #Clean the dataframe
    end_file = end_file.map(lambda x: x.replace('"', ''))
    end_file = end_file.map(lambda x: x.replace('[', '').replace(']', ''))
    #Make extra Dataframe for the majority model
    majority = end_file.copy()
    majority = MajorityVote(majority, rapports)
    majority_model = pd.DataFrame({'Model Name': ['Majority']})
    #Evaluate the answers from the LLm's against the self found answers
    end_file = Evaluation(end_file, rapports, models)
    majority = Evaluation(majority, rapports, majority_model)
    #Sort the dataframe 
    reasoning = reasoning.sort_values('Name', ascending=True)
    end_file = end_file.sort_values('Name', ascending=True)
    majority = majority.sort_values('Name', ascending=True)
    #Give score to Evaluation
    end_file = Score(end_file, len(rapports), len(questions), models)
    majority = Score(majority, len(rapports), len(questions), majority_model)
    #Write Dataframe to CSV file 
    end_file.to_csv('End_file.csv', sep=';', 
                    encoding='utf-8', index=True, header=True
        )

    majority.to_csv('Majority.csv', sep=';', 
                        encoding='utf-8', index=True, header=True
        )
    reasoning.to_csv('Reasoning.csv', sep=';', 
                        encoding='utf-8', index=True, header=True
        )
    
    