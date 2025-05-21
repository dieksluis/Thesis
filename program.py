import dspy
import os
import pandas as pd
import csv
from signature import QuestionAnswer

#function to make csv file to list of dictionaries     
def csv_naar_lijst(csv_file):
    vragenlijst = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        #Skip the Header row
        next(reader)
        for row in reader:
           if len(row) >= 2:
            vraag = row[0].strip()
            antwoorden = [opt.strip() for opt in row[1].split(',')]
            vragenlijst.append({vraag: antwoorden})
    return(vragenlijst)

#This function ask the questions to the LLm using the DSPy Signature  
#The anwswers will be returned in the form of a list of strings  
def AskQuestions(context, questions, lengte):
    answer_list = []
    reasoning_list = []
    for i in range(lengte): 
        result = dspy.ChainOfThought(QuestionAnswer)
        response = result(context=context, question=list(questions[i].keys())[0], 
                          options=list(questions[i].values())[0])
        answer_list.append(response.answer)
        #Chain of Thought module forces LLMs to output chain of thought
        reasoning_list.append(response.reasoning)
    return(answer_list, reasoning_list)

#This function will put the generated answers in the existing DataFrame
def MakeList(end_file, reasoning, rapports, questions, model_name):
    for i in range(len(rapports)): 
        title = rapports[i][0]
        context = rapports[i][1]
        answer_list, reasoning_list = AskQuestions(context, questions, len(questions))
        answer_list.insert(0, title + " " + model_name)
        reasoning_list.insert(0, title + " " + model_name + " Reasoning")
        end_file.loc[answer_list[0]] = answer_list[1:]
        reasoning.loc[reasoning_list[0]] = reasoning_list[1:]
    return(end_file, reasoning)

#Create a seperate DataFile with only the answers of the mojority
def MajorityVote(df, rapports):
    for i in range(len(rapports)):
        rapport = rapports[i][0]
        majority = df.filter(like=rapport + " ", axis=0)
        result = {}
        #Keep the result that is chosen the most. if there is none return 
        #Geen Antwoord'
        for col in majority.columns:
            modes = majority[col].mode()
            if len(modes) == 1:
                result[col] = modes[0]
            else:
                result[col] = 'Geen antwoord'
        df.loc[rapport + ' Majority'] = result
        #Drop the results from the models, we only need the Majority rows
        df = df.drop(majority.index)
    return(df)
    