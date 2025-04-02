import os
import pandas
from openai import OpenAI
import dspy
from dspy import LM
from signature import QuestionAnswer

#function to read text files 
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

#This function asks al the questions to the LLm using the DSPy Signature  
#The anwswers will be returned in the form of a list of strings  
def AskQuestions(context):
    answer_list = []
    reasoning_list = []

    question1 = "In welke levensfase zit het systeem?"
    options1 = ['Nieuwbouw', 'Doorontwikkeling', 'Beheer en' \
        'onderhoud', 'Vervanging', 'Geen antwoord']
    result = dspy.ChainOfThought(QuestionAnswer)
    response = result(context=context, question=question1, options=options1)
    answer_list.append(response.answer)
    reasoning_list.append(response.reasoning)

    question2 = "Wordt een substantieel gedeelte van de activiteit " \
        "door leveranciers uitgevoerd?"
    options2 = ['Geen leveranciers', 'Één leverancier',
        'Meerdere leveranciers', 'Leveranciersconsortium', 'Geen antwoord']
    result = dspy.ChainOfThought(QuestionAnswer)
    response = result(context=context, question=question2, options=options2)
    answer_list.append(response.answer)
    reasoning_list.append(response.reasoning)

    question3 = "Om hoeveel applicaties gaat het?"
    options3 = ['Één', 'Enkele', 'Een landschap van applicaties', 'Geen antwoord']
    result = dspy.ChainOfThought(QuestionAnswer)
    response = result(context=context, question=question3, options=options3)
    answer_list.append(response.answer)
    reasoning_list.append(response.reasoning)

    question4 = 'Wordt het systeem (of de systemen) door één of meerdere organisaties' \
        'gebruikt? In het geval van meerdere organisaties, gaat het dan om een keten ' \
        'of netwerk van samenwerkende organisaties of om een andere soort groep' \
        'van organisaties?'
    options4 = ['Één organisatie', 'Netwerk van organisaties', 'Ander soort groep',\
                'Geen antwoord']
    result = dspy.ChainOfThought(QuestionAnswer)
    response = result(context=context, question=question4, options=options4)
    answer_list.append(response.answer)
    reasoning_list.append(response.reasoning)

    question5 = "Wat voor soort applicatie is het?"
    options5 = ['Desktop applicatie', 'Webapplicatie', 'Mobiele applicatie', \
                'Middleware', 'Embedded software', 'Anders', 'Geen antwoord']
    result = dspy.ChainOfThought(QuestionAnswer)
    response = result(context=context, question=question5, options=options5)
    answer_list.append(response.answer)
    reasoning_list.append(response.reasoning)

    question6 = "Is het maatwerk of pakket software?"
    options6 = ['Maatwerk', 'Standaard pakket', 'Aangepast pakket', 'Geen antwoord']
    result = dspy.ChainOfThought(QuestionAnswer)
    response = result(context=context, question=question6, options=options6)
    answer_list.append(response.answer)
    reasoning_list.append(response.reasoning)

    question7 = "Wat voor soort systeem is het?"
    options7 = ['Interactief systeem', 'Transactie verwerkend systeem', \
                'Besturingssysteem', 'Rapportage systeem', 'gegevens beheer systeem', \
                'Anders', 'Geen antwoord']
    result = dspy.ChainOfThought(QuestionAnswer)
    response = result(context=context, question=question7, options=options7)
    answer_list.append(response.answer)
    reasoning_list.append(response.reasoning)

    question8 = "Wie zijn de gebruikers  van het systeem?"
    options8 = ['Medewerkers van een uitvoeringsorganisatie', 'Andere ambtenaren', \
                'Burgers', 'Anders', 'Geen antwoord']
    result = dspy.ChainOfThought(QuestionAnswer)
    response = result(context=context, question=question8, options=options8)
    answer_list.append(response.answer)
    reasoning_list.append(response.reasoning)

    question9 = "Hoe is de hosting van het systeem ingedeeld"
    options9 = ['Eigen datacentrum', 'Co-location', 'Cloud platform', 'Geen antwoord']
    result = dspy.ChainOfThought(QuestionAnswer)
    response = result(context=context, question=question9, options=options9)
    answer_list.append(response.answer)
    reasoning_list.append(response.reasoning)



    return(answer_list, reasoning_list)
    
    
if __name__ == "__main__":
    #retrieve text file
    text = load_text_file('Thesis.txt')
    #set API Key
    OPENAI_API_KEY = load_text_file('api.txt')

    #choose your Large Language Model
    lm = dspy.LM('openai/gpt-3.5-turbo', max_tokens=3000, 
        api_key=OPENAI_API_KEY)
    dspy.configure(lm=lm)

    context = text
    answer_list, reasoning_list = AskQuestions(context) 

    #Put in csv file
    df = pandas.DataFrame(data={"col1": answer_list, "col2": reasoning_list})
    df.to_csv("./file.csv", sep=',',index=False)
    #print(answer_list)
    #print(reasoning_list)