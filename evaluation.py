import pandas as pd
import csv

#This function will compare the results of the LLm to the actual answers
def Evaluation(df, rapports, models): 
    for i in range(len(rapports)):
        title = rapports[i][0]
        for j in range(len(models)):
            model = models.iloc[j]['Model Name']
            #Compare the answer row to the LLm row
            vergelijking = (
                df.loc[title].str.strip().str.lower() == 
                df.loc[title + " " + model].str.strip().str.lower()
            )
            nieuwe_rij = pd.DataFrame(
                [vergelijking], 
                index=[title + " " + model + " Evaluation"]
                )
            df = pd.concat([df, nieuwe_rij], axis=0)
            df.index.name = 'Name'
    return(df)

#This function will calculate all the scores, how good did the LLm perform
def Score(df, rapports, questions, models):
    x = rapports * len(models)
    y = questions
    #Make sure al the Booleans are Booleans
    df = df.map(lambda x: True if str(x).strip().lower() == 'true' else (
                          False if str(x).strip().lower() == 'false' else x))
    
    for i in range(len(models)):
        model = models.iloc[i]['Model Name'] 
        evaluation = df.filter(like=model + " Evaluation" , axis=0)
        totaal_per_vraag = evaluation.sum().astype(int).astype(str) + '/' + str(rapports)
        df.loc['Score ' + model] = totaal_per_vraag
    

    #Calculate score per question
    evaluation = df.filter(like='Evaluation', axis=0)
    totaal_per_vraag = evaluation.sum().astype(int).astype(str) + '/' + str(x)
    df.loc['Score Total'] = totaal_per_vraag

   #Calculate score per rapport
    evaluation_rows = [i for i in df.index if 'Evaluation' in str(i)]
    df['score'] = None  
    for i in evaluation_rows:
        juist = (
            df.loc[i].apply(lambda x: str(x).strip().lower() == 
            'true') if df.loc[i].dtype == object else df.loc[idx]
        )
        score = f"{int(juist.sum())}/{y}"
        df.loc[i, 'Score'] = score

    #Calculate total score
    aantal_true = (df == True).sum().sum()
    df.iloc[-1, -1] = f"{int(aantal_true)}/{rapports*questions*len(models)}"

    return(df)