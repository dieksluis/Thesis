import dspy
from typing import Literal

class QuestionAnswer(dspy.Signature):
    """
    je krijgt een vraag over een fragment van een rapport dat weer gegeven is in 
    de context.
    Geef antwoord op de vraag door een keuze te maken uit een van de mogelijkheden.
    als er meer antwoorden mogelijk zijn, scheid ze dan met een komma. Wanneer
    het antwoord niet duidelijk uit de tekst te halen is, reageer met 'geen antwoord'
    """
    context: str = dspy.InputField(desc="deel van het rapport")
    question: str = dspy.InputField(desc="de vraag")
    options: list[str] = dspy.InputField(desc="mogelijkheden")
    answer: str = dspy.OutputField()

