import dspy
from typing import Literal

#Very simple DSPy signature that tells the models what to do with each question
class QuestionAnswer(dspy.Signature):
    """
    Je krijgt een vraag over een fragment van een rapport dat weer gegeven is in de context.
    Geef antwoord op de vraag door een keuze te maken uit de gegeven antwoorden.
    De vragen gaan altijd over het nieuwe systeem of applicatie. 
    Wanneer het antwoord niet uit de tekst te halen is, reageer met 'Geen antwoord'.
    """
    context: str = dspy.InputField(desc="deel van het rapport")
    question: str = dspy.InputField(desc="de vraag")
    options: list[str] = dspy.InputField(desc="mogelijkheden")
    answer: str = dspy.OutputField()

