from fastapi import FastAPI
import pandas as pd
import pickle as pk
from sklearn.preprocessing import PolynomialFeatures

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# Cough_symptoms	Fever	Sore_throat	Shortness_of_breath	  Headache	Known_contact
@app.get("/covid/")
def read_item(
    Cough_symptoms: bool,
    Fever: bool,
    Sore_throat: bool,
    Shortness_of_breath: bool,
    Headache: bool,
    Known_contact: str,
):
    if Known_contact == "Abroad":
        Known_contact = 0
    elif Known_contact == "Contact with confirmed":
        Known_contact = 1
    else:
        Known_contact = 2
    df = pd.DataFrame(
        {
            "Cough_symptoms": [Cough_symptoms],
            "Fever": [Fever],
            "Sore_throat": [Sore_throat],
            "Shortness_of_breath": [Shortness_of_breath],
            "Headache": [Headache],
            "Known_contact": [Known_contact],
        }
    )
    dbfile = open("Covid_Classifier.pickle", "rb")
    model = pk.load(dbfile)
    result = model.predict(df)
    if result:
        result = "Positive"
    else:
        result = "Negative"
    return {"Covid": result}
