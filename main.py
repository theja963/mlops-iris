import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model, predict
from datetime import datetime

app = FastAPI(
    title="Iris Predictor",
    docs_url="/"
)

app.add_event_handler("startup", load_model)

class QueryIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class QueryOut(BaseModel):
    flower_class: str


@app.get("/ping")
def ping():
    return {"ping": "pong" ,"timestamp": datetime.now().strftime("%m/%D/%Y, %h:%m:%S")}

@app.get("/gotcha")
def gotcha():
    return {"gotcha": "Hello There"}

@app.get("/terminate")
def terminate():
    return {"Terminate":"Task Terminated"}

@app.post("/predict_flower", response_model=QueryOut, status_code=200)
def predict_flower(
    query_data: QueryIn
):
    output = {'flower_class': predict(query_data)}
    return output

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8888, reload=True)
