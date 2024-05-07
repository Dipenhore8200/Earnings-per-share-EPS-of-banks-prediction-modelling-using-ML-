from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import xgboost as xgb 
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates") 

# Load XGBoost model (assuming 'eps_v1.json' is in JSON format)
model = xgb.Booster()
model.load_model('eps_v1.json')
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(
    request: Request,
    ROCE: float = Form(...),
    CASA: float = Form(...),
    Return_on_Equity: float = Form(...), 
    Non_Interest_Income_Total_Assets: float = Form(...), 
    Operating_Profit_Total_Assets: float = Form(...),
    Operating_Expenses_Total_Assets: float = Form(...),
    Interest_Expenses_Total_Assets: float = Form(...),
    Face_value: float = Form(...)   
):
    input_data = xgb.DMatrix([
        ROCE, 
        CASA, 
        Return_on_Equity, 
        Non_Interest_Income_Total_Assets, 
        Operating_Profit_Total_Assets,
        Operating_Expenses_Total_Assets,
        Interest_Expenses_Total_Assets,
        Face_value
    ]) 
    result = model.predict(input_data)[0]  
    return templates.TemplateResponse("index.html", {"request": request, "result": result}) 
