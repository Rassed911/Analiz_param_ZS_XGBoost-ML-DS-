import joblib
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Spiral Frequency Bandwidth Predictor API")
# 1. Загружаем модель и сохраненный порядок колонок локально
model = xgb.Booster()
model.load_model("xgb_model.json")
feature_names = joblib.load("model_features.joblib")

#Определение входящих признаков
class SpiralFeatures(BaseModel):
    shag_spirali_mm: float       
    vnutr_diameter_d_mm: float
    r_provodnika: float
    dielectric_eps: float  

@app.post("/predict")
def predict(features: SpiralFeatures):
    try:
        # 1. Извлекаем данные из Pydantic в словарь
        data_dict = features.model_dump()
        #Создаем DataFrame строго из сохраненного списка признаков
        # Это гарантирует правильный порядок колонок, даже если на вход прислали иначе
        input_data = pd.DataFrame([data_dict], columns=feature_names)
        #Принудительно приводим типы к float
        input_data = input_data.astype('float32')
        #Переводим в родную структуру DMatrix для XGBoost Booster
        dmatrix_data = xgb.DMatrix(input_data)
        prediction = model.predict(dmatrix_data)
        predicted_value = float(prediction[0])
        return {
            "predicted_bandwidth": predicted_value,
            "status": "success"
        }
    except Exception as e:
        # Если что-то упадет, мы увидим понятную ошибку прямо в ответе API
        return {
            "status": "error",
            "message": str(e)
        }