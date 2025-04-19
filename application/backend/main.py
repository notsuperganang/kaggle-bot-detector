import uuid
import uvicorn
import pickle
from fastapi import File
from fastapi import UploadFile
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import EmailStr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# membuat instance FastAPI
app = FastAPI()

# mendefinisikan schema input 
class UserInput(BaseModel):
    follower_count: float
    following_count: float
    dataset_count: float
    code_count: float
    discussion_count: float
    avg_nb_read_time_min: float
    is_glogin: bool

def preprocess_pipeline():
    numeric_features = ['follower_count', 'following_count', 'dataset_count', 
                        'code_count', 'discussion_count', 'avg_nb_read_time_min']  # masukkan kolom-kolom numerik

    categorical_boolean_features = ['is_glogin'] # masukkan kolom kategorikal dan boolean

    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())  ## MinMaxScaler untuk fitur numerik sesuai notebook
    ])

    categorical_boolean_transformer = Pipeline(steps=[
        ('encoder', FunctionTransformer(lambda x: x.astype(int)))  ## Ubah boolean ke integer (0/1)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_boolean_transformer, categorical_boolean_features)
        ],
        remainder='drop'  # Drop kolom yang tidak di-transformasi
    )

    return preprocessor

# load model
# Sesuaikan path dengan lokasi model Anda
try:
    with open('../../model/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    # Alternatif path jika path pertama tidak ditemukan
    with open('../model/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)

# endpoint untuk menerima input dan menghasilkan prediksi
@app.post("/predict/", summary="Melakukan klasifikasi apakah suatu user tergolong bot atau bukan")
async def predict(user_input: UserInput):
    # Ubah input menjadi format yang sesuai (pandas DataFrame)
    data = pd.DataFrame({
        'follower_count': [user_input.follower_count],
        'following_count': [user_input.following_count],
        'dataset_count': [user_input.dataset_count],
        'code_count': [user_input.code_count],
        'discussion_count': [user_input.discussion_count],
        'avg_nb_read_time_min': [user_input.avg_nb_read_time_min],
        'is_glogin': [user_input.is_glogin]
    })
    
    # Terapkan pipeline untuk preprocessing
    preprocessing_pipeline = preprocess_pipeline()
    processed_data = preprocessing_pipeline.fit_transform(data)
    
    # Prediksi dengan model yang sudah dilatih
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[:, 1]
    
    # Mengembalikan hasil prediksi dan probabilitas
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0]),
        "is_bot": bool(prediction[0])
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)