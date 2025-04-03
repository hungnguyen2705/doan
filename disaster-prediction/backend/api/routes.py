from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
import os
from datetime import datetime

api = Blueprint('api', __name__)

def load_models():
    """Load both ML models"""
    try:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        print(f"Loading models from: {models_dir}")  # Debug print
        rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.joblib'))
        xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.joblib'))
        return rf_model, xgb_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

    return rf_model, xgb_model

@api.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input data
        if not all(key in data for key in ['province', 'month']):
            return jsonify({
                'success': False,
                'error': 'Thiếu thông tin đầu vào'
            }), 400

        # Chuyển đổi tháng thành số
        month = int(data['month'])
        year = 2025  # Năm dự đoán cố định
        
        # Giả lập dữ liệu trung bình mưa (có thể thay bằng dữ liệu thực tế)
        mua_tb_3thang = 150.0  # Giá trị mặc định cho demo
        mua_tb_12thang = 200.0  # Giá trị mặc định cho demo

        # Chuẩn bị dữ liệu đầu vào theo format mô hình
        input_data = pd.DataFrame({
            'month': [month],
            'year': [year],
            'mua_tb_3thang': [mua_tb_3thang],
            'mua_tb_12thang': [mua_tb_12thang]
        })

        # Load và sử dụng cả 2 mô hình
        rf_model, xgb_model = load_models()
        
        # Thực hiện dự đoán
        rf_prediction = float(rf_model.predict(input_data)[0])
        xgb_prediction = float(xgb_model.predict(input_data)[0])
        
        # Tính trung bình của 2 mô hình
        avg_prediction = (rf_prediction + xgb_prediction) / 2

        return jsonify({
            'success': True,
            'random_forest_prediction': rf_prediction,
            'xgboost_prediction': xgb_prediction,
            'average_prediction': avg_prediction,
            'province': data['province'],
            'month': month,
            'year': year,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500