from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import catboost as cb
import json

app = Flask(__name__, template_folder='templates', static_folder='static')

# ✅ 모델 로드
with open("webmodel/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ LabelEncoder 로드
label_encoders = {}
for col in ['배지', '세부 배지', '모델명']:
    with open(f"webmodel/{col}_encoder.pkl", "rb") as f:
        label_encoders[col] = pickle.load(f)

# ✅ 수동 인코딩: 시리즈_명 (원래 모델)
car_label_mapping = {
    '제네시스': 10, '팰리세이드': 9, '그랜저': 8, '싼타페': 7, '스타리아': 6,
    '쏘나타': 5, '투싼': 4, '코나': 3, '아반떼': 2, '스타렉스': 1
}

# ✅ 배지/세부 배지 매핑 로드
with open("webmodel/badge_mapping.json", "r", encoding="utf-8") as f:
    badge_mapping = json.load(f)

# ✅ 평균 가격 매핑 로드
with open("webmodel/model_avg_price.json", "r", encoding="utf-8") as f:
    model_avg_price = json.load(f)

with open("webmodel/model_name_avg_price.json", "r", encoding="utf-8") as f:
    model_name_avg_price = json.load(f)

# ✅ 전처리 함수
def preprocess_input(data):
    try:
        model_name = data.get('model_name')
        model_name_encoded = label_encoders['모델명'].transform([model_name])[0]

        # ✅ 시리즈_명 수동 인코딩
        series_value = -1
        for k, v in car_label_mapping.items():
            if k in model_name:
                series_value = v
                break

        badge = label_encoders['배지'].transform([data.get('badge')])[0]
        badge_detail_raw = data.get('badge_detail')
        if badge_detail_raw in [None, "", "선택 안 함"]:
            badge_detail_raw = "nan"
        badge_detail = label_encoders['세부 배지'].transform([badge_detail_raw])[0]

        year = int(data.get('year'))
        mileage = int(data.get('mileage'))
        model_year = int(data.get('model_year'))

        # ✅ 총 피해 금액 계산
        damage_mycar = float(data.get('damage_mycar', 0))
        damage_othercar = float(data.get('damage_othercar', 0))
        total_damage = damage_mycar + damage_othercar

        # ✅ 평균 가격
        model_avg = model_avg_price.get(str(series_value), 2000)
        model_name_avg = model_name_avg_price.get(model_name, 2000)

        return [
            model_name_encoded,
            year,
            mileage,
            model_year,
            total_damage,
            series_value,
            badge,
            badge_detail,
            model_avg,
            model_name_avg
        ]
    except Exception as e:
        print(f"⚠️ 전처리 오류: {e}")
        return None

# ✅ 예측 API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processed = preprocess_input(data)
    if processed is None:
        return jsonify({"error": "입력값 오류"}), 400
    pred = model.predict([processed])
    return jsonify({"predicted_price": round(pred[0], 2)})

# ✅ 모델명 자동완성용 JS 반환
@app.route('/models.js')
def model_js():
    model_list = list(badge_mapping.keys())
    options = "".join([f'<option value="{m}"></option>' for m in model_list])
    return f"const modelOptions = `{options}`;"

# ✅ 모델명별 배지 / 세부배지 목록 반환
@app.route('/badge-options/<model_name>')
def get_badge_options(model_name):
    result = badge_mapping.get(model_name, {"badge": [], "badge_detail": []})
    return jsonify(result)

# ✅ 홈
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5050)
