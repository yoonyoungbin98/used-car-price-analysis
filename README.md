# 중고차 가격 예측 시스템

중고차 사이트 데이터를 크롤링하여
차량 정보 기반 가격 예측 모델을 구축한 프로젝트입니다.
FastAPI를 활용하여 예측 API까지 구현했습니다.

# 기술스택
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
catboost
lightgbm
requests
beautifulsoup4
optuna
fastapi

# Features

- 중고차 데이터 크롤링
- 데이터 전처리 및 분석
- 머신러닝 가격 예측
- FastAPI 기반 API 구현

## Result

- RandomForest 모델 성능이 가장 우수
- R² Score: 0.91
- 주행거리와 연식이 가격에 큰 영향 확인
