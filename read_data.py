import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_and_preprocess(file_path, seq_len=24):
    df = pd.read_csv(file_path, sep=';', low_memory=False, na_values='?')
    df.dropna(inplace=True)

    # Datetime 병합 및 정렬
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.sort_values('Datetime', inplace=True)
    df.set_index('Datetime', inplace=True)

    # 필요한 컬럼만 사용
    data = df[['Global_active_power']].astype('float32')

    # 정규화
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 시퀀스 데이터 생성
    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    # 훈련/테스트 분할 (80/20)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler
