import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

players = pd.read_csv("top5_leagues_player.csv")  # 데이터 로드
players.head()  # 데이터 확인

sns.pairplot(players, y_vars="price", x_vars=['age', 'shirt_nr', 'foot', 'league'])  # Pairplot 시각화
plt.show()  # 그래프 출력

# 아웃라이어 처리
outlier = players[(players['age'] > 35) | (players['shirt_nr'] > 50)].index
new_players = players.drop(outlier)

# 숫자형 열만 선택
numeric_columns = new_players.select_dtypes(include=['number']).columns
numeric_data = new_players[numeric_columns]

# 숫자형 열 중에서도 문자열 값을 포함하는 열 확인
string_columns = []
for col in numeric_data.columns:
    if new_players[col].dtype == 'O':  # 'O'는 Object 타입 (문자열)
        string_columns.append(col)

# 숫자로 변환되지 않는 열 제거
clean_new_players = numeric_data.drop(string_columns, axis=1)

# 상관관계 히트맵
plt.figure(figsize=(10,10))
sns.heatmap(clean_new_players.corr(), annot=True, cmap='Oranges')
plt.show()

# 'Unnamed: 0', 'name', 'full_name' 불필요 컬럼 제거
clean_players = new_players.drop(["Unnamed: 0", "name", "full_name"], axis=1)

clean_players.dropna(inplace=True)
# clean_player.fillna(0,inplace=True) 0으로 채워도된다.
# max , min 최댓값 최솟값으로 채워도된다.

from sklearn.preprocessing import LabelEncoder
label_players = clean_players.copy()

cols= ['nationality', 'place_of_birth', 'position', 'outfitter', 'club', 'player_agent', 'foot', 'joined_club']
le = LabelEncoder()

for col in cols:
    label_players[col] = le.fit_transform(label_players[col])

label_players.head()

#원핫벡터
players_vectors = pd.get_dummies(label_players, drop_first=True, columns=["contract_expires","league"])

#데이터분리
from sklearn.model_selection import train_test_split

y = players_vectors['price']
X = players_vectors.drop("price", axis=1)

X_train, X_valid, y_train,y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#머신러닝 모델링
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=30, max_depth=9, random_state=42, min_samples_leaf=2, min_samples_split=3)
rf_model.fit(X_train, y_train)

#평가
from sklearn.metrics import mean_squared_error,mean_absolute_error

y_pred = rf_model.predict(X_valid)

rfr_mae = mean_absolute_error(y_valid, y_pred)
rfr_mse = mean_squared_error(y_valid, y_pred)
print(rfr_mae, rfr_mse)

#딥러닝 모델링
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

estopping = EarlyStopping(monitor='val_loss')
mcheckpoint = ModelCheckpoint(monitor='val_loss', filepath='soccer_price_model.h5', save_best_only=True)

from sklearn.preprocessing import StandardScaler

# 데이터 분할을 통해 X_test 생성
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler를 이용하여 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
history = model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test), callbacks=[estopping, mcheckpoint])

# 모델 예측
y_pred = model.predict(X_test_scaled)

plt.plot(history.history['loss'], 'y', label = 'mse')
plt.plot(history.history['val_loss'], 'r', label = 'val_mse')
plt.title("Model MSE")
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# 이적료가 높은 상위 10명의 선수 선택
top_10_players = players.nlargest(15, 'price')

# 막대 그래프로 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='price', y='full_name', data=top_10_players, palette='viridis')
plt.xlabel('Price')
plt.ylabel('Player')
plt.title('Top 10 Players with Highest Transfer Fee')
plt.show()

# 한국 선수들의 데이터를 필터링합니다.
korean_players = players[players['nationality'] == 'Korea, South']

# 한국 선수들 중 이적료가 높은 상위 10명을 선택합니다.
top_10_korean_players = korean_players.nlargest(10, 'price')

# 막대 그래프로 한국 선수들의 이적료를 시각화합니다.
plt.figure(figsize=(10, 6))
sns.barplot(x='price', y='name', data=top_10_korean_players, palette='viridis')
plt.xlabel('Price')
plt.ylabel('Player')
plt.title('Top 10 South Korean Players with Highest Transfer Fee')
plt.show()

print("원본 데이터셋 행의 개수:", players.shape[0])

# 아웃라이어 처리 후 데이터셋 행의 개수 확인
print("아웃라이어 처리 후 행의 개수:", new_players.shape[0])

# 문자열 값을 포함하지 않는 숫자형 열만 선택한 데이터셋 행의 개수 확인
print("문자열 값을 포함하지 않는 숫자형 열만 선택한 후 행의 개수:", clean_new_players.shape[0])

# 불필요한 열 제거 및 결측치 제거 후 최종 데이터셋 행의 개수 확인
print("불필요한 열 제거 및 결측치 제거 후 최종 데이터셋 행의 개수:", clean_players.shape[0])

# 최종적으로 사용된 피쳐 엔지니어링 및 인코딩 후의 데이터셋 행의 개수 확인
print("최종적으로 사용된 데이터셋 행의 개수:", players_vectors.shape[0])

# 학습 및 테스트 데이터셋으로 분리한 후의 행의 개수 확인
print("훈련 데이터셋 행의 개수:", X_train.shape[0])
print("검증 데이터셋 행의 개수:", X_valid.shape[0])

rfr_mae = mean_absolute_error(y_valid, y_pred)  # 머신러닝 모델의 MAE 계산
rfr_mse = mean_squared_error(y_valid, y_pred)  # 머신러닝 모델의 MSE 계산
print(rfr_mae, rfr_mse)

# ...

plt.plot(history.history['loss'], 'y', label='mse')  # 딥러닝 모델의 MSE 시각화
plt.plot(history.history['val_loss'], 'r', label='val_mse')  # 딥러닝 모델의 validation MSE 시각화
plt.title("Model MSE")
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

