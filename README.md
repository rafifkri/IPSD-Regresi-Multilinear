# Analisis dan Prediksi Harga Saham TLKM Menggunakan Regresi Linear

## Kaggle

https://www.kaggle.com/datasets/irfansaputranst/dataset-saham-tlkm-jk

## About Dataset

Dataset ini berisi data saham dari PT Telekomunikasi Indonesia Tbk (TLKM.JK) yang diperdagangkan di Bursa Efek Indonesia (BEI). Berikut adalah penjelasan untuk masing-masing kolom:

* Date: Tanggal pencatatan data dalam format DD/MM/YYYY.
* Adj Close: Harga penutupan yang telah disesuaikan untuk memberikan nilai lebih akurat, termasuk dividen dan pembagian saham.
* Close: Harga penutupan pada akhir hari perdagangan.
* High: Harga tertinggi yang dicapai saham pada hari perdagangan tersebut.
* Low: Harga terendah yang dicapai saham pada hari perdagangan tersebut.
* Open: Harga pembukaan saham di awal hari perdagangan.
* Volume: Jumlah volume saham yang diperdagangkan pada hari tersebut (dalam format angka bertitik, perlu diubah ke format numerik untuk analisis).

## Alur Program

### Program 1

#### melakukan impor beberapa library Python yang umum digunakan untuk analisis data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```

### Program 2

#### Membaca isi dari file csv yang digunakan dan menampilkan hanya 10 baris pertama dari data.

```python
df = pd.read_csv('SAHAM - PT Telekomunikasi Indonesia Tbk (TLKM.JK) - Sheet1.csv')
df.head(10)
```
<img width="411" height="289" alt="image" src="https://github.com/user-attachments/assets/731f0ae2-6a88-4f88-bbbf-89001d4a145b" />


### Program 3

#### Memberikan info singkat mengenai struktur yang ada pada DataFrame.

```python
df.info()
```
<img width="262" height="203" alt="image" src="https://github.com/user-attachments/assets/bb5e0d94-499e-4dc7-a7e5-0624abb2518d" />

### Program 4

#### memberikan info singkat mengenai statistik deskriptif untuk kolom-kolom numerik dalam DataFrame.

```python
df.describe()
```
<img width="448" height="235" alt="image" src="https://github.com/user-attachments/assets/6f49ece3-1fdb-4d64-930b-8e49c76aa1ca" />

### Program 5

#### Mengecek apakah pada DataFrame ada nilai null atau tidak.

```python
df.isna().sum()
```
<img width="84" height="230" alt="image" src="https://github.com/user-attachments/assets/8c296e83-9bf2-48ae-9dce-b6f3fb4e1e51" />

### Program 6

#### menciptakan fitur-fitur baru dari kolom 'Adj Close' yang sudah ada, yang sangat relevan untuk analisis deret waktu, dan kemudian membersihkan data dari nilai-nilai yang hilang yang mungkin muncul akibat proses pembuatan fitur baru tersebut.

```python
df['Volume'] = df['Volume'].astype(str).str.replace('.', '', regex=False).astype(float)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values(by='Date')
print("Data types after conversion and sorting:")
df.info()
```
<img width="300" height="224" alt="image" src="https://github.com/user-attachments/assets/f548951f-2c5a-4132-8950-c77f0105b061" />

### Program 7

#### Menyiapkan variabel target (y) dan fitur input (X) untuk digunakan dalam proses training model regresi linear.

```python
y = df['Adj Close']
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]

print("Shape of target variable y:", y.shape)
print("Shape of feature matrix X:", X.shape)
```
<img width="238" height="41" alt="image" src="https://github.com/user-attachments/assets/f3047638-07c2-4242-9f06-04fb36b5f6e0" />

### Program 8

#### Membagi dua dataset menjadi data tarin dan data test.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
```
<img width="183" height="68" alt="image" src="https://github.com/user-attachments/assets/380a303c-5aee-45e9-baae-885208157707" />

### Program 9

#### Melatih model Linear Regression agar dapat mempelajari hubungan antara variabel-variabel fitur (Open, High, Low, Close, Volume) dengan target (Adjusted Close).

```python
model = LinearRegression()
model.fit(X_train, y_train)

print("Linear Regression model trained successfully.")
```
### Program 10

######  Model Evaluation, Setelah model dilatih menggunakan data historis saham TLKM, dilakukan evaluasi menggunakan data uji untuk mengukur performanya.  

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```
<img width="257" height="67" alt="image" src="https://github.com/user-attachments/assets/800f403d-dab5-4cd5-8701-1e51c8487002" />

* MSE (Mean Squared Error): mengukur rata-rata kuadrat kesalahan prediksi.
* RMSE (Root Mean Squared Error): menunjukkan rata-rata kesalahan model dalam satuan harga saham.
* MAE (Mean Absolute Error): menggambarkan rata-rata besar kesalahan prediksi.
* R² (R-squared): menunjukkan seberapa besar variasi harga saham yang dapat dijelaskan oleh model.

### Program 11

#### membandingkan hasil prediksi model regresi linear dengan nilai aktual pada data uji, kita ingin lihat seberapa jauh perbedaan antara harga saham aktual (y_test) dan hasil prediksi (y_pred).

```python
results_df = pd.DataFrame({'Actual Adj Close': y_test, 'Predicted Adj Close': y_pred})
results_df['Difference'] = results_df['Actual Adj Close'] - results_df['Predicted Adj Close']
display(results_df.head())
display(results_df.tail())
```
<img width="344" height="310" alt="image" src="https://github.com/user-attachments/assets/96c5c4ff-7982-4e04-9530-04b79dbe9785" />

### Program 12

#### melihat bobot (koefisien) dari setiap variabel independen dalam model regresi linear. Koefisien ini menjelaskan seberapa besar pengaruh masing-masing fitur (seperti Open, High, Low, Close, Volume) terhadap target variabel yaitu Adj Close.

```python
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coefficients)
```
<img width="163" height="103" alt="image" src="https://github.com/user-attachments/assets/f9c299f0-217a-4a7d-93af-4f106583cf32" />

### Program 13

#### Visualisasi ini digunakan untuk membandingkan hasil prediksi model (y_pred) dengan nilai sebenarnya (y_test) dalam bentuk grafik scatter plot. Tujuannya adalah melihat seberapa dekat prediksi model terhadap nilai aktual harga saham TLKM.

```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Adj Close')
plt.ylabel('Predicted Adj Close')
plt.title('Actual vs. Predicted Adj Close Values')
plt.grid(True)
plt.show()
```

<img width="686" height="437" alt="Screenshot 2025-11-09 120829" src="https://github.com/user-attachments/assets/76fb47d4-4e20-45e8-9e92-865e71f9a3b8" />

### Program 14

#### mengevaluasi kesalahan prediksi (residual) dari model regresi linear terhadap data saham TLKM. Residual = selisih antara nilai aktual dan prediksi (Actual - Predicted). Analisis residual membantu memastikan apakah model sudah sesuai asumsi regresi linear (yaitu error menyebar acak tanpa pola tertentu).

```python
df = pd.read_csv('SAHAM - PT Telekomunikasi Indonesia Tbk (TLKM.JK) - Sheet1.csv')
df['Volume'] = df['Volume'].astype(str).str.replace('.', '', regex=False).astype(float)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values(by='Date')
y = df['Adj Close']
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results_df = pd.DataFrame({'Actual Adj Close': y_test, 'Predicted Adj Close': y_pred})
results_df['Difference'] = results_df['Actual Adj Close'] - results_df['Predicted Adj Close']

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1) 
plt.scatter(results_df['Predicted Adj Close'], results_df['Difference'], alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Adj Close')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.grid(True)

plt.subplot(1, 2, 2) 
plt.hist(results_df['Difference'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True)

plt.tight_layout()
plt.show()
```
<img width="910" height="383" alt="image" src="https://github.com/user-attachments/assets/4fdc123c-32ef-4d31-8ded-11df22730313" />

### Program 15

#### Visualisasi berikut menampilkan perbandingan antara nilai aktual dan prediksi harga saham TLKM dari model regresi linear dengan menunjukkan seberapa dekat hasil prediksi model regresi linear terhadap harga saham TLKM sebenarnya dari waktu ke waktu.

```python
plot_df = pd.DataFrame({
    'Date': df.loc[y_test.index, 'Date'],
    'Actual Adj Close': y_test,
    'Predicted Adj Close': y_pred
})

plot_df = plot_df.sort_values(by='Date').reset_index(drop=True)

plt.figure(figsize=(14, 7))
plt.plot(plot_df['Date'], plot_df['Actual Adj Close'], label='Actual Adj Close', color='blue', linewidth=2)
plt.plot(plot_df['Date'], plot_df['Predicted Adj Close'], label='Predicted Adj Close', color='red', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Adj Close Value')
plt.title('Actual vs. Predicted Adj Close Over Time (Test Set)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
<img width="901" height="447" alt="image" src="https://github.com/user-attachments/assets/3de77312-0d7d-4455-8882-7a77e3a75e3b" />

### Program 16

#### Membuat fitur-fitur baru yang relevan dengan data deret waktu, seperti nilai tertunda (lagged values) dari 'Adj Close' atau fitur lain yang mungkin relevan, seperti rata-rata bergerak atau indikator teknis lainnya jika sesuai.

```python
df['Adj Close_Lag1'] = df['Adj Close'].shift(1)
df['Adj Close_Lag2'] = df['Adj Close'].shift(2)
df['Adj Close_Lag3'] = df['Adj Close'].shift(3)

df['SMA_7'] = df['Adj Close'].rolling(window=7).mean()
df['SMA_30'] = df['Adj Close'].rolling(window=30).mean()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("membuat fitur baru.")
df.head()
```
<img width="875" height="171" alt="image" src="https://github.com/user-attachments/assets/59fcb662-8451-4d9c-94e3-454997b7a778" />

### Program 17

#### Program ini digunakan untuk melatih dan mengevaluasi model regresi linear menggunakan data saham TLKM, setelah dibuat fitur tambahan yang bisa membantu model memahami pola pergerakan harga dari waktu ke waktu dan mengukur seberapa baik model regresi linear memprediksi harga saham TLKM setelah ditambahkan fitur baru seperti lag (Adj Close_Lag1, Adj Close_Lag2, dan Adj Close_Lag3.) dan moving average (SMA_7, SMA_30).

```python
y = df['Adj Close']
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close_Lag1', 'Adj Close_Lag2', 'Adj Close_Lag3', 'SMA_7', 'SMA_30']]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.2, random_state=42)

model_new = LinearRegression()
model_new.fit(X_train_new, y_train_new)

y_pred_new = model_new.predict(X_test_new)

mse_new = mean_squared_error(y_test_new, y_pred_new)
rmse_new = np.sqrt(mse_new)
mae_new = mean_absolute_error(y_test_new, y_pred_new)
r2_new = r2_score(y_test_new, y_pred_new)

print("performa baru model dengan menggunakan fitur baru:")
print(f"Mean Squared Error (MSE): {mse_new:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_new:.2f}")
print(f"Mean Absolute Error (MAE): {mae_new:.2f}")
print(f"R-squared (R2) Score: {r2_new:.2f}")
```

<img width="317" height="79" alt="image" src="https://github.com/user-attachments/assets/227ae9f1-e8fd-46d5-bf90-8d56034932ea" />

## Kesimpulan

inti dari program ini adalah untuk memprediksi harga saham penutupan yang disesuaikan (Adjusted Close Price) dari saham PT Telekomunikasi Indonesia Tbk (TLKM.JK).

Fungsi Utama Program ini: Program ini menggunakan model Regresi Linear untuk menganalisis data historis harga saham (seperti harga pembukaan, tertinggi, terendah, penutupan, dan volume perdagangan) untuk memprediksi harga penutupan di masa depan.

Manfaat Program ini:

* Peningkatan Akurasi Prediksi: Dengan menambahkan fitur-fitur yang relevan dengan data deret waktu seperti nilai lagged (nilai historis 'Adj Close' sebelumnya) dan moving average (rata-rata bergerak), model menunjukkan peningkatan akurasi yang signifikan dalam memprediksi harga saham.
* Pemahaman Kinerja Model: Program ini tidak hanya memprediksi, tetapi juga mengevaluasi seberapa baik prediksinya melalui berbagai metrik (MSE, RMSE, MAE, R-squared) dan visualisasi (plot residu, plot perbandingan aktual vs. prediksi).
* Pengambilan Keputusan Lebih Baik: Bagi investor atau analis, prediksi harga saham dapat menjadi salah satu alat bantu untuk membuat keputusan yang lebih terinformasi, meskipun harus selalu diingat bahwa pasar saham sangat kompleks dan tidak ada model yang 100% akurat.
* Landasan untuk Analisis Lanjutan: Program ini juga menekankan pentingnya metode evaluasi yang tepat untuk data deret waktu (seperti Time Series Cross-Validation), yang merupakan praktik terbaik untuk membangun model prediksi saham yang lebih robust dan dapat diandalkan.


















