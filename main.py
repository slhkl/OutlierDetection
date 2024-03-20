import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def detect_outliers_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    labels = dbscan.labels_
    outliers_indices = np.where(labels == -1)[0]
    return outliers_indices

# Boxplot ile aykırı değerlerin tespit edilmesi
def detect_outliers_boxplot(X):
    Q1 = np.percentile(X, 25)
    Q3 = np.percentile(X, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_indices = np.where((X < lower_bound) | (X > upper_bound))[0]
    return outliers_indices

# Örnek veri kümesi oluşturalım
np.random.seed(0)
X = np.random.normal(loc=0, scale=1, size=100).reshape(-1, 1)

# Karışık veri seti oluşturma
X = np.concatenate((X, np.random.uniform(low=-10, high=10, size=(20, 1))))

# İşlem yapılmadan önce veri setinin özelliklerini hesaplayalım
mean_before = np.mean(X)
std_before = np.std(X)
median_before = np.median(X)
variance_before = np.var(X)

# DBSCAN kullanarak aykırı değerleri tespit edelim
outliers_dbscan = detect_outliers_dbscan(X, eps=0.5, min_samples=5)

# Boxplot kullanarak aykırı değerleri tespit edelim
outliers_boxplot = detect_outliers_boxplot(X)

# Boxplot çizimi
plt.figure(figsize=(6, 4))
plt.boxplot(X)
plt.title('Boxplot ile Aykırı Değerlerin Belirlenmesi')
plt.ylabel('Değerler')
plt.show()

# Aykırı değerleri baskılama yöntemi ile güncelleyelim
median_X = np.median(X)
X_clipped = X.copy()  # Orijinal veri setini kopyalayalım
X_clipped[outliers_dbscan] = median_X  # Aykırı değerleri median ile değiştirelim

# İşlem yapıldıktan sonra veri setinin özelliklerini hesaplayalım
mean_after = np.mean(X_clipped)
std_after = np.std(X_clipped)
median_after = np.median(X_clipped)
variance_after = np.var(X_clipped)

print("DBSCAN ile Tespit Edilen Aykırı Değerlerin İndeksleri:", outliers_dbscan)
print("BoxPlot ile Tespit Edilen Aykırı Değerlerin İndeksleri:", outliers_boxplot)

print("\n--- Önce ---")
print("Ortalama:", mean_before)
print("Standart Sapma:", std_before)
print("Medyan:", median_before)
print("Varyans:", variance_before)

print("\n--- Sonra ---")
print("Ortalama:", mean_after)
print("Standart Sapma:", std_after)
print("Medyan:", median_after)
print("Varyans:", variance_after)
