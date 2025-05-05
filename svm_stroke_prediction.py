 # 1. Import Library yang dibutuhkan
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 2. Load Dataset (upload file di Colab terlebih dahulu)
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 3. Encoding fitur kategorikal
le = LabelEncoder()
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df[col] = le.fit_transform(df[col])

# 4. Hapus baris dengan nilai kosong di kolom 'bmi'
df = df.dropna()

# 5. Pisahkan fitur dan label
X = df.drop(['id', 'stroke'], axis=1)  # kolom 'id' tidak relevan
y = df['stroke']

# 6. Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Menggunakan SMOTE untuk menangani data tidak seimbang
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 8. Model SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_res, y_train_res)

# 9. Prediksi dan evaluasi SVM
y_pred_svm = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("SVM Confusion Matrix:\n", cm_svm)
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# 10. Menampilkan Support Vectors
print("Support Vectors:", svm_model.support_)

# 11. Visualisasi Confusion Matrix untuk SVM
plt.figure(figsize=(7, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("svm_confusion_matrix.png")  # Menyimpan visualisasi sebagai .png file
plt.show()
