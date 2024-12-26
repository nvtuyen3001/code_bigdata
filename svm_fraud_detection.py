import numpy as np # linear algebra
import pandas as pd #import thư viện pandas để đọc file cơ sở dữ liệu
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC #thuat toan svm
from sklearn.metrics import classification_report # Báo cáo phân loại từ thuật
from sklearn.metrics import confusion_matrix # ma trận báo cáo thực tế và dự
from sklearn.metrics import f1_score # trung bình của recall_score va
from sklearn.metrics import accuracy_score # tỉ lệ dữ liệu dự đoán đúng trong
from sklearn.metrics import recall_score # tỉ lệ dự đoán mô hình đúng đúng
from sklearn.metrics import precision_score, precision_recall_curve  # tỉ lệ dự
# auc: tính diện tích dưới đường cong ROC
# average_precision_score: độ chính xác của dự đoán
from sklearn.metrics import roc_auc_score, roc_curve, auc,average_precision_score # độ chính xác của dự đoán
#StandardScaler: biến đổi dữ liệu của bạn sao cho phân phối của nó sẽ có giá

from sklearn.preprocessing import StandardScaler
# Dùng để chia mảng hoặc ma trận thành các tập con thử nghiệm và huấn
from sklearn.model_selection import train_test_split
# Dùng để vẽ đồ thị
from mlxtend.plotting import plot_confusion_matrix
# Link drive
from collections import Counter
from google.colab import drive
drive.mount('/gdrive')
# import modules để thao tác với tệp
import os
data = pd.read_csv("creditcard_2023.csv")
#Hiển thị sự không cân bằng của dữ liệu bằng biểu đồ tròn
count = pd.Series(data['Class']).value_counts().sort_index()
print(count)
count.plot(kind='pie')
plt.title('Unbalance Data')
plt.show()
No_of_frauds = len(data[data["Class"] == 1])
No_of_normals = len(data[data["Class"] == 0])
total = No_of_frauds + No_of_normals
Fraud_percent = (No_of_frauds / total)*100
Normal_percent = (No_of_normals / total)*100
print("So giao dich khong co su bat thuong (Class = 0) la: ", No_of_normals)
print("So giao dich co su bat thuong (Class = 1) la: ", No_of_frauds)
print("Ti le giao dich binh thuong (Class = 0) = ", Normal_percent)
print("Ti le giao dich bat thuong (Class = 1) = ", Fraud_percent)
cols = data.columns
att_cols = cols[1:-1]
lab_cols = cols[-1]
print(att_cols)
#Import the preprocessing module
from sklearn import preprocessing
data[att_cols] = preprocessing.MinMaxScaler((0, 1)).fit_transform(data[att_cols])
X = data[att_cols].values
y = data[lab_cols].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9,
shuffle=True)

# Create and train a model (example: Support Vector Classifier)
model = SVC() # create a SVC model
model.fit(X_train, y_train) # train the model on the training data

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Độ chính xác: {:.2f}%".format(accuracy * 100))
print("Nhớ lại: {:.2f}%".format(recall * 100))
print("Chính xác dự đoán: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
