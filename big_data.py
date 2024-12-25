import numpy as np # linear algebra
import pandas as pd #import thư viện pandas để đọc file cơ sở dữ liệu
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC #thuat toan svm
from sklearn.metrics import classification_report # Báo cáo phân loại từ thuật toán SVM
from sklearn.metrics import confusion_matrix # ma trận báo cáo thực tế và dự
from sklearn.metrics import f1_score # trung bình của recall_score va precision
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc,average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Sửa lỗi: chọn cột cuối cùng làm nhãn, chọn tất cả cột trừ cột đầu và cuối làm thuộc tính
att_cols = cols[1:-1]
lab_cols = cols[-1]

print(att_cols)

# Chuẩn hóa các thuộc tính sử dụng MinMaxScaler
data[att_cols] = preprocessing.MinMaxScaler((0, 1)).fit_transform(data[att_cols])

# Tách X là các thuộc tính, y là nhãn
X = data[att_cols].values
y = data[lab_cols].values

# Tách tập dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True)

# Khởi tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán kết quả trên tập test
y_pred = model.predict(X_test)

# Tính các chỉ số đánh giá
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# Hiển thị kết quả và đánh giá mô hình
print("Độ chính xác: {:.2f}%".format(accuracy * 100))
print("Nhớ lại: {:.2f}%".format(recall * 100))
print("Chính xác dự đoán: {:.2f}%".format(precision * 100))
cm = np.array(confusion_matrix(y_test, y_pred, labels=[1,0])) # Biến đổi ma trận thành mảng
# Hiển thị dữ liệu dưới dạng bảng  có thể thay đổi kích thước và không đồng
confusion = pd.DataFrame(cm, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
print(confusion)
print(classification_report(y_test, y_pred)) #hien thi bang bao cao do chinh xac
# Tính và hiển thị đồ thị Precision-Recall
average_precision = average_precision_score(y_test, y_pred)
precision, recall, curve = precision_recall_curve(y_test, y_pred)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Nhớ lại (Recall)')
plt.ylabel('Chính xác dự đoán (Precision)')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Đồ thị Precision-Recall: AP={:.2f}'.format(average_precision))
plt.show()
auc_pr=auc(recall,precision)
plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='blue', label=f'Precision-Recall Curve (AUC = {auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-RecallCurve')
plt.legend(loc='best')
plt.show()
seconds_in_a_day = 24 * 60 * 60
seconds_in_a_day
seconds_in_a_week = 7 * seconds_in_a_day
seconds_in_a_week