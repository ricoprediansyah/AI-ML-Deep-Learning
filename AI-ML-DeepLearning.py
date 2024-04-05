# ML-7 Deep Learning
# Pada materi kali ini, kita akan membangun sebuah model Deep Learning sederhana menggunakan TensorFlow-Keras.

## 1. Gathering Data
# E:\Kuliah Semester 6\Program AI4JOB\ML-Deep Learning\image.png
# Kita akan menggunakan dataset sederhana untuk kasus klasifikasi, yakni dataset Iris.

# Dataset Iris memiliki 4 feature (variable), antara lain:

# Feature	Keterangan
# SepalLengthCm	  panjang sepal bunga Iris dalam ukuran cm
# SepalWidthCm	  lebar sepal bunga Iris dalam ukuran cm
# PetalLengthCm	  panjang kelopak bunga Iris dalam ukuran cm
# PetalWidthCm	  lebar kelopak

# Dataset ini juga memiliki sebuah label dengan 3 buah class (jenis bunga Iris), yakni:

# Iris Setosa
# Iris Versicolor
# Iris Virginica

from pandas import read_csv
url = 'https://raw.githubusercontent.com/achmatim/data-mining/main/Dataset/iris.csv'
df = read_csv(url)
df.head()

## 2. Preparing Data
# Langkah selanjutnya adalah mengolah dataset sehingga siap digunakan untuk mentraining Deep Learning.
# A. Pisahkan feature (X) dan label (y) dari dataset
X = df.values[:, :-1]
y = df.values[:, -1]
X[0:4]

X = X.astype('float32') #ubah x object menjadi float
X[0:4]
y

from sklearn.preprocessing import LabelEncoder #encode label nilai kategori mejadi nilai numerik
y = LabelEncoder().fit_transform(y)
y

# B. Split Data menjadi training dan testing dataset
from sklearn.model_selection import train_test_split
#split datast
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('ukuran X train:', X_train.shape)
print('ukuran y train:', y_train.shape)
print()
print('ukuran X test:', X_test.shape)
print('ukuran y test:', X_test.shape)

## 3. Deep Learning Model Life-Cycle (DLMLC)
# Setelah data siap digunakan untuk men-train model, langkah selanjutnya adalah membangun model Deep Learning.
# Deep Learning Model Life-Cycle (DLMLC) dapat kita gunakan sebagai pedoman dalam membangun model Deep Learning.
# E:\Kuliah Semester 6\Program AI4JOB\ML-Deep Learning\image2.png

# A. Define the Model
# Tahap pertama yang kita lakukan pada DLMLC adalah mendefinisikan model yang hendak dikembangkan. Kita harus menentukan arsitektur/topologi Deep Learning.
# Arsitektur/Topologi Deep Learning sangat bergantung pada dataset yang kita miliki. Mari kita ingat lagi bentuk dataset Iris.

# Dataset Iris memiliki 4 feature dan 3 class. Artinya, arsitektur Deep Learning yang kita bangun harus memiliki:

# Input layer dengan 4 neuron
# Output layer dengan 3 neuron
# Kita bebas menentukan jumlah hidden layer dan neuron dalam tiap hidden layer.

# Pada praktik kali ini, kita akan membangun Deep Learning dengan arsitektur/topologi sebagai berikut:
# E:\Kuliah Semester 6\Program AI4JOB\ML-Deep Learning\image3.png

# Arsitektur ini terdiri atas:
# Input layer dengan 4 neuron
# 2 Hidden layer, masing-masing hidden layer memiliki 3 neuron dan Activation Function ReLU
# Output layer dengan 3 neuron dengan Activation Function SoftMax
# Kita memakai Activation Function SoftMax pada output layer karena dataset kita memiliki 3 buah class (Multi-Class Classification).
# https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.tensorflow.org%2Fapi_docs%2Fpython%2Ftf%2Fkeras%2Factivations
#import library Tensorflow-Keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense
# Note: Code Sequential API 1, Sequential API 2, dan Functional API di bawah menghasilkan arsitektur Deep Learning yang sama persis. Cukup jalankan salah satu code.


# PILIH 1
# Define the Model using Sequential API 1
model = Sequential([
    Input(shape=(4,)),
    Dense(3, activation='relu'),
    Dense(3, activation='relu'),
    Dense(3, activation='softmax'),
], name='Sequential_API_1')

# model = Sequential(name=('Sequential_API_2'))
# model.add(Input(shape=(4,))),
# model.add(Dense(3, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# input_layer = Input(shape=(4,))
# hid_layer_1 = Dense(3, activation='relu')(input_layer)
# hid_layer_2 = Dense(3, activation='relu')(hid_layer_1)
# output_layer = Dense(3, activation='softmax')(hid_layer_2)
# model = Model(inputs=input_layer, outputs=output_layer, name='function_API')

# Pastikan arsitektur model yang dibangun sudah sesuai kebutuhan. Hal ini dapat dipastikan dengan memvisualisasikan model menggunakan Model Text Description dan Model Architecture Plot.

## Model Text Description
model.summary()

## Model Architecture Plot
from tensorflow.keras.utils import plot_model
plot_model(model, 'model.png', show_shapes=True)

# B. Compile the Model
# Pada tahap kedua DLMLC, dilakukan pemilihan loss function, optimizer, dan metrics untuk menilai performa model.
# Loss                               Function	Implementasi
# Mean Square Error (MSE)	           Regression
# Mean Absolute Error (MAE)          Regression
# Binary Cross Entropy               Binary Classification
# Categorical Cross Entropy          Multi-class Classification
# Sparse Categorical Cross Entropy   Multi-class Classification

# TensorFlow-Keras Loss Function

# Optimizer	    Keterangan
# SGD 	        Stochastic Gradient Descent
# Momentum	    SGD with Momentum
# RMSprop	      Root Mean Squared Propagation
# AdaDelta	    Adaptive Delta
# AdaGrad	      Adaptive Gradient Algorithm
# Adam	        Adaptive Moment Estimation

# TensorFlow Keras Optimizer

# Metrics
# Accuracy
# Precission
# Recall

# TensorFlow-Keras Metrics

# Kali ini kita memilih:

# . loss : Sparse Categorical Cross Entropy
# . optimizer : adam
# . metrics : Accuracy
# Sparse Categorical Cross Entropy cocok untuk Multi-Class Classification.

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(
    optimizer= Adam(learning_rate=0.001),
    loss = SparseCategoricalCrossentropy(),
    metrics = ['Accuracy']
)

# C. Fit the Model
# Tahap selanjutnya ialah melakukan training. Kita juga harus memilih konfigurasi training, seperti menentukan jumlah epoch dan batch size.
hist = model.fit(
    x=X_train,
    y= y_train,
    validation_data = (X_test, y_test),
    batch_size = 32,
    epochs = 200,
    verbose=2
)

# Plotting Learning Curves
# Performa model setelah proses training bisa kita visualisasikan menggunakan learning curves (LCs).
from matplotlib import pyplot
#plotaccuracy learning curves
pyplot.title('Learning curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Accuracy')
pyplot.plot(hist.history['Accuracy'], label='train')
pyplot.plot(hist.history['val_Accuracy'], label='val')
pyplot.legend()
pyplot.show()

#plot loss learning curves
pyplot.title('Learning curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
pyplot.plot(hist.history['loss'], label='train loss')
pyplot.plot(hist.history['val_loss'], label='val loss')
pyplot.legend()
pyplot.show()

# D. Evaluate the Model
# Tahap keempat DLMLC adalah mengevaluasi model menggunakan testing dataset.
# Evaluasi model dilakukan dengan cara memberikan testing dataset pada model untuk diprediksi.
# Hasil prediksi model selanjutnya akan dibandingkan dengan label/target yang diharapkan.
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print (f'Test Accuracy: {acc}')

# E. Make Prediction
# Tahap terakhir yang kita lakukan pada DLMLC adalah memanfaatkan model yang telah dibangun untuk memprediksi data baru.
# Data baru yang dimaksud bukan merupakan bagian dari training set ataupun testing set, melainkan data yang benar-benar “baru” atau data tanpa label.
from numpy import argmax

#input data baru
new_sepal_length = float(input('Input sepal length: '))
new_sepal_width = float(input('Input sepal width: '))
new_petal_length = float(input('Input petal length: '))
new_petal_width = float(input('Input petal width: '))

new_data = [new_sepal_length, new_sepal_width, new_petal_length, new_petal_width]

#prediksi data lalu cari kelasnya
y_pred = model.predict([new_data])
y_class = argmax(y_pred)

#cetak hasil prediksi
print (f'\nHasil Prediksi: {y_pred} (class=[y_class])\n')

if y_class == 0:
  print('Iris sentosa')
elif y_class ==1:
  print('iris versicolor')
elif y_class == 2:
  print('iris virginica')

#   Save Model
# Alasan kita menyimpan model:

# Model yang disimpan dalam file tidak akan hilang ketika program ditutup, karena file bersifat persistent.
# Model dapat di-training ulang di kemudian hari.
# Model dapat di-load/di-export/di-deploy ke berbagai platform (web, smartphone, embedded device) untuk memprediksi data baru.
model.save('model.h5')
# Note: Jangan lupa men-download file model yang sudah di-save! Model yang lupa di-download akan hilang saat Colab ditutup
# Load Model
from tensorflow.keras.models import load_model

#load model from file
model = load_model('model.h5')

#make prediction
new_data = [5.1, 3.5, 1.4, 0.2]
y_pred = model.predict([new_data])
print('\nPredicton:%s (class=%d)' % (y_pred, argmax(y_pred)))

# Deep Learning Techniques
# Coba tambahkan Dropout atau Batch Normalization dalam arsitektur Deep Learning yang sudah kita bangun sebelumnya!

# Dropout
# Dropout termasuk salah satu teknik yang bisa kita terapkan untuk mencegah overfitting.
# Dropout adalah mengabaikan neuron pada layer tertentu secara acak selama proses training berlangsung.
# “Mengabaikan” maksudnya ialah tidak mengikutsertakan neuron pada tahap forward pass dan backward pass.

from tensorflow.keras.layers import Dropout
model = Sequential(name='Dropout_Example')
model.add(Dense(100, input_shape=(10,)))
model.add(Dense(80))
model.add(Dropout(0.5)) # layer 2
model.add(Dense(30))
model.add(Dropout(0.4)) # layer 3
model.add(Dense(10))
model.add(Dropout(0.2)) # layer 4
model.add(Dense(5))
model.add(Dense(1))

# Batch Normalization
# Batch normalization juga termasuk salah satu teknik yang bisa kita terapkan untuk mencegah overfitting.
# Batch normalization adalah melakukan operasi standarisasi dan normalisasi input sebuah layer yang berasal dari layer sebelumnya.
from tensorflow.keras.layers import BatchNormalization
model = Sequential(name='Batch Normalization Example')
model.add(Dense(100, input_shape=(10,)))
model.add(BatchNormalization()) # layer 2
model.add(Dense(80))
model.add(BatchNormalization()) # layer 3
model.add(Dense(30))
model.add(BatchNormalization()) # layer 4
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))