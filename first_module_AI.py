#import pandas
import pandas as pd
dataset = pd.read_csv("cancer.csv")

#print dataset and label columns
print(dataset.head())
x = dataset.drop(["diagnosis(1=m, 0=b)"], axis = 1)

#define y
y = dataset["diagnosis(1=m, 0=b)"]

#import sklearn and tester
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

#inport tensorflow
import tensorflow as tf

#define model
model = tf.keras.models.Sequential()


#add laters and declare activation function 
model.add(tf.keras.layers.Dense(256, input_shape = x_train.shape, activation = "sigmoid"))
model.add(tf.keras.layers.Dense(256, activation = "sigmoid"))
model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))

#compile model
model.compile(optimizer = "adam", loss="binary_crossentropy", metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 1000)
#evaluate model
model.evaluate(x_train, y_train)