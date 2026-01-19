import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import os

# os.chdir("/home/anthbapt/Documents/fMNIST_DNN_training/wk")

# Dinamik yol - our_data_fmnist klasöründen veri oku
data_path = os.path.join(os.getcwd(), 'our_data_fmnist')
os.chdir(data_path)

# Çıktı klasörü oluştur (ana klasörde training_outputs)
output_dir = os.path.join(os.path.dirname(os.getcwd()), 'training_outputs')
os.makedirs(output_dir, exist_ok=True)

# Read data
x_test = pd.read_csv("fashion-mnist_test.csv")
y_test = x_test['label']
x_test = x_test.iloc[:, 1:]

x_train = pd.read_csv("fashion-mnist_train.csv")
y_train = x_train['label']
x_train = x_train.iloc[:, 1:]


# Restrict to labels 5 and 9
labels_1_7 = [5, 9]
train_1_7 = np.concatenate([np.where(y_train == label)[0] for label in labels_1_7])
test_1_7 = np.concatenate([np.where(y_test == label)[0] for label in labels_1_7])

y_train = y_train.iloc[train_1_7].values
y_test = y_test.iloc[test_1_7].values

y_test[y_test == 5] = 0
y_test[y_test == 9] = 1

y_train[y_train == 5] = 0
y_train[y_train == 9] = 1

x_train = np.array(x_train.iloc[train_1_7, :])
x_test = np.array(x_test.iloc[test_1_7, :])

# Print dimensions
print("Dimensions of x_train:", x_train.shape)
print("Dimensions of x_test:", x_test.shape)


b = 3  # Increased to 25 for robust statistics (paper likely used 20-50 models)
accuracy = list()
model_predict = np.empty(b, dtype = object)

for j in range(b):
    # Define DNN architecturex_test.shapex_test.shape
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_shape=(x_test.shape[1],)))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Binary cross-entropy loss function for all models
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # Train model on training data
    dnn_history = model.fit(x_train, y_train,
                            epochs=50, batch_size=32,
                            validation_split=0.2)

    # Check accuracy on test data
    acc = model.evaluate(x_test, y_test)[1]
    accuracy.append(acc)

    # Output the layers on implementation over test data
    # Get activations by manually passing data through each layer
    activations = []
    current_input = x_test
    # Extract only hidden layer outputs (exclude the final sigmoid output layer)
    for layer in model.layers[:-1]:  # Exclude last layer
        current_output = layer(current_input)
        activations.append(current_output.numpy())
        current_input = current_output
        
    
    model_predict[j] = activations


# np.save("model_predict.npy", model_predict)
# np.save("accuracy.npy", accuracy)
# pd.DataFrame(x_test).to_csv("x_test.csv", index=False, header = None)
# pd.DataFrame(y_test).to_csv("y_test.csv", index=False, header = None)

# Yukarıdaki kod çıktıyı direkt olarak ana dizine yazıyordu, bu yüzden aşağıdaki kodu kullanıyoruz


np.save(os.path.join(output_dir, "model_predict.npy"), model_predict)
np.save(os.path.join(output_dir, "accuracy.npy"), accuracy)
pd.DataFrame(x_test).to_csv(os.path.join(output_dir, "x_test.csv"), index=False, header = None)
pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header = None)