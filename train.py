'''
The code was referenced from the 
"Music Genre Classification Using DenseNet and Data Augmentation"
'''

# import libraries

import numpy as np 
import pandas as pd 
import glob
import os
import datetime
import csv, ast
import pickle
import matplotlib.pyplot as plt
from datautils import *
from model import 
import aasist
from evaluate_tDCF_asvspoof19 import compute_eer
import warnings
warnings.filterwarnings('ignore')
#-------------------------------------------------------------------

number_class = 8 
batch_sz = 32
nepoch = 150
add_info = "Noise_Echo_FMA"
lr_min=0.00001
str_lr_min = "1e-5"

# Path to the *.CSV file  list of files for training and validation
train_data = pd.read_csv("/home/hwang-gyuhan/Workspace/DenseNet/ForTraining&Testing/train_valid_data.csv")

# Verify if CSV file is qualified
for i in range(train_data.genre_name.shape[0]):
    train_data.genre_name.loc[i] = ast.literal_eval(train_data.genre_name[i])



# Path to directory containing image files *.png
train_img_path = "/home/hwang-gyuhan/Workspace/dataset/FMA_IMAGES/"

# 8 music genres of FMA
name_class = ["Electronic",
            "Experimental",
            "Folk",
            "Hip_Hop",
            "Instrumental",
            "International",
            "Pop",
            "Rock"
    ]
def data_generator(train, val):
    train_datagen = ImageDataGenerator(rescale=1/255.)
    val_datagen = ImageDataGenerator(rescale=1/255.)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=train_img_path,
        x_col="song_id",
        y_col="genre_name",
        batch_size=batch_sz,
        target_size=(224,224),
        shuffle=True,
        class_mode='categorical',
        classes=name_class
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val,
        directory=train_img_path,
        x_col="song_id",
        y_col="genre_name",
        batch_size=batch_sz,
        target_size=(224,224),
        shuffle=False,
        class_mode='categorical',
        classes=name_class
    )   
    return train_generator, val_generator


class SaveEvery10Epochs(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 10번째 에포크마다 모델 저장
        if (epoch + 1) % 10 == 0:  
            model_name = f"./weights/epoch_{epoch+1:03d}_weights.h5"
            self.model.save_weights(model_name)
            print(f"Saved model at {model_name}")
            
def train_model(model, train_gen, val_gen, train_steps, val_steps, epochs):
    callbacks = [
        EarlyStopping(patience=50, verbose=1, mode='auto'),
        ReduceLROnPlateau(factor=0.1, patience=2, min_lr=lr_min, verbose=1, mode='auto'),
        SaveEvery10Epochs() # Save every 10 epochs
    ]

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                metrics = ['binary_accuracy'])

    report_dir = "./report/"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    working_dir = "./figures/"
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    history_dir = "./history"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    weights_dir = "./weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    from contextlib import redirect_stdout
    # Store model architecture in text file, it is a very long file for DenseNet
    with open('./modelNewDenseNet121.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model.summary()
    file_txt = working_dir + "time121.txt"
    x1 = datetime.datetime.now()
    time_now_begin = x1.strftime("%d%b%y%a") + "_" + x1.strftime("%I%p%M") + "\n"
    H = model.fit(train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=val_steps,
            verbose=1)

    with open('./history/history.pickle', 'wb') as file_pi:
        pickle.dump(H.history, file_pi)

    # visualizing losses and accuracy
    train_loss = H.history['loss']
    train_acc = H.history['binary_accuracy']
    val_loss = H.history['val_loss']
    val_acc = H.history['val_binary_accuracy']
    xc = range(len(train_loss))
    figures_dir = working_dir
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5)
    plt.subplot(211)
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "val_loss"], loc="upper right")
    plt.subplot(212)

    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train_acc", "val_acc"], loc="lower right")
    max_val_acc = max(val_acc)
    plt.title("Max Validation Accuracy = " + str(max_val_acc))
    x = datetime.datetime.now()
    time_now = x.strftime("%d%b%y%a") + "_" + x.strftime("%I%p%M")
    fig_filename1 = figures_dir + "LossAcc"+"-"+str(batch_sz)+"_"+str_lr_min+"_"+add_info+"_"+time_now
    fig.savefig(fig_filename1)

# Data split: No K-Fold, just a single train/test split
train, val = train_test_split(train_data, test_size=0.2, random_state=42)

train_generator, val_generator = data_generator(train, val)

# Using DenseNet
base_model_densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
headModel = base_model_densenet.output
headModel = Dropout(0.5)(headModel)
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Flatten()(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Dense(number_class, activation='sigmoid')(headModel)
model_dense_121 = Model(inputs=base_model_densenet.input, outputs=headModel)

train_steps = int(len(train)/batch_sz)
val_steps = int(len(val)/batch_sz)

train_model(model_dense_121, train_generator, val_generator, train_steps, val_steps, nepoch)

# Model evaluation after training
y_true = []
y_pred = []

# Collect predictions for the validation set
for i in range(len(val_generator)):
    x_batch, y_batch = val_generator[i]
    predictions = model_dense_121.predict(x_batch)
    y_true.extend(np.argmax(y_batch, axis=1))  # Convert one-hot encoded labels to class indices
    y_pred.extend(np.argmax(predictions, axis=1))  # Convert probabilities to predicted class indices

# Get the classification report
report = classification_report(y_true, y_pred, target_names=name_class)

# Print the classification report to the console
print("Classification Report")
print(report)

# Save the classification report every 10 epochs
epoch_interval = 10
for epoch in range(1, nepoch + 1):
    if epoch % epoch_interval == 0:
        report_filename = f'./report/classification_report_epoch_{epoch}.txt'
        
        # Save classification report to a text file with epoch number in the file name
        with open(report_filename, 'w') as f:
            f.write(report)

# Clear session for the next run
del model_dense_121
K.c'''
The code was referenced from the 
"Music Genre Classification Using DenseNet and Data Augmentation"
'''

# import libraries

import numpy as np 
import pandas as pd 
import glob
import os
import datetime
import csv, ast
import pickle
import matplotlib.pyplot as plt
from datautils import *
from model import 
import aasist
from evaluate_tDCF_asvspoof19 import compute_eer
import warnings
warnings.filterwarnings('ignore')
#-------------------------------------------------------------------

number_class = 8 
batch_sz = 32
nepoch = 150
add_info = "Noise_Echo_FMA"
lr_min=0.00001
str_lr_min = "1e-5"

# Path to the *.CSV file 
train_data = pd.read_csv("/home/hwang-gyuhan/Workspace/DenseNet/ForTraining&Testing/train_valid_data.csv")

# Verify if CSV file is qualified
for i in range(train_data.genre_name.shape[0]):
    train_data.genre_name.loc[i] = ast.literal_eval(train_data.genre_name[i])



# Path to directory containing image files *.png
train_img_path = "/home/hwang-gyuhan/Workspace/dataset/FMA_IMAGES/"

# 8 music genres of FMA
name_class = ["Electronic",
            "Experimental",
            "Folk",
            "Hip_Hop",
            "Instrumental",
            "International",
            "Pop",
            "Rock"
    ]
def data_generator(train, val):
    train_datagen = ImageDataGenerator(rescale=1/255.)
    val_datagen = ImageDataGenerator(rescale=1/255.)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=train_img_path,
        x_col="song_id",
        y_col="genre_name",
        batch_size=batch_sz,
        target_size=(224,224),
        shuffle=True,
        class_mode='categorical',
        classes=name_class
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val,
        directory=train_img_path,
        x_col="song_id",
        y_col="genre_name",
        batch_size=batch_sz,
        target_size=(224,224),
        shuffle=False,
        class_mode='categorical',
        classes=name_class
    )   
    return train_generator, val_generator


class SaveEvery10Epochs(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 10번째 에포크마다 모델 저장
        if (epoch + 1) % 10 == 0:  
            model_name = f"./weights/epoch_{epoch+1:03d}_weights.h5"
            self.model.save_weights(model_name)
            print(f"Saved model at {model_name}")
            
def train_model(model, train_gen, val_gen, train_steps, val_steps, epochs):
    callbacks = [
        EarlyStopping(patience=50, verbose=1, mode='auto'),
        ReduceLROnPlateau(factor=0.1, patience=2, min_lr=lr_min, verbose=1, mode='auto'),
        SaveEvery10Epochs() # Save every 10 epochs
    ]

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                metrics = ['binary_accuracy'])

    report_dir = "./report/"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    working_dir = "./figures/"
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    history_dir = "./history"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    weights_dir = "./weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    from contextlib import redirect_stdout
    # Store model architecture in text file, it is a very long file for DenseNet
    with open('./modelNewDenseNet121.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model.summary()
    file_txt = working_dir + "time121.txt"
    x1 = datetime.datetime.now()
    time_now_begin = x1.strftime("%d%b%y%a") + "_" + x1.strftime("%I%p%M") + "\n"
    H = model.fit(train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=val_steps,
            verbose=1)

    with open('./history/history.pickle', 'wb') as file_pi:
        pickle.dump(H.history, file_pi)

    # visualizing losses and accuracy
    train_loss = H.history['loss']
    train_acc = H.history['binary_accuracy']
    val_loss = H.history['val_loss']
    val_acc = H.history['val_binary_accuracy']
    xc = range(len(train_loss))
    figures_dir = working_dir
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5)
    plt.subplot(211)
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "val_loss"], loc="upper right")
    plt.subplot(212)

    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train_acc", "val_acc"], loc="lower right")
    max_val_acc = max(val_acc)
    plt.title("Max Validation Accuracy = " + str(max_val_acc))
    x = datetime.datetime.now()
    time_now = x.strftime("%d%b%y%a") + "_" + x.strftime("%I%p%M")
    fig_filename1 = figures_dir + "LossAcc"+"-"+str(batch_sz)+"_"+str_lr_min+"_"+add_info+"_"+time_now
    fig.savefig(fig_filename1)

# Data split: No K-Fold, just a single train/test split
train, val = train_test_split(train_data, test_size=0.2, random_state=42)

train_generator, val_generator = data_generator(train, val)

# Using DenseNet
base_model_densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
headModel = base_model_densenet.output
headModel = Dropout(0.5)(headModel)
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Flatten()(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Dense(number_class, activation='sigmoid')(headModel)
model_dense_121 = Model(inputs=base_model_densenet.input, outputs=headModel)

train_steps = int(len(train)/batch_sz)
val_steps = int(len(val)/batch_sz)

train_model(model_dense_121, train_generator, val_generator, train_steps, val_steps, nepoch)

# Model evaluation after training
y_true = []
y_pred = []

# Collect predictions for the validation set
for i in range(len(val_generator)):
    x_batch, y_batch = val_generator[i]
    predictions = model_dense_121.predict(x_batch)
    y_true.extend(np.argmax(y_batch, axis=1))  # Convert one-hot encoded labels to class indices
    y_pred.extend(np.argmax(predictions, axis=1))  # Convert probabilities to predicted class indices

# Get the classification report
report = classification_report(y_true, y_pred, target_names=name_class)

# Print the classification report to the console
print("Classification Report")
print(report)

# Save the classification report every 10 epochs
epoch_interval = 10
for epoch in range(1, nepoch + 1):
    if epoch % epoch_interval == 0:
        report_filename = f'./report/classification_report_epoch_{epoch}.txt'
        
        # Save classification report to a text file with epoch number in the file name
        with open(report_filename, 'w') as f:
            f.write(report)

# Clear session for the next run
del model_dense_121
K.c