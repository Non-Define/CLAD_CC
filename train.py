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
import yaml
import json

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import IPython.display as ipd
import torchaudio.tracsforms
import matplotlib.pyplot as plt

# from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
# FAR, FRR, EER, F1_score 구현

from datautils import *
from moco_downstream import *
import aasist
from evaluate_tDCF_asvspoof19 import compute_eer
import warnings
warnings.filterwarnings('ignore')
#-------------------------------------------------------------------
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = 0  # GPU id to use
torch.cuda.set_device(gpu)
print(device)

# hyper parameters
batch = 24
epoch = 150
lr = 0.0005
#-------------------------------------------------------------------
with open("/home/hwang-gyuhan/Workspace/ND/config.conf", "r") as f_json:
    config = json.loads(f_json.read())

def load_model(model_name:str, config:dict):
    if model_name == "CLAD":
        with open(config['aasist_config_path'], "r") as f_json:        
            aasist_config = json.loads(f_json.read())
        aasist_model_config = aasist_config["model_config"]
        aasist_encoder = aasist.AasistEncoder(aasist_model_config).to(device)
        return aasist_encoder

def evaluation_19_LA_eval(model, score_save_path, model_name, database_path, augmentations=None, augmentations_on_cpu=None, batch_size = 1024, manipulation_on_real=True, cut_length = 64600):
    # In asvspoof dataset, label = 1 means bonafide.
    model.eval()
    device = "cuda"
    # load asvspoof 2019 LA train dataset
    d_label_trn, file_train, utt2spk = genSpoof_list(dir_meta=database_path+"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt", is_train=False, is_eval=False)
    print('no. of ASVspoof 2019 LA training trials', len(file_train))
    asvspoof_LA_eval_dataset = Dataset_ASVspoof2019_train(list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(
        database_path+'ASVspoof2019_LA_train/'), cut_length=cut_length, utt2spk=utt2spk)
    asvspoof_2019_LA_train_dataloader = DataLoader(asvspoof_LA_train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)  # added num_workders param to speed up.
    with open(score_save_path, 'w') as file:  # This creates an empty file or empties an existing file
        pass

def train_generator(train):
    train_datagen = ImageDataGenerator(rescale=1/255.)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=database_path+"ASVspoof2019_LA_train/flac/",
        x_col="song_id",
        y_col="genre_name",
        batch_size=batch_sz,
        target_size=(224,224),
        shuffle=True,
        class_mode='categorical',
        classes=name_class
    )
    return train_generator

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
K.clear_session()