import os 
import cv2
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical 
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications import ResNet152V2, Xception, ResNet101V2, ResNet50V2

K.set_image_data_format('channels_last')
BATCH = 15
EPOCH = 2
img_size_w = 256
img_size_h = 256
CHANNEL = 3


# Verilen path içerisindeki dosyaların isimlerini döner.
def fileNameReturner(path):
    return os.listdir(path)


# Değişken olarak görüntülerin pathlerini ve görüntülerin isimlerini alır.
# Path ile isimlerin birleştirilmiş hallerini döner.
def filePathMaker(path,image_names):
    file_paths = []
    for i in image_names:
        file_paths.append(path + "//" + i)
    return file_paths


# Değişken olarak görüntülerin tam yollarını ve hangi sınıfa ait olduklarının bilgisini veren sayı değeri alırlar
# İçerisinde her görüntüye ait path ve sınıfın bulunduğu dataFrame döner
def dataFrameMaker(image_paths,class_number):
    dataFrame = pd.DataFrame(image_paths, columns = ["path"])
    dataFrame["class"] = int(class_number)
    return dataFrame


# fileNameReturner, filePathMaker, dataFrameMaker fonksiyonlarının tek bir fonksiyonda koşturulması
def dataFrameMakerAllInOne(path, class_number):
    image_names = fileNameReturner(path)
    image_paths = filePathMaker(path, image_names)
    df = dataFrameMaker(image_paths, class_number)
    return df


# Birçok sınıfa ait görüntünün dataFrame'lerinin birleştirilmesi (Maksimum 5 df)
# Dönüş değeri olarak label olarak kullanılacak class değerleri ve class değerlerinden ayrılmış sadece pathlerin olduğu df
def dfAppendMx5(count_of_df, df1 = None, df2 = None, df3 = None, df4 = None, df5 = None):
    data = df1.append(df2, ignore_index=True)
    data = data.append(df3, ignore_index=True)
    data = data.append(df4, ignore_index=True)
    data = data.append(df5, ignore_index=True)
    
    classes = data["class"]
    classes = to_categorical(classes,num_classes = count_of_df)
    
    data = data.drop(["class"], axis=1)

    return data,classes


#Görüntü formatını 0 ise RGB 1 ise GRAY formata dönüştürür.
def convertImage(img):
    if CHANNEL == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    return img


# Fonksiyona verilen görüntüyü düzeltip, kırpıp tekrardan resize eden fonksiyon
def croppingImage(img):
    thresh = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    thresh_rotated = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
    thresh_rotated = cv2.threshold(thresh_rotated, 100, 255, cv2.THRESH_BINARY)[1]


    cnts,_ = cv2.findContours(thresh_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea)

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = rotated[y:y+h, x:x+w]
    crop_son=cv2.resize(ROI, (img_size_w,img_size_h))
    
    return crop_son


# Yolları verilen görüntülerin okunması, isteğe göre görüntü formatlarının değişmesi, kırpılması, resize edilmesi
def imageRead(images_path, is_crop = False):
    images = []
    for path in images_path["path"]:
        img = cv2.imread(path)
        img = convertImage(img)
        img = cv2.resize(img, (img_size_w,img_size_h))
        
        if is_crop != False:
            img = croppingImage(img)
        
        images.append(img)
        
    images = np.array(images,dtype="float32")
    images = images/255
    
    return images


# Görüntü üretme
def imageGenerator():
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range = 360)
    return datagen


# Seçilen modeli dönen fonksiyon
def appPicker(app = 0, is_weights = False ):
    
    if app == 0 and is_weights == False:
        arc = ResNet101V2(weights = None ,include_top=False, input_shape=(img_size_w, img_size_h, CHANNEL))
        
    elif app == 0 and is_weights == True:
        arc = ResNet101V2(weights = "imagenet" ,include_top=False, input_shape=(img_size_w, img_size_h, CHANNEL))
        
    elif app == 1 and is_weights == False:
        arc = ResNet50V2(weights = None ,include_top=False, input_shape=(img_size_w, img_size_h, CHANNEL))
        
    elif app == 1 and is_weights == True:
        arc = ResNet50V2(weights = "imagenet" ,include_top=False, input_shape=(img_size_w, img_size_h, CHANNEL))

    elif app == 2 and is_weights == False:
        arc = ResNet152V2(weights = None ,include_top=False, input_shape=(img_size_w, img_size_h, CHANNEL))
        
    elif app == 2 and is_weights == True:
        arc = ResNet152V2(weights = "imagenet" ,include_top=False, input_shape=(img_size_w, img_size_h, CHANNEL))
        
    elif app == 3 and is_weights == False:
        arc = Xception(weights = None ,include_top=False, input_shape=(img_size_w, img_size_h, CHANNEL))
        
    elif app == 3 and is_weights == True:
        arc = Xception(weights = "imagenet" ,include_top=False, input_shape=(img_size_w, img_size_h, CHANNEL))
        
    return arc


# appPicker fonksiyonundan dönen mimarinin modele dönüştürülmesi
def modelCreate(class_size, app = 0, is_weights = False):
    
    arc = appPicker(app = app, is_weights = is_weights)
    model = Sequential()
    model.add(arc)
    model.add(Flatten())
    
    if class_size == 2:
        model.add(Dense(2, activation = "sigmoid"))
        model.compile(loss='binary_crossentropy',optimizer = SGD(lr=0.01, momentum=0.9),metrics=['accuracy'])
    else:
        model.add(Dense(class_size, activation = "softmax"))
        model.compile(loss='categorical_crossentropy',optimizer = SGD(lr=0.01, momentum=0.9),metrics=['accuracy'])
    
    return model
    

# Modelin kaydedilme durumlarının belirlendiği fonksiyon
def checkPointer(model_save_path):
    checkopointer = tf.keras.callbacks.ModelCheckpoint(
        filepath = model_save_path,
        monitor = "val_loss",
        verbose = 1,
        save_best_only = True,
        save_freq = "epoch",
        mode = "min"
        )
    return checkopointer

# Gerekli değerlerin fonksiyona verilerek eğitimin gerçekleştirilmesi ve sonuçların görselleştirilmesi
def startTraining(images, classes, model_save_path, class_size, app = 0, is_weights = False):
    model = modelCreate(class_size, app = 0, is_weights = False)
    datagen = imageGenerator()
    checkpointer = checkPointer(model_save_path)
    x_train,x_val,y_train,y_val = train_test_split(images,classes,random_state=42,test_size=0.2)

    
    history = model.fit(datagen.flow(x_train, y_train, 
                               batch_size = BATCH),
                               steps_per_epoch = len(x_train) / BATCH, 
                               epochs = EPOCH,
                               validation_data = (x_val, y_val),
                               callbacks = [checkpointer], 
                               shuffle = True)
    model.save(filepath = model_save_path + ".h5")
    plottingHistory(history)

# History değişkeni içerisindeki değerlerin görselleştirilmesi
def plottingHistory(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
def testingModel(test_image_ways, test_classes, model_way):
    test_images = imageRead(test_image_ways, is_crop = True)
    loaded_model = tf.keras.models.load_model(model_way)
    confMatrix(test_images, test_classes, loaded_model)

    
def confMatrix(test_images, test_classes, loaded_model):
    Y_pred = loaded_model.predict(test_images)
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    Y_true = np.argmax(test_classes,axis = 1) 
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    f,ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    
#%%
class_size = 2
model_save_path = "model//model"

# TRAIN Veri Yolların belirlenmesi
path_kre1 = "kre1"   
path_kre2 = "kre2"

# TRAIN Dataframe'lerin oluşturulması ve birleştirilmesi
kre1_df = dataFrameMakerAllInOne(path_kre1, 0)
kre2_df = dataFrameMakerAllInOne(path_kre2, 1)
data,classes = dfAppendMx5(class_size, kre1_df, kre2_df)

# Görüntülerin okunup ön işlemlerin gerçekleştirilmesi
images = imageRead(data, is_crop = True)

#Eğitimin başlatılması
startTraining(images, classes, model_save_path, class_size, app = 0, is_weights = False)

# TEST Veri Yolların belirlenmesi
test_path_kre1 = "test_kre1"   
test_path_kre2 = "test_kre2"

# TEST Dataframe'lerin oluşturulması ve birleştirilmesi
test_kre1_df = dataFrameMakerAllInOne(path_kre1, 0)
test_kre2_df = dataFrameMakerAllInOne(path_kre2, 1)
test_data,test_classes = dfAppendMx5(class_size, kre1_df, kre2_df)

testingModel(test_data, test_classes, model_save_path + ".h5")
