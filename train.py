import numpy as np
import pandas as pd
import cv2

from keras.layers import Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, Flatten
from keras.models import Model
from keras.applications.nasnet import NASNetMobile, preprocess_input

from keras.optimizers import Adam

#np.random.seed(0)


TRAIN_SIZE = 1000
MINI_BATCH_SIZE = 64
LEARNING_RATE = 0.00001
VALIDATION_SPLIT = 0.1
EPOCHS = 20
INPUT_SHAPE = (96, 96, 3)



def reshape_patch(patch,number):
    image_size = INPUT_SHAPE[0]
    if number < 6:
        return patch
    if number == 6:
        start_top = int(image_size*0.1)
        start_left = int(image_size*0.1)
    if number == 7:
        start_top = 0
        start_left = int(image_size*0.1)
    if number == 8:
        start_top = int(image_size * 0.1)
        start_left = 0
    if number == 9:
        start_top = 0
        start_left = 0

    patch = patch[start_left:image_size-start_left,start_top:image_size-start_top,:]
    patch = cv2.resize(patch,(INPUT_SHAPE[0], INPUT_SHAPE[1]))
    return patch

def load_patches(patch_list,path):
    batch_patches = []
    for file in patch_list:
        patch = path + file + ".tif"

        patch = cv2.imread(patch, cv2.IMREAD_COLOR)
        cv2.imshow("a",patch)
        patch = reshape_patch(patch,np.random.randint(1,10))
        cv2.imshow("b", patch)
        cv2.waitKey(0)
        patch = preprocess_input(patch)
        batch_patches.append(patch)


    return batch_patches


def trainBatch(model, patches_train, label_train, epoch, validation_split):
    patches_train = np.asarray(patches_train)
    label_train = np.asarray(label_train)

    model.fit(patches_train,
              label_train,
              initial_epoch=epoch,
              epochs=epoch + 1,
              batch_size=MINI_BATCH_SIZE,
              shuffle=True,
              validation_split=validation_split)

    return model


def train_one_epoch(model, epoch, patches_list,images_path, validation_split):
    batch_size = TRAIN_SIZE
    total_patches = len(patches_list)

    for start_batch_idx in np.arange(0, total_patches, batch_size):
        end_batch_idx = min(start_batch_idx + batch_size, total_patches)
        batch_label = patches_list.iloc[start_batch_idx : end_batch_idx]["label"]
        batch_patches = load_patches(patches_list["id"][start_batch_idx : end_batch_idx], path=images_path)
        model = trainBatch(model=model, patches_train=batch_patches, label_train=batch_label, epoch=epoch, validation_split=validation_split)

    return model


def define_model():
    inputs = Input(INPUT_SHAPE)
    base_model = NASNetMobile(weights =None,include_top=False, input_shape=INPUT_SHAPE)
    # for layer in base_model.layers:
    #     layer.trainable=False
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    return model


def train_model(model, patches_list, images_path):
    for epoch in range(EPOCHS):
        print('Epoch #{} (of {}):'.format(epoch + 1, VALIDATION_SPLIT))
        model = train_one_epoch(model=model,epoch=epoch, patches_list=patches_list, images_path=images_path, validation_split=VALIDATION_SPLIT)




images_path = r"C:\Users\yoav.v\Desktop\kaggle\train\\"
#model = define_model()
#model.load_weights(r"C:\Users\yoav.v\Desktop\kaggle\model.h5")
patches_list = pd.read_csv(r"C:\Users\yoav.v\Desktop\kaggle\train_labels.csv")
print(patches_list["id"])
load_patches(patches_list["id"], images_path)
exit()
train_model(model=model, patches_list=patches_list, images_path= images_path)
