# connect to google drive
from google.colab import drive
drive.mount('/content/gdrive/')

!unzip -q '/content/gdrive/My Drive/Colab Notebooks/vip2/NWPUvip.zip'
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img

train_dir = '/content/NWPU-RESISC12/train'
validation_dir = '/content/NWPU-RESISC12/test'
image_size = 224
nTrain = 6600
nVal = 1800

#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze all layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(12, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

# Specify the data augmentation techniques in the trainig image data generator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 50
val_batchsize = 10

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical',
        shuffle=True)

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the Model
# multiplied the steps_per_epoch by 2 for data augmentation.
history = model.fit_generator(
      train_generator,
      steps_per_epoch=2*train_generator.samples/train_generator.batch_size ,
      epochs=40,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

# Save the Model
model.save('/content/gdrive/My Drive/Colab Notebooks/vip2/VGG-FineTune-DA.h5')

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predictions = np.argmax(predictions,axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

# Evaluate model
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,f1_score
print(confusion_matrix(ground_truth, predictions, labels=None, sample_weight=None))
print(precision_score(ground_truth, predictions,average='macro'))
print(recall_score(ground_truth, predictions,average='macro'))
print(accuracy_score(ground_truth, predictions))
print(f1_score(ground_truth, predictions,average='macro'))