#Clothing Classifier Using Computer Vision with Convolutional Neural Network
#Computer Vision using Fashion MNIST Data
import tensorflow as tf
import matplotlib.pyplot as plt
#print(tf.__version__)

#CALLBACK FUNCTION
#set target to optimize processing time to avoid overfitting or overtraining
target_accuracy = 70 #in percent

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<(1.0-float(target_accuracy/100))):
      print("\nReached ", target_accuracy,"% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

#load Fashion MNIST from Keras datasets API
mnist = tf.keras.datasets.fashion_mnist

#load training data / test data and labels
(training_images, training_labels),(test_images, test_labels) = mnist.load_data()

#change index for different training image/label
training_index = 2
#change index for different test image/label
testing_index = 2

#number of neurons in middle layer
num_neurons = 512

#print to see what image is being displayed
#plt.imshow shows the image
#training label is the number associated with the classification (shoe = 9)
  #this is done to avoid bias
#training image is the matrix of pixels
plt.imshow(training_images[training_index])
print("Training Labels: ")
print(training_labels[training_index])
print("\nTraining Images: ")
print(training_images[training_index])

#reshape the data
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

#CNN: 2 CNNs; 1 convolution and 1 pooling layer per CNN
#64  convolutions
#3x3 grid for convolution size
#2x2 pooling to quarter the image sizes
#more convolutions and layers means more process time 
#but more accurate in the end; but careful not to over-fit the data

#DNN: 3 layers
#first layer is flattening layer to linearize matrix into linear array 
  #to flatten 28x28 matrix from picture; 1st layer should be same shape as data
#hidden middle layer has a number of neurons
  #can increase for greater accuracy but more processing time
#end layer must have 10 neurons because 10 classes of clothing

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

#parameter meanings:
#Sequential: defines a SEQUENCE of layers in the neural network
#Flatten: takes that square and turns it into a 1 dimensional set.
#Dense: Adds a layer of neurons  
#Relu: effectively means "If X>0 return X, else return 0" 
  #only passes values 0 or greater to the next layer in the network.
#Softmax: takes a set of values, and picks the biggest one 

#compile with optimizer and loss function
#train it by calling *model.fit*  to fit your training data to training labels
print("\n***Compiling Model***: \n")
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#TRAINING####################################
#can increase epochs for more training; increases run time; try not to overfit
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

#TESTING######################################
#call to evaluate the model based on test images
#model.summary to see the size and shape of the network
model.summary()
test_loss = model.evaluate(test_images, test_labels)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\n***Test Loss*** =", test_loss)
print("\n***Test Accuracy*** =", test_acc)

#DATA VISUALIZATION############################
#first 100 test labels
print("\n***Test Labels***: \n")
print(test_labels[:100])

f, axarr = plt.subplots(3,4)

#change the image numbers to change the data displayed
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26

#change the number of convolutions
#more convolutions, you can see the essential features the model recognizes
#for each fashion item
CONVOLUTION_NUMBER = 1

from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
  
