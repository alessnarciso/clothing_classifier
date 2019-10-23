#Clothing Classifier Using Computer Vision
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

#normalize the pixel values in the matrices to floating point for Flatten layer
training_images  = training_images / 255.0
test_images = test_images / 255.0

#3 layers
#first layer is flattening layer to linearize matrix into linear array 
  #to flatten 28x28 matrix from picture; 1st layer should be same shape as data
#hidden middle layer has a number of neurons
  #can increase for greater accuracy but more processing time
#end layer must have 10 neurons because 10 classes of clothing
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                           tf.keras.layers.Dense(num_neurons, activation=tf.nn.relu), 
                           tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

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
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
#TRAINING
#can increase epochs for more training; increases run time; try not to overfit
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

#TESTING
#call to evaluate the model based on test images
print("\n***Model Evaluation***: \n")
model.evaluate(test_images, test_labels)

#print classification label probability
#max value is the predicted label
classifications = model.predict(test_images)
print("\n***Model Classifications***: \n")
print(classifications[testing_index])

print("\n***Max Probability***: ", max(classifications[testing_index]))

#this should be the same index as the max of the printed array above
print("\n***Test Label***: ")
print(test_labels[testing_index])

#this is printed last as the image is printed last by default
print("\n***Training Image***: \n")
