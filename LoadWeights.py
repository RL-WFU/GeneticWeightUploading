import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from Prediction import *



# Save the weights of a model into an array
def array_model_weights(saved_weights):
    # Load model and model size
    model = load_model("%s" % saved_weights)
    num_layers = len(model.layers)

    array_weights = [list] * num_layers

    # For every layer, append the layers weights to array_weights
    for i in range(num_layers):
        # Set next layer
        curr_layer = model.layers[i]

        # Set weights
        # Observe that get_weights() returns weights in [0] and biases in [1]
        layer_weights = curr_layer.get_weights()
        array_weights[i] = layer_weights

    return array_weights



def np_weights(saved_weights, h5):
    """
    This function prepares your weights so that they can be fed into the kernel and bias initializers
    :param saved_weights: the weight data. If h5 file, calls array_model_weights to turn into array form. The first axis
    of this array should represent the number of layers in the model. This can be either with the empty layers (like
    pooling or flatten) or without them. If there are empty layers, this function will remove those from the returned list.
    It turns the weights into numpy arrays to make them compatible with keras initializers
    :param h5: Boolean - whether the weights are in an h5 file, or already in array form
    :return: formatted weights. First axis corresponds to layer index out of the layers WITH weights (flatten/pooling not included)
    Second axis corresponds to either weights or biases. Ex: layer_info[2][0] is the weights of the third valid layer
    """
    if h5:
        array_weights = array_model_weights(saved_weights) #Turn weights into a list if not already

    layer_info = []
    for i in range(len(array_weights)): #Iterate through the layers
        layer = np.asarray(array_weights[i], dtype=object) #Weights and biases of ith layer
        if len(layer) != 0: #If the layer is not empty (if it is not pooling or flatten)
            layer_weights = np.asarray(layer[0]) #weights
            layer_biases = np.asarray(layer[1]) #biases
            layer_info.append((layer_weights, layer_biases))
        else:
            continue


    return layer_info



class initialize_weights(keras.initializers.Initializer):
    def __init__(self, weights):
        """
        Initializer subclassing. This is how we can pass our own weights as kernel and bias initializers
        :param weights: This is either an array of kernel weights or an array of bias weights. Gotten from layer_info
        It must be a numpy array. If it is not, you can pass your weights through np_weights and it will do that
        """
        super(initialize_weights, self).__init__()
        self.myWeights = weights
    def __call__(self, shape, dtype=None):
        """
        This is automatically called when defining kernel_initializer or bias_initializer. Keras backend takes care
        of this and it should never be manually called
        :param shape: shape of the weights. Passed by keras
        :param dtype: dtype of weights. Passed by keras
        :return:
        """
        return self.myWeights



def build_baseline_with_weights(layer_info):
    """
    :param layer_info: a list of weights, with at least two axes. The first axis specifies the layer index, not
    including layers with no weights, like pooling or flatten layers. The second axis denotes either weights or biases,
    index 0 being weights, and index 1 being biases. So layer_info[0][0] denotes the weights of the first layer
    If weights are not in the above format, run them through the np_weights function

    :return: model: a keras model object which holds the model layers and parameters. Calling .predict_on_batch on this
    model object will return an array of shape (1, 10), containing the probabilities of each label
    """
    model = Sequential()

    # CONSTRUCT LAYERS

    #kernel_initializer and bias_initializer are replaced with Initializer subclassing (talked about above).
    #Pass as argument to initialize_weights the corresponding weights or biases of the layer. In this case, it is
    #the first layer's weights and biases, so layer_info[0][0] and layer_info[0][1] are passed, respectively
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=initialize_weights(layer_info[0][0]), bias_initializer=initialize_weights(layer_info[0][1]), input_shape=(28, 28, 1)))
    # Max pooling layer with 2x2 filter
    #Max pooling and flatten do not have weights
    model.add(MaxPooling2D((2, 2)))
    # Flatten filter maps to pass to classifier
    model.add(Flatten())
    # Fully-connected layer with 100 nodes, ReLU activation function
    #Second layer that needs weights from layer_info
    model.add(Dense(100, activation='relu', kernel_initializer=initialize_weights(layer_info[1][0]), bias_initializer=initialize_weights(layer_info[1][1])))
    # Fully-connected output layer with 10 nodes (for the labels [0,9])
    #Third, and last layer with initialized weights
    model.add(Dense(10, activation='softmax', kernel_initializer=initialize_weights(layer_info[2][0]), bias_initializer=initialize_weights(layer_info[2][1])))

    return model


def black_box_sub_model(layer_info):
    model = Sequential()

    # CONSTRUCT LAYERS

    """
    This substitute model for JSMA black box attacks does not have a convolutional layer, only two hidden layers
    of 200 output units, and an output layer of 10 units
    """
    # Flatten filter maps to pass to classifier
    model.add(Flatten())

    model.add(Dense(200, activation='relu', kernel_initializer=initialize_weights(layer_info[0][0]),
                    bias_initializer=initialize_weights(layer_info[0][1])))
    model.add(Dense(200, activation='relu', kernel_initializer=initialize_weights(layer_info[1][0]),
                    bias_initializer=initialize_weights(layer_info[1][1])))

    """
    NOTE: This output layer does not output probabilities, it outputs LOGITS. Call tf.nn.softmax on the output,
    to turn it into probabilities
    """
    model.add(Dense(10, activation='linear', kernel_initializer=initialize_weights(layer_info[2][0]),
                    bias_initializer=initialize_weights(layer_info[2][1])))

    return model




def prepare_model(fpath, sub=False):
    """
    For the substitute model, I have the weights already in array format, in a .npy file (produced by numpy.save)
    """
    if sub:
        weights = np.load('weights_arr.npy', allow_pickle=True)
        weights = np.reshape(weights, [3, 2])
        model = black_box_sub_model(weights)

    else:
        layer_information = np_weights(fpath, h5=True)
        model = build_baseline_with_weights(layer_information)

    return model






if __name__ == "__main__":
    #Directory where test images are
    test_directory = "PredictionImages/Raiford_Handwritten/"

    path_to_weights = "sub_weights_array.txt"

    model = prepare_model(path_to_weights, sub=True)


    #Preprocess the image. Replace "0a.png" with test image
    img = load_image(test_directory + "0a.png")

    #Use this if you wish to evaluate multiple images
    #imgs = load_directory(test_directory)


    #This is the feed forward part. Results is an MxN array, where M is the amount of images and N is the number of labels
    #Either feed in img if evaluating one image, or imgs if evaluating multiple
    results = np.squeeze(model.predict_on_batch(img))
    print(results)


