# GeneticWeightUploading

These are files to upload weights into a model, retrieved from a genetic algorithm.

The main file, and the one which should be run, is LoadWeights.py. test_directory should be the filepath to the folder of test images. The weights variable should hold the list of weights gotten from the genetic algorithm, or from an h5 file.

Depending on whether you want to evaluate one image or multiple, comment/uncomment lines 126 and 129 which load one image, or a directory of them, respectively.

results contains a numpy array of probabilities corresponding to each label, produced by the model.

To change the model shape, change the layers inside build_baseline_with_weights. The weights contained in weights must be the correct shape, required by the new model.

If using images other than MNIST, you will need to change the helper functions load_image and load_directory in Prediction.py. Each 28 must be changed to the dimension of the new image.

If you run into unexpected shape errors. There is most likely an issue with the shape of the weights. Check to make sure the list of weights is in the format described in either np_weights or build_baseline_with_weights. If the problem is still occuring, the weights probably don't fit the model's layers correctly. Run model.summary() for a detailed explanation of the shape of the outputs.

More detailed instructions regarding parameters and returns are commented in LoadWeights.py
