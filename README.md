# GeneticWeightUploading

These are files to upload weights into a model, retrieved from a genetic algorithm.

The main file, and the one which should be run, is LoadWeights.py. test_directory should be the filepath to the folder of test images. The weights variable should hold the list of weights gotten from the genetic algorithm, or from an h5 file.

Depending on whether you want to evaluate one image or multiple, comment/uncomment lines 126 and 129 which load one image, or a directory of them, respectively.

results contains a numpy array of probabilities corresponding to each label, produced by the model.

To change the model shape, change the layers inside build_baseline_with_weights. The weights contained in weights must be the correct shape, required by the new model.

If using images other than MNIST, you will need to change the helper functions load_image and load_directory in Prediction.py. Each 28 must be changed to the dimension of the new image.

More detailed instructions regarding parameters and returns are commented in LoadWeights.py
