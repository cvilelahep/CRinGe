# Repository for Cherenkov ring generator neural network prototypes

## apps
- train_model.py
  - Trains neural network model.
```
usage: train_model.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-j NUM_WORKERS]
                      [-t TRAIN_FRACTION] [-s SAVE_INTERVAL] [-o OUTPUT_DIR]
                      [-r RANDOM_SEED]
                      data_dirs data_flavour model
                      [model_arguments [model_arguments ...]]
```
  - As an example, the following command will train the CRinGe_SK_MultiGaus network for 5 epochs, using a batch size of 150, 4 IO workers, using 75% of the dataset, saving the neural network output every 1000 iterations to output directory ~/NNOutput/, and using a mixture of 6 Gaussians. All files in /path/to/data ending in .h5 are used for training.
    - `python train_model.py -e 5 -b 150 -j 4 -t 0.75 -s 1000 -o ~/NNOutput/ /path/to/data .h5 CRinGe_SK_MultiGaus N_GAUS:6`
  - Note that only the location of the training data, the data filename ending and the model are required arguments. All other arguments will take reasonable values if left unspecified.


## models
- Neural network models:
  - CRinGe_SK_MultiGaus.py
    - Binary cross-entropy "hit" vs "unhit" loss
    - Gaussian mixture loss using hit PMT charge and time, optionally.
    - Super-K geometry and samples produced with WCSim.
  - CRinGe_Gaus.py
    - Binary cross-entropy "hit" vs "unhit" loss
    - Mean squared-error loss using hit PMT charge.
    - IWCD geometry and WatChMaL samples.

## iotools
- IO framework ported over from WatChMaL workshop code.
