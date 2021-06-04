# Repository for Cherenkov ring generator neural network prototypes

## iotools
- IO framework ported over from WatChMaL workshop code.

## Cris' playground
- CRinGe.py: initial attempt at a generator. Based on arXiv:1411.5928
- plotCRinGe.py: plot generated rings
- CRinGeView.py: bokeh application to interactively display generator output.
- CRinGe_FIP2.py: Unsuccessful attempt to include unsupervised generator outputs by running the following training sequence:
  1. initialize unsupervised inputs with random vector;
  2. run forward path and calculate loss;
  3. update unsupervised parameters using autograd;
  4. run forward path and calculate loss;
  5. update neural network parameters;
  - It might be worth revisiting this, paying more attention to the way the input parameters are updated...
- CRinGeGAN.py: implementation of a conditional generative adversarial network, inspired by arXiv:1605.05396, arXiv:1411.1784, ...
- plotTrainLog.py: very simple script to plot training progress using the standard stream outputs of the scripts above.

- CRinGe_MultiGaus.py/CRinGe_MultiLogNorm.py: multiple peaks with charge PDFs only.
- CRinGe_MultiGaus_Time.py/CRinGe_MultiLogNorm_Time.py: multiple peaks with charge PDFs and single peak for timing, charge and timing are independent.
- CRinGe_MultiGausTime_Corr.py: mutiple gaussian peaks for correlated charge and timing peaks.

## Usage
- To train generator run:
  `python -m CrisPlayground.CRinGe`