# Jet prediction

LHC sends paquets of proton 40 milions times by seconds. Each collision produce many particles measured by sensors.But it's too many data to save : the majority of data is lost. So the idea is to construct a system who choses which collisions to keep in real time.This decision need to be done in microseconds.
When particles are produced with a collision, there are not observed direclty, there are fragmented in packs of particles called jets. By studying the form and the structure of this jet, we can deduce which type of particle it is.


## 1. The objective

The objective is to predict which type of particle is produced by the simultaed jet from our dataset. The idea is to keep in mind the importance of latence in our model construction.

## 2. The dataset

We we will use the dataset hls4ml composed of simulated jets dispattched between 5 classes depending on their particle origin : gluon (g), quark léger (q), boson W (w), boson Z (z) et quark top (t). We have 830 000 data with each jet described by 16 features that sums up their form : mass,intern structure, number of particles...

## 3. The model

We will create a simple MLP model. We don't need a complex Transformer or CNN. Construction of a complex model isn't the goal of this project.
We can alreayd some hypothesis : our model will confuse gluon/quart and bosonsW/bosonsZ because of their similiraity in their structure.

## 4. Results

We have a model with 76.7% accuracy and an AUC of 0.94 on the test set, consistent with the validation results. The confusion matrix shows that the main errors are—as expected—gluon/quark confusion, and W and Z bosons classified as light quarks.

