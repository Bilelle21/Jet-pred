# Jet prediction

LHC sends paquets of proton 40 milions times by seconds. Each collision produce many particles measured by sensors.But it's too many data save : the majority of data is lots. So the idea is to construct a system who choses which collisions to keep in real time.This decision need to be done in microseconds.
When particles are produced with a collision, there are not observed direclty, there are fragmented in packs of particles called jets. By studying the form and the structure of this jet, we can deduce which type of particle it is.


## 1. The objective

The objective is to predict which type of particel is produced 

## 2. The data

We we will use the dataset hls4ml composed of simulated jets dispattched between 5 classes depending on their particle origin : gluon (g), quark léger (q), boson W (w), boson Z (z) et quark top (t). We have 830 000 data with each jet described by 16 features that sums up their form : mass,intern structure, number of particles...