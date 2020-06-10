# SeisML
![Test](https://github.com/blainerothrock/seisml/workflows/Test/badge.svg)
[![codecov](https://codecov.io/gh/blainerothrock/seisml/branch/master/graph/badge.svg)](https://codecov.io/gh/blainerothrock/seisml)

A working repository for experiementations in machine learning in seismology.

## Repo Structure
* `_old/`
    - contains unused code from *Automating the Detection of Dynamically Triggered Earthquakes via a Deep Metric Learning Algorithm* paper
* `experiments/`
    - the directory that contains code specific to a experiment utilizing seisml components.
    - This includes, model training code, inference and hyperparameter configuration 
* `playground/`
    - A place for experiements in progress, example code, data exploration, etc.
* `seisml`
    - The root directory for the framework (python package)
    * `core`
        - contains universal components
        - `transforms`
            - model after transforms in `torchvision`, used for preprocessing steps before feeding data into a model
    * `datasets`
        - build in Pytorch datasets for use in experiments
    * `metrics`
        - calculations modeling
    * `networks`
        - custom Pytorch models
    * `utility`
        - universal helper methods
* `tests`
    - Pytest unit tests for seiml code. These test run in continuous integration and utilize limited resources
* `tests-non-ci`
    - Pytest test that are more specific to experiement debugging. Not intended for continuous integration because of the amount of resources used.
* `environment.yml`
    - conda environment file for CI and use
    
## Installation
* clone the repository and `cd` to root
* create a new Anaconda environment
```shell script
conda create env -f environment.yml
``` 
* run a experiments following the `README.md` found in the specific experiment directory.

## Experiemnts
* [Triggered Earthquake](experiments/triggered_earthquke/README.md)
    - Deep metric learning for classification of small earthquakes triggered by larger distant earthquakes
* [Triggered Tremor](...)
    - Deep Metric and Convolutional network for classificationing Tremors triggered by earthquakes
* [Mars](experiments/mars_insight_clustering/README.md) **IN PROGRESS**
   - Unsupervised clustering of seimic data from the mars insight lander. 
 

## Inspiration 
The inspiration and starting codebase for this model is from the Seismological Research Letters paper 
*[Automating the Detection of Dynamically Triggered Earthquakes via a Deep Metric Learning Algorithm](https://pubs.geoscienceworld.org/ssa/srl/article-abstract/91/2A/901/579921/Automating-the-Detection-of-Dynamically-Triggered)*
([original codebase](https://github.com/interactiveaudiolab/earthquakes)). 
* Steps to reproducing this paper can be found [here](experiments/triggered_earthquake/README.md).
