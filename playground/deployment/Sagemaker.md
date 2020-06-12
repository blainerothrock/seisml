# Running From Sagemaker
A quick guide to getting a model to use from sagemaker

## Starting Sagemaker
* Navigate to https://aws.amazon.com/sagemaker/
* On the top right corner, Sign in to the console
* From the console, create a new user with the default options
* Wait for the user to load and select Open Studio

## Download the Repo
* Open a new Launcher from the file menu
* Select Pytorch (Optimized for GPU) from the image selection dropdown
* Select image terminal
* Wait for the terminal to load, then clone the repo with `git clone`
* Move into the seimsl directory created

## Training
* Open the `environment.yml` file and and delete the line with `ld_impl_linux-64==2.34=h53a641e_0`
* Open `/seisml/seisml/utility/download_data.py` and change `with open(os.path.join('seisml/utility/checksums', '{}.md5'.format(name)), 'r') as f:` to `with open(os.path.join('root/seisml/seisml/utility/checksums', '{}.md5'.format(name)), 'r') as f:`
* Move `train.py` `utils.py` and `config.gin` from `/experiments/triggered_earthquake` to the root directory
* In `config.gin` change `'/home/b/.seisml/experiments/triggered_earthquakes/'` to `'/root/.seisml/experiments/triggered_earthquakes/'`
* From the seisml directory, run `bash` to open a new bash shell
* Create your conda environment with `conda env create`
* Start your conda environment with `source activate seisml`
* Run `python train.py`
