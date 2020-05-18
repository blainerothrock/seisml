# Mars Insight Seismic Data Exploration


## Instructions for Running event_examples.ipynb
​
Download GitHub Repo and Navigate to mars_insight_seismic
​
Folder Set Up
1. You'll want to create a 'data' folder in mars_insight_seismic
2. Download the following into 'data' folder :https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/mars_metadata.tar.gz
     *(I would suggest moving the zipped folder contents directly into data, if you keep the metadata folder you may have some path issues)*
3. In your 'data' folder, create an 'event_images' folder
    *(other folders will be created by the code but this one isnt for some reason)*
    
​
File Set Up
1. Navigate to JupyterLabs
2. Make sure you are in an obspy environment/kernel
3. If not, make sure you have obspy downloaded or download obspy
    - then open a terminal file and type
    -`$ conda activate obspy`
    - `(obspy)$ conda install ipykernel`
    - `(obspy)$ ipython kernel install --user --name=<any_name_for_kernel>`
    - `(obspy)$ conda deactivate`
4. In the event_examples file, in the first box, set DATA_PATH equal to the path to your 'data' folder
