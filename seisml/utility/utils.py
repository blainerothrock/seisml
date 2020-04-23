import numpy as np
import torch
import shutil
import os
import json
from seisml import networks
import inspect

import pickle
from scipy.signal import butter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

model_functions = {
    'fc': networks.FullyConnected,
    'conv': networks.DilatedConvolutional
}

def show_model(model):
    print(model)
    num_parameters = 0
    for p in model.parameters():
        if p.requires_grad:
            num_parameters += np.cumprod(p.size())[-1]
    print('Number of parameters: %d' % num_parameters)

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-3] + '_best.h5')

def load_class_from_params(params, class_func):
    arguments = inspect.getfullargspec(class_func).args[1:]
    if 'input_size' not in params and 'input_size' in arguments:
        params['input_size'] = params['length']
    filtered_params = {p: params[p] for p in params if p in arguments}
    return class_func(**filtered_params)

def load_model(run_directory, device_target='cuda'):
    with open(os.path.join(run_directory, 'args.json'), 'r') as f:
        args = json.load(f)

    model = None
    device = None

    saved_model_path = os.path.join(run_directory, 'checkpoints/latest.h5')
    device = torch.device('cuda') if device_target == 'cuda' else torch.device('cpu')
    class_func = model_functions[args['model_type']]
    model = load_class_from_params(args, class_func).to(device)

    model.eval()
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    return model, args, device

def visualize_embedding(embeddings, labels, output_file, pca=None):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(embeddings)
    output = pca.transform(embeddings)
    colors = np.argmax(labels, axis=-1)
    plt.style.use('classic')
    plt.scatter(output[:, 0], output[:, 1], c=colors, cmap='coolwarm')
    #plt.xlim([-1.0, 1.0])
    #plt.ylim([-1.0, 1.0])
    plt.xlabel('PCA0')
    plt.ylabel('PCA1')
    plt.title('Visualization of learned embedding space')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_file)

    return pca


# from original top level
def save_file(file_name, data):
    f = open(file_name, 'wb')
    pickle.dump(data, f)
    f.close()

def load_file(file_name):
    f = open(file_name, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def parallel_process(array, function, n_jobs=4, use_kwargs=False, front_num=1):
    """
        A parallel version of the map function with a progress bar.
        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    else:
        front = []
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
            #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out
