# Replicate & Batch (RnB)  

This repository holds the codes and models for the paper: 
**RnB: Intelligent Replication & Batching for Pipelined NN Inference on GPUs**, authors, conference, year. 
[Arxiv Preprint]()

## Installation 

### External Dependencies 
* **NVVL**
NVVL (NVIDIA Video Loader) helps loading sampled frames from video files. 
Since [the forked repository] (https://github.com/jsjason/nvvl) has a customized implementation of loader and dataset classes for RnB, we will be using the forked one instead of the official one distributed by NVIDIA. 
* ffmpeg 4.0
* OpenCV 3.4.2 
* PyTorch 0.4.1

The following external dependencies are required and can be easily installed in a virtual environment by following the instructions down below.  

### Setting up the environment 
You can use `spec-file.txt` to create your own conda environment for testing the code. If you already have such an environment, then you can ignore this file and the installation process below.

```bash
$ conda create -n <env_name> --file spec-file.txt # install ffmpeg 4.0, OpenCV 3.4.2, PyTorch 0.4.1, and other dependencies
$ source activate <env_name>
$ cd ~
$ export PKG_CONFIG_PATH=/home/<username>/miniconda2/envs/<env_name>/lib/pkgconfig:$PKG_CONFIG_PATH # change accordingly, should include the file libavformat.pc
$ git clone https://github.com/jsjason/nvvl
$ cd nvvl/pytorch
$ python setup.py install # install nvvl

$ cd ~/rnb # or any directory that doesn't have a subdirectory called 'nvvl'
$ python -c 'from nvvl import RnBLoader' # should finish without any errors

```
### Model Reference  
This repository contains the codes and models from the paper: 

[A Closer Look at Spatiotemporal Convolutions for Action Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf), Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri, CVPR, 2018. 

The authors officially call this model R(2+1)D for short, but we will use `r2p1d` without both parentheses and capital letters in the codes & filenames when referring to this model for convenience.
The model is already imported under the directory named `models` for your ease.

### Trained Model Checkpoint Preparation 
The checkpoint is stored at `cmsdata/ssd0/cmslab/Kinetics-400/ckpt/model_data.pth.ts`, and is already hardcoded in `r2p1d_runner.py` under `ckpt`. 
We will later give a public link for downloading the checkpoint.

### Data Preparation
The data used for this test benchmark uses [Kinetics-400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/). The data is stored under `/cmsdata/ssd0/cmslab/Kinetics-400/sparta`, and this full directory is already given in `client.py` under a variable named `root`.   

## Benchmark Test
RnB aims to optimize DNN inference on **video analytics models with mutliple steps** in **multiple GPU environment**. The models can include tasks like video frame extraction, feature extraction and main neural network execution.  
Meanwhile, we also provide a manual test benchmark for runnning video inference jobs in any valid setting with regard to the number of GPUs, the number of replicas per step etc.
The test benchmark can be used to run not only the optimal configuration by RnB but other settings as well for for comparison purposes. 
Thus this test benchmark serves to seek optimal configuration for running R(2+1)D model and achieve lower latency and higher throughput. 

![Benchmark Test Pipeline Image](/images/benchmark_test.png)

The figure above delineates the end-to-end pipeline of the test benchmark. As you can see from the figure above, there exist three jobs in total: client, loader and runner. In this test benchmark, three processes are spawned by default, meaning one for each step. The number of processes for each step, however, can be modified, and the analysis on the tests with different number of processes, batches, replications and GPUs will be explored to find the ultimate optimal configuration. 

In order to concurrently execute multiple processes at the same time, two queues exist and let items pass across processes.  
  - `filename_queue`: This queue will pass each path of videos from `client` process to `r2p1d_loader` process.   
  - `frame_queue`: This queue will pass a batched tensor of a single video from `r2p1d_loader` process to `r2p1d_runner` process. 

### Files Description
Descriptions on each files are given to introduce you how this test benchmark performs. 

- `benchmark.py`
This file serves as the main process that spawns all the other processes. We aim to find the optimal relationship between different parameters and video-analytics tasks by testing out the combinations of different parameters.  
  - `-mi` or `--mean_interval_ms`
     Indicates how often you expect **Client** to generate inference requests in milliseconds. This given interval abides by the Poisson distribution, meaning we will assume the work of sending videos to **Loader** occurs independently and happens every indicated interval time.
  - `-g` or `--gpus`
     Indicates how many GPUs you are willing to use. 
  - `-r` or `--replicas_per_gpu`
     Indicates how many runner processes you would like to spawn per GPU.   
  - `-b` or `--batch_size`
     Indicates how many videos you would like to process in a batch. The default batch size is set to 1, meaning only one video will be run on the neural network. 
  - `-v` or `--videos`
     Indicates how many videos you would like to test. The default is set to 2K videos. 

- `client.py`
This file first prepares a list of video filepaths. If the length of the list is smaller than the number of videos you have given for `-v`, the list will be repeated sufficient number of times until the number of videos that will be tested matches with the given parameter. After the list of filepaths is ready, one filepath will be sent to the `filename_queue` in every interval sampled from Poisson distribution, where the mean interval is given as the argument for `-mi`.

- `r2p1d_sampler.py`
Before we move onto the next step, loader, it is important to understand the role of sampler. The sampler class in this file will choose indices of frames to read of a given video based on the number of clip length (number of frames per clip) and the number of clips. By following the implementation of the R(2+1)D paper, we will sample 10 clips in which one clip consists of 8 frames.    

- `r2p1d_loader.py`
This file will extract features of sampled clips with a certain number of frames. This extraction process uses NVVL, and the details regarding NVVL will be further described below. The extracted features will be put in to `frame_queue`.  

- `r2p1d_runner.py`
This file runs the R(2+1)D network with extracted features of a sampled video from `frame_queue`. This runner process is spawned as many as the number of GPUs * the number of replica per GPU.    


### How NVVL works in RnB
RnB implementations on top of NVVL cannot be directly accessed in this git repository. Knowing how NVVL is used and what information is coming in as an input or coming out as an output in what form will help you understand the test benchmark better. Thus this description aims to provide general idea on how NVVL is used in this system, especially for R(2+1)D model. 

Let's look into some lines and  the **loader** in `r2p1d_loader.py` file.  
```python
loader = nvvl.RnBLoader(width=112, height=112,
                        consecutive_frames=8, device_id=g_idx,
                        sampler=R2P1DSampler(clip_length=8))

while True:
  tpl = filename_queue.get()
  filename, time_enqueue_filename = tpl 
  loader.loadfile(filename)
  for frames in loader:
    pass 
  loader.flush()

  frame_queue.put((frames, 
                   ... )) 
```
Before moving onto some methods implemented inside **RnBLoader** in `~/nvvl/pythorch/nvvl/rnb_loader.py`, we can see that the `loader` first creates an `RnBLoader` object. 
`tpl` dequeued from the `filename_queue` will give a full path to a video and save the path under the `filename`.

```python
from .rnb_dataset import RnBDataset 

class RnBLoader(object):
  def __init__ ( ... ):
    self.dataset = RnBDataset (...)
    self.tensor_queue = collections.dequeue() 
    self.batch_size_queue= = collections.dequeue()
    ... 

  def  _receive_batch(self):
    batch_size = self.batch_size_queue.popleft()
    t = self.dataset._create_tensor_map(batch_size)
    labels = []
    for i in range(batch_size):
      _, label = self.dataset._start_receive(t, i)
      labels.append(label)

    self.tensor_queue.append((batch_size, t, labels))
  
  def loadfile(self, filename):
    length = self.dataset.get_length(filename)
    frame_indices = self.sampler.sample(length)
    self.dataset._read_file(filename, frame_indices)
    self.batch_size_queue.append(len(frame_indices))
  
  def __next__(self):
    if self.batch_size_queue:
      self._receive_batch()

    batch_size, t, labels = self.tensor_queue.popleft()
    for i in range(batch_size):
      self.dataset._finish_receive()

    return t['input'] 
```
Let's also take a look into `_read_file` method inside `RnBDataset` in `~/nvvl/pytorch/nvvl/rnb_dataset.py`
```python
class RnBDataset(torch.utils.data.Dataset):
  def __init__( ... ):
    ...

  def _read_file(self, filename, frame_indices):
    for index in frame_indices:
      lib.nvvl_read_sequence(self.loader, str.encode(filename),
                             index, self.sequence_length)
      self.seq_info_queue.append((filename, index))
      self.samples_left += 1 
```
When the method `loadfile` is called with `filename` as parameter, it will read indices of the sampled sequences (`frame_indices`) and start reading frame by frame while looping over all the indices of the video in `_read_file` method. 
During this process, the number of indices will be appended to the `batch_size_queue` and a tuple of (filename, index) will be appended to `seg_info_queue`. 
When the `loader` iterates, it checks whether there is an item inside `batch_size_queue`. If there is, meaning there has at least been one file that called `loadfile` method, the method `_receive_batch` will be called next. 
This method will append a tuple to `tensor_queue` which contains information including but not limited to **batch size, batched tensor map and label**. Here we can see all the features of individual frames extracted from one video are now merged into one tensor map, and that ***one mapped tensor will be returned and be put into the `frame_queue` on every iteration of the `loader`.*** 

### How to Run 
Since we have data and the models ready, it's time to test in an end-to-end manner. 
First, designate GPUs that you would like to utilize. 
```bash
$ export CUDA_VISIBLE_DEVICES=0,1 # or any combination you wish from [0,1,2,3,4,5]
$ python benchmark.py -mi 100 -g 2 -r 1 -v 100
```
When the upper-mentioned commands are run, the following messages will be printed and logs will be saved.  
```bash
Args: Namespace(batch_size=1, gpus=2, mean_interval_ms=100, replicas_per_gpu=1, videos=100)
START! 1553666151.042721
FINISH! 1553666159.479689
That took 8.436968 seconds
Waiting for child processes to return...
Average filename queue wait time: 1.336455 ms
Average frame extraction time: 10.847920 ms
Average frame queue wait time: 11.619222 ms
Average inter-GPU data transmission time: 0.006759 ms
Average neural net time: 84.054208 ms

$ ls logs
190327_145542-mi100-g2-r1-b1-v100

$ ls logs/190327_145542-mi100-g2-r1-b1-v100/
g0-r0.txt  g1-r0.txt  log-meta.txt
```
HAVE FUN EXPERIMENTING! 

## [FAQ]()

