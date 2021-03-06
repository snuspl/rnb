# Replicate & Batch (RnB) Test Benchmark  

RnB aims to optimize DNN inference on **video analytics models with multiple steps** in **multi-GPU environments**. Steps include video frame extraction, feature extraction, main neural network execution and other potential computations that are necessary for processing videos.  
Meanwhile, we also provide a manual test benchmark for running video inference jobs in any valid setting with regard to the number of GPUs, the number of replicas per step, etc.
The test benchmark can be used to run not only the optimal configuration generated by RnB but other settings as well for comparison purposes. 
Thus this test benchmark serves as the testing ground for experiment with various configurations in the hope of achieving lower latency and higher throughput. 

![Benchmark Test Pipeline Image](/images/benchmark_test.png)

The figure above shows the end-to-end pipeline of the test benchmark. 
As you can see from the figure above, there exist two kinds of processes in total: `client` and `runner`. The number of processes for each step can vary, and the analysis on the tests with different number of processes, batches, replications and GPUs will be explored to find the ultimate optimal configuration. 

In order to concurrently execute multiple processes at the same time, input and output queues exist and let items pass across processes. You can either make global queues or local queues per GPU by configuring `-p` option in `benchmark.py`.
  - `filename_queue`: This queue will pass each path of videos from `client` process to `runner` process.   
  - `frame_queue`: This queue will pass a batched tensor of a single video between `runner` processes. 


## Files Description
Descriptions on files are given to introduce you how this test benchmark performs.
- `benchmark.py`
This file serves as the main process that spawns all other processes. We aim to find optimal configuration for video analytics tasks by testing out combinations of different parameters.  
  The following arguments can be adjusted to try different settings.   
  - `-mi` or `--mean_interval_ms`
     Indicates how often you expect **Client** to generate inference requests in milliseconds. This given interval abides by the Poisson distribution, meaning we will assume the work of putting video filename in the `filename_queue` occurs independently and happens every indicated interval time.
  - `-b` or `--batch_size`
     Indicates how many videos you would like to process in a batch. The default batch size is set to 1, meaning only one video will be run on a neural network. 
  - `-v` or `--videos`
     Indicates how many videos you would like to test. The default is set to 2K videos.
  - `-qs` or `--queue_size`
     Indicates the maximum queue size of two queues mentioned above: `filename_queue`, `frame_queue`. The default is set to 500. If enqueue occurs to a queue that is full, the benchmark will abort the job assuming the current step placement cannot handle the input stream.  
  - `-c` or `--config_file_path`
     Indicates the file path of the pipeline configuration file.
  - `-p` or `--per_gpu_queue`
     Indicates whether to place intermediate queues on each GPU

- `client.py`
This file first prepares a list of video filepaths. If the length of the list is smaller than the number of videos you have given for `-v`, the list will be repeated sufficient number of times until the number of videos that will be tested matches with the given parameter. After the list of filepaths is ready, one filepath will be sent to the `filename_queue` in every interval sampled from Poisson distribution, where the mean interval is given as the argument for `-mi`.

- `runner.py`
This file is used to run partitioned model and we use CUDA stream to avoid synchronizing with other processes.  

## How NVVL works in RnB
Knowing how NVVL is used and what information is coming in as an input or coming out as an output in what form will help you understand the test benchmark better. The following description aims to provide general idea on how NVVL is used in this system, especially for the R(2+1)D model. Please be advised some files mentioned in this section are not from this current repository, but from [the forked NVVL repository](https://github.com/jsjason/nvvl). 

Let's look into some lines and the **loader** in `r2p1d_loader.py` file.  
```python
# ./r2p1d_loader.py

loader = nvvl.RnBLoader(width=112, height=112,         .......... (0)
                        consecutive_frames=8, device_id=g_idx,
                        sampler=R2P1DSampler(clip_length=8))

while True:
  tpl = filename_queue.get()
  filename, time_enqueue_filename = tpl 
  loader.loadfile(filename)                            .......... (1) 
  for frames in loader:
    pass 
  loader.flush()

  frame_queue.put((frames,                             .......... (9)  
                   ... )) 
```
Before moving onto some methods implemented inside **RnBLoader** in `rnb_loader.py`, we can see that the `loader` first creates an `RnBLoader` object (0). 
`tpl` dequeued from the `filename_queue` gives a full path to a video and save the path under the variable `filename`.

```python
# Note that this file is from "https://github.com/jsjason/nvvl/"
# https://github.com/jsjason/nvvl/blob/master/pytorch/nvvl/rnb_loader.py

from .rnb_dataset import RnBDataset 

class RnBLoader(object):
  def __init__ ( ... ):
    self.dataset = RnBDataset (...)
    self.tensor_queue = collections.dequeue() 
    self.batch_size_queue = collections.dequeue()
    ... 

  def _receive_batch(self):                             .......... (7)  
    batch_size = self.batch_size_queue.popleft()
    t = self.dataset._create_tensor_map(batch_size)
    labels = []
    for i in range(batch_size):
      _, label = self.dataset._start_receive(t, i)
      labels.append(label)

    self.tensor_queue.append((batch_size, t, labels))   .......... (8)  
  
  def loadfile(self, filename):                         .......... (2)
    length = self.dataset.get_length(filename)
    frame_indices = self.sampler.sample(length)         .......... (3)
    self.dataset._read_file(filename, frame_indices)    .......... (4)
    self.batch_size_queue.append(len(frame_indices))    .......... (5) 
  
  def __next__(self):
    if self.batch_size_queue:                           .......... (6) 
      self._receive_batch()                             .......... (6')  

    batch_size, t, labels = self.tensor_queue.popleft()
    for i in range(batch_size):
      self.dataset._finish_receive()

    return t['input'] 
```
When the method `loadfile`(2) is called with `filename` as parameter in `r2p1d_loader.py` (1), it reads indices of sampled sequences (`frame_indices`) (3) and starts reading frames one by one. 
NVVL internally processes frames of a video which filename is given with the sampled indices, and holds on to the processed tensors until requested again by the loader (4). 
The number of indices in total is appended to the `batch_size_queue` (5).
When the `loader` iterates, it checks whether there is an item inside `batch_size_queue` (6). If there is, meaning there has at least been one file that called `loadfile` method, the method `_receive_batch` will be called next (6'). 
This method (7) will append a tuple to `tensor_queue` which contains information about **batch size, one batched tensor map and labels of each batched frame** (8). Here we can see that features of frames extracted from a video are now merged into one tensor map, and that ***one mapped tensor will be returned and be put into the `frame_queue` every iteration of the `loader`*** (9).

## How to Run 
### External Dependencies 
The test benchmark depends on the following: 

* NVVL: NVVL (**NV**IDIA **V**ideo **L**oader) is a library which loads tensors of sampled frames of video files straight on GPUs to accelerate and facilitate machine learning tasks with video. Since [the forked repository](https://github.com/jsjason/nvvl) has a customized implementation of loader and dataset classes for RnB, we will be using the forked one instead of the official one distributed by NVIDIA. 
* ffmpeg 4.0
* OpenCV 3.4.2 
* PyTorch 0.4.1

The external dependencies above can be easily installed in a virtual environment by following the instructions below.  

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

$ pip install py3nvml # install Py3 bindings for NVML (unavailable for Anaconda)

```
### Reference Model  
This repository contains the codes and models from the paper: 

[A Closer Look at Spatiotemporal Convolutions for Action Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf), Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri, CVPR, 2018. 

The authors officially call this model R(2+1)D for short, but we will use `r2p1d` without both parentheses and capital letters in the codes & filenames when referring to this model for convenience.
The model is already imported under the directory named `models` for your ease.

### Trained Model Checkpoint Preparation 
We use checkpoint of trained model. The checkpoint is stored at `cmsdata/ssd0/cmslab/Kinetics-400/ckpt/model_data.pth.tar`, and is already hardcoded in `models/r2p1d/model.py` under `ckpt`. 
We will later give a public link for downloading the checkpoint.

Descriptions on files related to the model are given below:

- `models/r2p1d/model.py`
This file runs the model either in a single step, or in several steps using partitioned model depending on the configuration.

- `models/r2p1d/module.py`
This file does factored R2Plus1D convolution.

### Data Preparation
The data used for this test benchmark is [Kinetics-400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/). The data is stored under `/cmsdata/ssd0/cmslab/Kinetics-400/sparta`, and this full directory is already given in `client.py` under a variable named `root`.   

Since we have data and the models ready, it's time to test in an end-to-end manner. 

Designate GPUs that you would like to utilize. Then run benchmark.py with proper arguments.

```bash
$ export CUDA_VISIBLE_DEVICES=0 # or any combination you wish from [0,1,2,3,4,5]
$ python benchmark.py -c config/r2p1d-whole.json -v 500 -mi 90
```
When the upper-mentioned commands are run, the following messages will be printed and logs will be saved.  
```bash
Args: Namespace(batch_size=1, config_file_path='config/r2p1d-whole.json', mean_interval_ms=90, per_gpu_queue=False, queue_size=500, videos=500)
START! 1561964947.928291
Finished processing 500 videos
FINISH! 1561964992.177985
That took 44.249694 seconds
Waiting for child processes to return...
Average time between enqueue_filename and runner0_start: 1.557826 ms
Average time between runner0_start and inference0_start: 0.005217 ms
Average time between inference0_start and inference0_finish: 13.254091 ms
Average time between inference0_finish and runner1_start: 1311.518943 ms
Average time between runner1_start and inference1_start: 0.005107 ms
Average time between inference1_start and inference1_finish: 85.233334 ms

$ ls logs
190701_022549-mi90-b1-v500-qs500-p0

$ ls logs/190701_022549-mi90-b1-v500-qs500-p0/
g0-r0.txt  log-meta.txt  r2p1d-whole.json
```

## Testing CUPTI
The NVIDIA CUDA Profiling Tools Interface (CUPTI) allows us to collect GPU hardware-specific metrics for profiling a CUDA application.
We are currently experimenting with CUPTI to track the achieved occupancy of a GPU.
Unfortunately, CUPTI is provided in CUDA C/C++ so we made custom Python bindings to access CUPTI features within a Python program.
Follow the steps below to build the Python-C++ bridge and test it.

```bash
# compiles utils/cupti.cpp to generate utils/cupti.so
$ ./build_cupti.sh

# let the linker know of CUPTI libraries
# this step can be skipped if LD_LIBRARY_PATH is already set correctly
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

# fetch and print GPU kernel names using CUPTI
$ python test_cupti.py
_Z21kernelPointwiseApply2I6CopyOpIffEffjLi1ELi2EEv10OffsetInfoIT0_T2_XT3_EES2_IT1_S4_XT4_EES4_T_
sgemm_32x32x32_NT_vec
```

