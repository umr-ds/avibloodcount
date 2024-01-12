# AviBloodCount
This software enables fully automated counting of blood cells, including erythrocytes as well as leukocytes, in avian blood samples. AviBloodCount takes whole slide images as input and outputs the overall cell counts of all areas determined as countable.

We provide our models and annotated data at [data.uni-marburg.de](https://data.uni-marburg.de/handle/dataumr/250).

## Installation
We provide a docker file to create an environment containing all dependencies necessary to run our model on whole slide images stored in the SVS file format.

### Install Docker
To install Docker, follow the instructions at [docker.com](https://www.docker.com/get-started/)

### Build Docker image
To build the docker image go to the `docker` directory and run the following command:

` docker build --build-arg="USER_ID=$(id -u)" -t umr-ds/avibloodcount .`

## Usage
### Preparation
1. Download our trained models, i.e., `efficientNet_B0.onnx` and `condInst_R101.pth` from [here](https://data.uni-marburg.de/handle/dataumr/250) to `./models`. 
2. Place the SVS files to be processed in `./input`. We provide several example SVS files at the above link.

### Run inference
Running the Docker image as follows automatically starts inference with default parameters on all SVS files located at `./data`:

`docker run --rm -it -v ../code:/code -v /path/to/data:/data --shm-size 8gb --name avibloodcount umr-ds/avibloodcount`

To run with GPU support, do the following:

`docker run --rm -it -v ../code:/code -v /path/to/data:/data --shm-size 8gb --name avibloodcount umr-ds/avibloodcount --gpu 0`

In this example, the code will run on GPU with index 0.

You can modify several parameters by passing them trailing the `docker run command`. For example, you can set the thresholds used for the countability classification and the instance segmentation models via `--cls_thresh` and `--det_thresh`, respectively. All available parameters can be listed by issuing 

`docker run --rm -it -v ../code:/code -v /path/to/data:/data --shm-size 8gb --name avibloodcount umr-ds/avibloodcount --help`


## Scientific Usage & Citation

If you are using this software in academia, we'd appreciate if you cited our [scientific research paper](#) as follows:

> M. Vogelbacher. F. Strehmann, H. Bellafkir, M. Mühling, N. Korfhage, D. Schneider, S. Rösner, D. G. Schabo, N. Farwig, and B. Freisleben, "Identifying and Counting Avian Blood Cells in Whole Slide Images", Submitted for publication, 2024.

```bibtex
@article{vogelbacher2024identifying,
  title = {{Identifying and Counting Avian Blood Cells in Whole Slide Images}},
  author = {Vogelbacher, Markus and Strehmann, Finja and Bellafkir, Hicham and M{\"u}hling, Markus and Korfhage, Nikolaus and Schneider, Daniel and R{\"o}sner, Sascha and Schabo, Dana G. and Farwig, Nina and Freisleben, Bernd},
  booktitle = {Submitted for publication.},
  year = {2024},
  month = Jan,
  journal = {},
  keywords = {cell segmentation, bird blood analysis, microscopy images, blood smear images, object detection, ornithology},
  doi = {},
}
```