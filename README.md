# FYP 2DGS


## Installation

#### 1. Install LaTeX

This is for SciencePlots to work properly.

Through linux, this can be done using:

```bash
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```


More information is available at:

https://github.com/garrettj403/SciencePlots/wiki/FAQ


#### 2. Install Python Dependencies

In this repository, torch 2.11.0 and torchvision 0.25.0 were used.

```bash
pip install torch==2.11.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```




## Running

The main program contains an array of functions that demonstrate the example performance of the model as well as multiple benchmarks aimed at recording in various ways.

```bash
python main.py
```