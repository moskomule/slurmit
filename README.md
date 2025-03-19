# slurmit

**This project is work in progress**

![pytest](https://github.com/moskomule/slurmit/workflows/pytest/badge.svg)
[![document](https://img.shields.io/static/v1?label=doc&message=slurmit&color=blue)](https://moskomule.github.io/slurmit)

A minimalistic reimplementation of submitit for Slurm.

## Features

- Submit Python functions to SLURM
- Configure SLURM job parameters through templates
- Handle SLURM and runtime errors
- Retrieve job results

## Installation

```
hatch create env
```

or

```
pip install -U -e .
```

## Usage

Prepare template

```shell
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{num_gpu}

module load cuda
```

Run the following script

```python
from slurmit import SlurmExecutor


def add(x, y):
    return x + y


ex = SlurmExecutor("slurm_outputs",
                   "template.sh",
                   dict(partition="main", num_gpu=1))
job = ex.submit(add, 1, y=2)
print(job.result())  # 3
```