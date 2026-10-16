# Jobs (OpenPBS)    <!-- omit from toc -->

This directory contains the `.pbs` scripts used to submit experiments to the cluster via **OpenPBS**.

The goal is to keep a versioned and reproducible entry point for running training jobs in the CENAPAD HPC environment.

- [1. What is OpenPBS?](#1-what-is-openpbs)
- [2. Repository Objective (in the context of jobs)](#2-repository-objective-in-the-context-of-jobs)
- [3. The templates](#3-the-templates)
- [4. Anatomy of a PBS script](#4-anatomy-of-a-pbs-script)
- [5. How to submit a job](#5-how-to-submit-a-job)
- [6. Scratch disk and outputs](#6-scratch-disk-and-outputs)

## 1. What is OpenPBS?

**OpenPBS** is a workload manager used in HPC clusters.

It is responsible for:

- Allocating resources (GPU, CPU, memory, walltime)
- Queueing jobs
- Scheduling execution on available compute nodes
- Isolating execution environments

In our workflow, we never run training directly on the login node.  
Instead, we submit a `.pbs` script using:

```bash
qsub jobs/job.pbs
```

The scheduler then:

1. Places the job in the appropriate queue
2. Waits for resources to become available
3. Executes the script on a compute node

## 2. Repository Objective (in the context of jobs)

This repository organizes:

- PBS submission templates
- A minimal reproducible execution setup
- Standardized logs (`<output-path>/out.log`, `<output-path>/err.log`)

The idea is that every experiment can be executed in a way that is:

- Reproducible
- Documented
- Versioned alongside the code

## 3. The templates

Currently, there are two template scripts:

```
jobs/eval.pbs
jobs/job.pbs
```

## 4. Anatomy of a PBS script

A `.pbs` script is simply a bash script with special scheduler directives at the top:

```bash
#PBS -N job-name
#PBS -q queue-name
#PBS -l walltime=HH:MM:SS
#PBS -e error-file
#PBS -o output-file
```

These directives tell the scheduler:

- The job name
- Which queue to use
- The maximum walltime
- Where to write logs

After that, it is standard bash:

1. GPU check (`nvidia-smi`)
2. Python script execution

## 5. How to submit a job

From the repository root:

```bash
qsub jobs/job.pbs
```

After submission:

- The job receives an ID
- It waits in the queue
- Logs are written to `<output-path>/out.log` and `<output-path>/err.log`

To monitor:

```bash
qstat -u $USER
```

## 6. Scratch disk and outputs

Run training and store intermediate artifacts on the cluster scratch disk,
such as `/work/username/`. Do not train directly in your home directory:
scratch storage is intended for active jobs and avoids filling your home
quota.

Before submitting, replace the `/work/username/` paths in the PBS template with
your own scratch directory. The training template writes results there during
the job and copies the completed run to the repository's `outputs/` directory
after training finishes:

```bash
cp -r /work/username/islr-jepa/vjepa/si-split/<run> \
	outputs/vjepa/si-split/<run>
```

Keep the copied results, including checkpoints, configs, and logs, as the
long-term record of the experiment. Adapt the destination if your cluster
requires completed results to be copied to a different persistent location.