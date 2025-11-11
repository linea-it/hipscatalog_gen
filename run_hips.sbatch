#!/bin/bash
#SBATCH --job-name=hipscatalog_gen
#SBATCH --output=hipscatalog_gen_%j.out
#SBATCH --error=hipscatalog_gen_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --account=hpc-bpglsst

# =============================================================================
# HiPS Catalog Pipeline — SLURM submission script
# -----------------------------------------------------------------------------
# This template runs the HiPS catalog generation pipeline on LIneA’s Apollo cluster,
# using the modular hipscatalog_gen package.
#
# Before submitting, make sure to:
#   1) Adjust the 'partition' (--partition) and 'account' (--account) directives
#      according to your access permissions.
#      → If you don’t have access to 'hpc-bpglsst', choose another valid account.
#      → For details on partitions, accounts, and job submission, see:
#        https://docs.linea.org.br/processamento/apollo/index.html
#   2) Update the paths for your Conda environment and code directory below.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Activate the Conda environment
# -----------------------------------------------------------------------------
# Replace the path below with the location of your environment.
# The environment must include Dask, LSDB, and all pipeline dependencies.
source /scripts/$USER/hipscatalog_gen_env/bin/activate

# -----------------------------------------------------------------------------
# 2. Go to the pipeline directory
# -----------------------------------------------------------------------------
# This directory must contain the 'hipscatalog_gen/' package and your config file.
cd /scripts/$USER/hipscatalog_gen/

# -----------------------------------------------------------------------------
# 3. Run the HiPS Catalog Pipeline (modular package)
# -----------------------------------------------------------------------------
# Adjust the configuration file name/path as needed.
python -m hipscatalog_gen.cli --config config.yaml
