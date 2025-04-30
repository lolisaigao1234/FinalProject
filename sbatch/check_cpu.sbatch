#!/bin/bash
#SBATCH --job-name=check_cpu
#SBATCH --output=cpu_info.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --account=jywu3-ic
#SBATCH --partition=IllinoisComputes

# Print general CPU information
echo "=== CPU Information ==="
lscpu

# Print number of processing units (logical cores)
echo -e "\n=== Number of logical cores ==="
nproc
