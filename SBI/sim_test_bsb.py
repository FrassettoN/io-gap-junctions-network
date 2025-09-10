import numpy as np
import torch
from mpi4py import MPI

import bsb
from bsb import from_storage
from bsb.services.mpi import MPIService
from bsb_nest import NestAdapter

from utils import plot_vm, plot_sr, analyze_vm

def simulate():
    adapter.reset_kernel()
    adapter.prepare(simulation)
    results = adapter.run(simulation)
    results = adapter.collect(results)[0]
    spiketrains = results.spiketrains
    signals = results.analogsignals

    if not signals:
      print("No Signal")
      return 0.

    mean = np.mean(signals[0])
    return float(mean)
    

if __name__ == "__main__":
    simulation_name = "basal_activity"
    scaffold = from_storage("io_sbi.hdf5")
    simulation = scaffold.get_simulation(simulation_name)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    adapter = NestAdapter(comm = comm)

    n_sims = 100
    local_indices = list(range(rank, n_sims, size))
    local_results = []

    for i in local_indices:
        result = simulate()
        local_results.append(result)

    local_results = torch.tensor(local_results, dtype=torch.float32)
    all_results = comm.gather(local_results, root=0)
    if rank == 0:
        x = torch.cat(all_results, dim=0)
        print(f"Completed {n_sims} simulations. Final result shapes: x = {x.shape}")
        # Create dictionary with unique values and their occurrences
        unique_values, counts = torch.unique(x, return_counts=True)
        value_dict = {float(val): int(count) for val, count in zip(unique_values, counts)}
        
        print("Unique values and their occurrences:")
        for value, count in value_dict.items():
            print(f"  {value}: {count} times")