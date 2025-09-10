from utils import analyze_vm
from mpi4py import MPI
import time
import torch

def simulate():
    import nest
    nest.ResetKernel()
    nest.set_verbosity(100)
    nest.SetKernelStatus({
        "local_num_threads": 1,
    })

    neuron = nest.Create("aeif_cond_alpha_multisynapse", 1)
    vm = nest.Create("voltmeter", params={"interval": 0.1})
    sr = nest.Create("spike_recorder")
    nest.Connect(vm, neuron)
    nest.Connect(neuron, sr)
    nest.Simulate(5000.0)

    if not len(vm.events["V_m"]):
      return 0.

    return analyze_vm(vm)

if __name__ == "__main__":    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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