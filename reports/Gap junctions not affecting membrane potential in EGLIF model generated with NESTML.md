#### **Summary**
I attempted to implement gap junctions in the EGLIF model ([eglif_cond_alpha_multisyn.nestml](https://github.com/dbbs-lab/cerebellar-models/blob/master/cerebellar_models/nest_models/eglif_cond_alpha_multisyn.nestml)) following [NESTML documentation](https://nestml.readthedocs.io/en/latest/running/running_nest.html#gap-junctions-electrical-synapses ). Although code generation and compilation succeed, neurons connected with a gap junction do not display any difference compared to not connected controls.

Important note: the same implementation and testing protocol work correctly for ```aeif_cond_exp``` and ```hh_psc_alpha```, where connected neurons converge toward one another as expected.
#### **Environment**
- NEST version: 3.8.0-post0.dev0
- NESTML version: 8.0.1
- Python version:  3.10.13
- OS: Ubuntu 20.04.5 (Windows Subsystem for Linux)

#### **Code Availability**
All simulation scripts and test code used for this report are available at  [gap_tester.ipynb](https://github.com/FrassettoN/io-gap-junctions-network/blob/main/nestml_gap_junctions/gap_tester.ipynb)
#### **Model**
The neuron model used is [eglif_cond_alpha_multisyn.nestml](https://github.com/dbbs-lab/cerebellar-models/blob/master/cerebellar_models/nest_models/eglif_cond_alpha_multisyn.nestml).
Parameters were modified to enforce subthreshold oscillatory dynamics and avoid spiking, since the model includes a nonzero escape rate. This allows for a direct comparison of voltage traces between connected and not connected neurons.
#### **Code Generation**
Gap junctions support is enabled during code generation:

```
def generate_code(neuron_model: str, models_path=""):
    codegen_opts = {"gap_junctions": {"enable": True,
                                        "gap_current_port": "I_stim",
                                        "membrane_potential_variable": "V_m"}}

    files = os.path.join(models_path, neuron_model + ".nestml")
    generate_nest_target(input_path=files,
                            logging_level="WARNING",
                            module_name="nestml_gap_" + neuron_model + "_module",
                            suffix="_nestml",
                            codegen_opts=codegen_opts)

    return neuron_model
```

```
generate_code(
	neuron_model="eglif_cond_alpha_multisyn", 
	models_path="../nest_models"
)
```

Compilation succeeds without errors.
#### **Simulation Setup**

Four neurons were created:
- Neurons 1 and 2: connected by a gap junction
- Neurons 3 and 4: not connected (control)
- Initial conditions:
    - Neurons 1 and 3 start at -46 mV
    - Neurons 2 and 4 start at -45 mV

Full parameterization is provided to reproduce the subthreshold oscillatory regime.

```
nest.Install("nestml_gap_eglif_cond_alpha_multisyn_module")
nest.resolution = 0.05
neurons = nest.Create("eglif_cond_alpha_multisyn_nestml", 4)
neurons.set({
        "V_m": -45,
        "E_L": -45,
        "C_m": 189,
        "tau_m": 11,
        "I_e": -18.101,
        "k_adap": 1.928,
        "k_1": 0.191,
        "k_2": 0.090909,
        "V_th": -35,
        })
        
neurons[0].V_m = -46.0
neurons[2].V_m = -46.0
```
##### Gap Junction Connection
```
nest.Connect(
	neurons[0], neurons[1], 
	{"rule": "one_to_one", "allow_autapses": False,"make_symmetric":True}, 
	{"synapse_model": "gap_junction", "weight": 100}
)
```
Connection is present in `nest.GetConnections()`
##### Voltmeter, Simulation & Analysis
```
vm = nest.Create("voltmeter", params={"interval": 0.1})
nest.Connect(vm, neurons, "all_to_all")

nest.Simulate(1000.0)

vm_values = vm.events["V_m"]
senders = vm.events["senders"]
vm_per_cell = {}
for cell_num in np.unique(vm.events["senders"]):
	vm_per_cell[cell_num] = vm_values[np.where(senders == cell_num)]

print(f"V_m of cell 1 and cell 3 are equal: {np.array_equal(vm_per_cell[1], vm_per_cell[3])}")
print(f"V_m of cell 2 and cell 4 are equal: {np.array_equal(vm_per_cell[2], vm_per_cell[4])}")
```
#### **Results for ```eglif_cond_alpha_multisyn```**
Both connected and not connected neuron pairs produce identical traces:
```
V_m of cell 1 and cell 3 are equal: True 
V_m of cell 2 and cell 4 are equal: True
```

EGLIF Voltage Trace
![EGLIF voltage trace](./images/EGLIF%20Voltage%20Trace.png)
Overlapping traces show no coupling

#### **Results for ```aeif_cond_exp``` and ```hh_psc_alpha```** 
```
V_m of cell 1 and cell 3 are equal: False 
V_m of cell 2 and cell 4 are equal: False
```

AEIF Voltage Trace
![AEIF voltage trace](./images/AEIF%20Voltage%20Trace.png)
Converging traces show correct gap junction behavior.

For this test, the following parameter modifications were applied:
- Constant current input: I_e = 20 pA for all neurons
- Initial membrane potential: V_m = –60 mV for neurons 0 and 2

#### **Tried Solutions**
1. **Modified connection weight**
	- Tested weights from 1 up to 10000.
	- No observable difference in EGLIF; effect appears correctly in AEIF.
2. **Changed `gap_current_port` variable**
	- Tried both `"I_stim"` (current input port in EGLIF) and `"I_gap"`.
	- `I_gap` was added to the nestml model as continuous input and inserted in the model equation. 
	- The same approach in AEIF works correctly.
3. **Record `I_stim`**
	- Created `recordable inline I_stim_recordable pA = I_stim` and used it in the equations.
	- For both EGLIF and AEIF, `I_stim_recorded` stays at zero for gap junctions, but correctly shows input from a `dc_generator`.
4. **Changed EGLIF Parameters**
	- Randomized EGLIF parameters: tested both subthreshold and spiking regimes.
	- No difference between connected and not connected neurons.
5. **Changed EGLIF Equation**
	- Added the minus sign at the beginning of the equation, to exclude possible effects of this difference from a LIF model
	- No difference between connected and not connected neurons.
6. **Removed `steps()` and `resolution()` from EGLIF model**
	- Removed calls to fixed-timestep functions (resolution() and/or steps()) to avoid the compilation warning: _“Model contains a call to fixed-timestep functions. This restricts the model to being compatible only with fixed-timestep simulators.”_
	- No difference between connected and not connected neurons.
7. **Changed NEST resolution**
	- From 0.05 to 1 
	- No effect on EGLIF.
8. **Adjusted iterative solver ([Jacobi Waveform Relaxation](http://journal.frontiersin.org/article/10.3389/fninf.2015.00022/full))**
	- ```
	  nest.SetKernelStatus({'use_wfr': True,
                      'wfr_comm_interval': 1.0,
                      'wfr_tol': 0.0001,
                      'wfr_max_iterations': 15,
                      'wfr_interpolation_order': 3})
	  ```
	- Tried interpolation_order 1 and 3 (2 fails for both EGLIF and AEIF); varied other parameters.
	- No effect on EGLIF.
