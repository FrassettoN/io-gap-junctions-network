from matplotlib import pylab as plt 
import quantities as pq


# Get and extract device data from the configuration
def extract_device_data(config, simulation_name="basal_activity"):
    """
    Extract device information from the simulation configuration.
    
    Args:
        config: The BSB configuration object
        simulation_name: Name of the simulation to extract devices from
        
    Returns:
        List of device dictionaries with relevant properties
    """
    try:
        devices = config["simulations"][simulation_name]["devices"]
        devices_data = []
        
        # Define properties to extract
        properties_to_check = [
            "device", "start", "stop", "weight", "rate", "amplitude", 
            "frequency", "target", "receptor"
        ]
        
        # Loop through all devices
        for device_name, device_obj in devices.items():
            # Create base device info
            device_info = {
                "name": device_name,
                "type": type(device_obj).__name__
            }
            
            # Try to access various common properties
            for prop in properties_to_check:
                try:
                    if hasattr(device_obj, prop):
                        device_info[prop] = getattr(device_obj, prop)
                    elif isinstance(device_obj, dict) and prop in device_obj:
                        device_info[prop] = device_obj[prop]
                except (AttributeError, KeyError):
                    device_info[prop] = None
                    
            # Extract parameters from parameters attribute if it exists
            if hasattr(device_obj, "parameters"):
                for param, value in device_obj.parameters.items():
                    if param not in device_info:
                        device_info[param] = value
            
            # Add to device list
            devices_data.append(device_info)
            
        return devices_data
    
    except KeyError as e:
        print(f"Error: Could not find simulation '{simulation_name}' or its devices: {e}")
        return []
    except Exception as e:
        print(f"Error extracting device data: {e}")
        return []

def construct_label(device):
    """
    Construct a descriptive label for a device based on its properties.
    
    Args:
        device: Dictionary containing device properties
        
    Returns:
        String containing formatted device label
    """
    
    device_name = device["name"]

    # Collect all relevant properties
    extra_info = []
    for prop in ["type", "amplitude", "rate","weight"]:
        if prop in device and device[prop] is not None:
            value = device[prop]

            if prop == "type":
                extra_info.append(f"{value}")                        
            elif prop == "rate" or prop == "frequency":
                extra_info.append(f"fr: {value:.2f} Hz")
            elif prop == "weight":
                extra_info.append(f"w: {value:.2f}")
            elif prop == "amplitude":
                extra_info.append(f"a: {value:.2f}")
            else:
                extra_info.append(f"{prop}: {value}")

    
    # Construct final label
    if extra_info:
        return f"{device_name} ({', '.join(extra_info)})"
    else:
        return device_name

def plot_signal(signal, devices=[], start=None, stop=None, figsize=(10, 4), title=None, output=None):
    # Assuming `signal` is your AnalogSignal object from Neo

    # Slice the signal
    segment=signal
    if start:
        start_time = start * pq.ms   
        segment = segment.time_slice(t_start=start_time, t_stop=segment.t_stop)

    if stop:
        stop_time = stop * pq.ms 
        segment = segment.time_slice(t_start=segment.t_start, t_stop=stop_time)  


    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(segment.times.rescale('ms'), segment.magnitude,)

    # Axis labels
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel(f"Potential ({segment.units.dimensionality.string})", fontsize=12)
    
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan' 'magenta']

    for i, device in enumerate(devices):
        label = construct_label(device)
            
        if "start" in device:
            plt.axvline(x=device["start"], color=colors[i], linestyle=':', linewidth=1.5, label=label)
        if "stop" in device:
            plt.axvline(x=device["stop"], color=colors[i], linestyle=':', linewidth=1.5, label=label)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')

    
    # Title
    if title:
      plt.title(label=title, fontsize=10)
    plt.tight_layout()

    if output:
        plt.savefig(output,dpi=100)
    plt.show()