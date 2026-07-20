import numpy as np

dts = [0.004, 0.006, 0.008, 0.01]
std_devs_toxvaerd = [1.89995e-03, 3.98508e-03, 6.84969e-03, 1.08299e-02]
std_devs_bias_0 = [6.0967e-03, 1.3857e-02, 2.4842e-02, 3.9022e-02]
std_devs_toxvaerd_bias0 = [3.82771e-02, 8.60608e-02, 1.52302e-01, 2.35870e-01]

log_dts = np.log10(dts)
print("Slope Toxvaerd (bias=1):", np.polyfit(log_dts, np.log10(std_devs_toxvaerd), 1)[0])
print("Slope Cosine C1 (bias=0):", np.polyfit(log_dts, np.log10(std_devs_bias_0), 1)[0])
print("Slope Toxvaerd (bias=0):", np.polyfit(log_dts, np.log10(std_devs_toxvaerd_bias0), 1)[0])
