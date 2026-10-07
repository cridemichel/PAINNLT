# cython: language_level=3

from libcpp.string cimport string
from libcpp.memory cimport make_shared, shared_ptr
from libcpp cimport bool as cpp_bool

# Dichiara l'interfaccia C++
cdef extern from "core/nonbonded_interactions/PaiNN_ML_Potential.hpp":
    cdef cppclass PaiNN_ML_Potential:
        PaiNN_ML_Potential(const string& model_path, int num_species, int hidden_channels, int n_layers, int num_rbf, double cutoff, double toxvaerd_alpha, int ordered_geometry_nodes, int ordered_geometry_head_layers, int ordered_geometry_head_width, double ordered_geometry_energy_scale_kj_mol, cpp_bool ordered_geometry_head_only, int ordered_geometry_copies, cpp_bool tel22_shared_geometry, const string& device_str, const string& precision_str, cpp_bool thermo_heads, double thermo_T0, double temperature_K)
        double get_cutoff()
        cpp_bool has_thermo_heads()
        void set_temperature(double temperature_K) except +
        double get_temperature()
        double get_last_energy()
        void configure_profiling(bint enabled, long long warmup_calls)
        void reset_profiling()
        string get_profile_json()
        
    cdef shared_ptr[PaiNN_ML_Potential] global_painn_potential

def get_painn_energy():
    """Restituisce l'ultima energia potenziale calcolata dal modello PaiNN in C++"""
    if global_painn_potential.get() != NULL:
        return global_painn_potential.get().get_last_energy()
    return 0.0

def deactivate_painn_potential():
    """Rilascia il potenziale PaiNN globale e le sue risorse sul device.

    Da chiamare prima che il processo esca (run_cg_md.py lo registra con
    atexit): cosi' tensori e grafi CUDA vengono liberati mentre il runtime CUDA
    e' ancora attivo, invece che dai distruttori statici a fine processo.
    """
    global global_painn_potential
    global_painn_potential.reset()

def configure_painn_profiling(enabled: bool = True, warmup_calls: int = 0):
    """Enable/disable low-overhead C++ PaiNN stage profiling."""
    if global_painn_potential.get() == NULL:
        raise RuntimeError("PaiNN potential is not active")
    if warmup_calls < 0:
        raise ValueError("warmup_calls must be non-negative")
    global_painn_potential.get().configure_profiling(enabled, warmup_calls)

def reset_painn_profiling():
    """Reset PaiNN profiling accumulators while preserving enable/warmup settings."""
    if global_painn_potential.get() == NULL:
        raise RuntimeError("PaiNN potential is not active")
    global_painn_potential.get().reset_profiling()

def get_painn_profile():
    """Return the current PaiNN C++ profiling snapshot as a Python dict."""
    if global_painn_potential.get() == NULL:
        raise RuntimeError("PaiNN potential is not active")
    import json
    cdef string payload = global_painn_potential.get().get_profile_json()
    return json.loads((<bytes>payload).decode("utf-8"))

def activate_painn_potential(model_path: str, num_species: int, hidden_channels: int, n_layers: int, num_rbf: int, cutoff: float, toxvaerd_alpha: float, device: str = "auto", precision: str = "float32", ordered_geometry_nodes: int = 0, ordered_geometry_head_layers: int = 0, ordered_geometry_head_width: int = 0, ordered_geometry_energy_scale_kj_mol: float = 0.0, ordered_geometry_head_only: bool = False, ordered_geometry_copies: int = 1, tel22_shared_geometry: bool = False, thermo_heads: bool = False, thermo_T0: float = 300.0, temperature_K: float = 0.0):
    """
    Attiva il potenziale globale PaiNN in ESPResSo.
    
    :param model_path: path of file with model weights (.pt)
    :param num_species: Number of species for embedding (e.g. 100)
    :param hidden_channels: Number of hidden channels of the PaiNN model
    :param n_layers: Number of layes for message passing
    :param num_rbf: Number of gaussian bases
    :param cutoff: cutoff radius
    :param device: "auto", "cpu", "cuda", "mps"
    :param precision: "float32" (production default) or "float64" (CPU diagnostic)
    :param thermo_heads: model with thermodynamic heads, U(T) = H - T S
    :param thermo_T0: reference temperature of the heads (K), as in training
    :param temperature_K: temperature of the ML potential (K), required with thermo_heads
    """
    global global_painn_potential
    
    cdef string cpp_path = model_path.encode('utf-8')
    cdef double c_cutoff = cutoff
    cdef double c_toxvaerd_alpha = toxvaerd_alpha
    cdef string cpp_device = device.encode('utf-8')
    cdef string cpp_precision = precision.encode('utf-8')
    cdef int c_num_species = num_species
    cdef int c_hidden_channels = hidden_channels
    cdef int c_n_layers = n_layers
    cdef int c_num_rbf = num_rbf
    cdef int c_ordered_geometry_nodes = ordered_geometry_nodes
    cdef int c_ordered_geometry_head_layers = ordered_geometry_head_layers
    cdef int c_ordered_geometry_head_width = ordered_geometry_head_width
    cdef double c_ordered_geometry_energy_scale_kj_mol = ordered_geometry_energy_scale_kj_mol
    cdef cpp_bool c_ordered_geometry_head_only = ordered_geometry_head_only
    cdef int c_ordered_geometry_copies = ordered_geometry_copies
    cdef cpp_bool c_tel22_shared_geometry = tel22_shared_geometry
    cdef cpp_bool c_thermo_heads = thermo_heads
    cdef double c_thermo_T0 = thermo_T0
    cdef double c_temperature_K = temperature_K
    if thermo_heads and not temperature_K > 0.0:
        raise ValueError("thermo_heads=True requires temperature_K > 0")
    
    global_painn_potential = make_shared[PaiNN_ML_Potential](
        cpp_path, c_num_species, c_hidden_channels, c_n_layers, c_num_rbf, c_cutoff, c_toxvaerd_alpha, c_ordered_geometry_nodes, c_ordered_geometry_head_layers, c_ordered_geometry_head_width, c_ordered_geometry_energy_scale_kj_mol, c_ordered_geometry_head_only, c_ordered_geometry_copies, c_tel22_shared_geometry, cpp_device, cpp_precision, c_thermo_heads, c_thermo_T0, c_temperature_K
    )
    
    print(f"PaiNN ML Potential attivato: {model_path} (cutoff={cutoff}, device={device}, precision={precision})")

def set_painn_temperature(temperature_K: float):
    """Change the temperature of a PaiNN potential with thermodynamic heads (K)."""
    if global_painn_potential.get() == NULL:
        raise RuntimeError("PaiNN potential is not active")
    if not global_painn_potential.get().has_thermo_heads():
        raise RuntimeError("the active PaiNN model has no thermodynamic heads")
    global_painn_potential.get().set_temperature(temperature_K)

def get_painn_temperature():
    """Temperature (K) of the active PaiNN potential (T0 for models without thermodynamic heads)."""
    if global_painn_potential.get() == NULL:
        raise RuntimeError("PaiNN potential is not active")
    return global_painn_potential.get().get_temperature()
