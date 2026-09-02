# cython: language_level=3

from libcpp.string cimport string
from libcpp.memory cimport make_shared, shared_ptr
from libcpp cimport bool as cpp_bool

# Dichiara l'interfaccia C++
cdef extern from "core/nonbonded_interactions/PaiNN_ML_Potential.hpp":
    cdef cppclass PaiNN_ML_Potential:
        PaiNN_ML_Potential(const string& model_path, int num_species, int hidden_channels, int n_layers, int num_rbf, double cutoff, double toxvaerd_alpha, int ordered_geometry_nodes, int ordered_geometry_head_layers, int ordered_geometry_head_width, double ordered_geometry_energy_scale_kj_mol, cpp_bool ordered_geometry_head_only, const string& device_str, const string& precision_str)
        double get_cutoff()
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

def activate_painn_potential(model_path: str, num_species: int, hidden_channels: int, n_layers: int, num_rbf: int, cutoff: float, toxvaerd_alpha: float, device: str = "auto", precision: str = "float32", ordered_geometry_nodes: int = 0, ordered_geometry_head_layers: int = 0, ordered_geometry_head_width: int = 0, ordered_geometry_energy_scale_kj_mol: float = 0.0, ordered_geometry_head_only: bool = False):
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
    
    global_painn_potential = make_shared[PaiNN_ML_Potential](
        cpp_path, c_num_species, c_hidden_channels, c_n_layers, c_num_rbf, c_cutoff, c_toxvaerd_alpha, c_ordered_geometry_nodes, c_ordered_geometry_head_layers, c_ordered_geometry_head_width, c_ordered_geometry_energy_scale_kj_mol, c_ordered_geometry_head_only, cpp_device, cpp_precision
    )
    
    print(f"PaiNN ML Potential attivato: {model_path} (cutoff={cutoff}, device={device}, precision={precision})")
