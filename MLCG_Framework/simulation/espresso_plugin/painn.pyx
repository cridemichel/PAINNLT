# cython: language_level=3

from libcpp.string cimport string
from libcpp.memory cimport make_shared, shared_ptr

# Dichiara l'interfaccia C++
cdef extern from "core/nonbonded_interactions/PaiNN_ML_Potential.hpp":
    cdef cppclass PaiNN_ML_Potential:
        PaiNN_ML_Potential(const string& model_path, int num_species, int hidden_channels, int n_layers, int num_rbf, double cutoff, bool apply_envelope, bool use_bias, double toxvaerd_alpha, const string& device_str)
        double get_cutoff()
        double get_last_energy()
        
    cdef shared_ptr[PaiNN_ML_Potential] global_painn_potential

def get_painn_energy():
    """Restituisce l'ultima energia potenziale calcolata dal modello PaiNN in C++"""
    if global_painn_potential.get() != NULL:
        return global_painn_potential.get().get_last_energy()
    return 0.0

def activate_painn_potential(model_path: str, num_species: int, hidden_channels: int, n_layers: int, num_rbf: int, cutoff: float, apply_envelope: bool = False, use_bias: bool = False, toxvaerd_alpha: float = 0.0, device: str = "auto"):
    """
    Attiva il potenziale globale PaiNN in ESPResSo.
    
    :param model_path: path of file with model weights (.pt)
    :param num_species: Number of species for embedding (e.g. 100)
    :param hidden_channels: Number of hidden channels of the PaiNN model
    :param n_layers: Number of layes for message passing
    :param num_rbf: Number of gaussian bases
    :param cutoff: cutoff radius
    :param toxvaerd_alpha: toxvaerd alpha parameter (default 0.0)
    :param device: "auto", "cpu", "cuda", "mps"
    """
    global global_painn_potential
    
    cdef string cpp_path = model_path.encode('utf-8')
    cdef string cpp_device = device.encode('utf-8')
    cdef int c_num_species = num_species
    cdef int c_hidden_channels = hidden_channels
    cdef int c_n_layers = n_layers
    cdef int c_num_rbf = num_rbf
    cdef double c_cutoff = cutoff
    cdef bint c_apply_envelope = apply_envelope
    cdef bint c_use_bias = use_bias
    cdef double c_toxvaerd_alpha = toxvaerd_alpha
    
    global_painn_potential = make_shared[PaiNN_ML_Potential](
        cpp_path, c_num_species, c_hidden_channels, c_n_layers, c_num_rbf, c_cutoff, c_apply_envelope, c_use_bias, c_toxvaerd_alpha, cpp_device
    )
    
    print(f"PaiNN ML Potential attivato: {model_path} (cutoff={cutoff}, device={device})")
