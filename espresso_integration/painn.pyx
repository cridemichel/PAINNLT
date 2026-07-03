# cython: language_level=3

from libcpp.string cimport string
from libcpp.memory cimport make_shared, shared_ptr

# Dichiara l'interfaccia C++
cdef extern from "core/nonbonded_interactions/PaiNN_ML_Potential.hpp":
    cdef cppclass PaiNN_ML_Potential:
        PaiNN_ML_Potential(const string& model_path, int num_atoms, int hidden_channels, int n_layers, int num_rbf, double cutoff, const string& device_str)
        double get_cutoff()
        double get_last_energy()
        
    cdef shared_ptr[PaiNN_ML_Potential] global_painn_potential

def get_painn_energy():
    """Restituisce l'ultima energia potenziale calcolata dal modello PaiNN in C++"""
    if global_painn_potential.get() != NULL:
        return global_painn_potential.get().get_last_energy()
    return 0.0

def activate_painn_potential(model_path: str, num_atoms: int, hidden_channels: int, n_layers: int, num_rbf: int, cutoff: float, device: str = "auto"):
    """
    Attiva il potenziale globale PaiNN in ESPResSo.
    
    :param model_path: Percorso al file dei pesi (.pt)
    :param num_atoms: Numero di atomi per l'embedding (es. 100)
    :param hidden_channels: Dimensione dei canali nascosti
    :param n_layers: Numero di layer del message passing
    :param num_rbf: Numero di basi gaussiane
    :param cutoff: Raggio di cutoff
    :param device: "auto", "cpu", "cuda", "mps"
    """
    global global_painn_potential
    
    cdef string cpp_path = model_path.encode('utf-8')
    cdef string cpp_device = device.encode('utf-8')
    cdef int c_num_atoms = num_atoms
    cdef int c_hidden_channels = hidden_channels
    cdef int c_n_layers = n_layers
    cdef int c_num_rbf = num_rbf
    cdef double c_cutoff = cutoff
    
    global_painn_potential = make_shared[PaiNN_ML_Potential](
        cpp_path, c_num_atoms, c_hidden_channels, c_n_layers, c_num_rbf, c_cutoff, cpp_device
    )
    
    print(f"PaiNN ML Potential attivato: {model_path} (cutoff={cutoff}, device={device})")
