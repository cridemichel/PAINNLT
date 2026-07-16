import numpy as np

def kabsch_align(P, Q):
    H = np.dot(P.T, Q)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R

# Test
P = np.array([[1,0,0], [0,1,0], [0,0,1]])
# Rotate P by 90 degrees around Z
theta = np.pi/2
R_true = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
Q = (R_true @ P.T).T

R = kabsch_align(P, Q)
P_aligned = (R @ P.T).T
print("R error:", np.linalg.norm(R - R_true))
print("P_aligned error:", np.linalg.norm(P_aligned - Q))
