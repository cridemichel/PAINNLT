"""TICA: analisi delle componenti indipendenti a tempo ritardato.

Implementata su numpy/scipy perche' deeptime e pyemma non sono installati e
l'algoritmo e' un problema agli autovalori generalizzato: aggiungere una
dipendenza per trenta righe non si giustifica.

Date le feature mean-free x_t, si risolve

    C_tau v = lambda C_0 v,
    C_0   = <x_t x_t^T>,
    C_tau = ((<x_t x_{t+tau}^T>) + trasposta) / 2

Le componenti con lambda maggiore sono i gradi di liberta' che decorrelano piu'
lentamente.  La simmetrizzazione di C_tau impone la reversibilita' e rende
reali gli autovalori.

Traiettorie multiple: le 10 copie di TEL22 condividono l'asse temporale ma
evolvono indipendentemente, quindi le covarianze si accumulano PER COPIA e si
sommano.  Concatenare le copie in un'unica serie introdurrebbe coppie
(t, t+tau) a cavallo del confine fra copie, cioe' correlazioni inesistenti.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg


class TICA:
    def __init__(self, lag: int = 10, n_components: int = 2, var_cutoff: float = 1e-8):
        self.lag = int(lag)
        self.n_components = int(n_components)
        self.var_cutoff = float(var_cutoff)

    def fit(self, trajs: list[np.ndarray]) -> "TICA":
        """trajs: lista di (T_i, n_feat), una per copia, in ordine temporale."""
        trajs = [np.asarray(t, dtype=np.float64) for t in trajs
                 if t.shape[0] > self.lag]
        if not trajs:
            raise ValueError(f"nessuna traiettoria piu' lunga del lag {self.lag}")
        n = trajs[0].shape[1]

        total = sum(t.shape[0] for t in trajs)
        self.mean_ = sum(t.sum(axis=0) for t in trajs) / total

        C0 = np.zeros((n, n))
        Ct = np.zeros((n, n))
        pairs = 0
        for t in trajs:
            x = t - self.mean_
            a, b = x[:-self.lag], x[self.lag:]
            # C0 sulle istanze che entrano in almeno una coppia, cosi' C0 e Ct
            # sono stimate sulla stessa popolazione.
            C0 += a.T @ a + b.T @ b
            Ct += a.T @ b
            pairs += a.shape[0]
        C0 /= 2.0 * pairs
        Ct /= pairs
        Ct = 0.5 * (Ct + Ct.T)

        # Sbiancamento: 231 feature da distanze a coppie sono fortemente
        # ridondanti, quindi C0 e' quasi singolare e il problema generalizzato
        # va risolto nel sottospazio a varianza non trascurabile.
        w, V = np.linalg.eigh(C0)
        keep = w > self.var_cutoff * float(w.max())
        if keep.sum() < self.n_components:
            raise ValueError("rango di C0 insufficiente per le componenti richieste")
        W = V[:, keep] / np.sqrt(w[keep])
        lam, U = scipy.linalg.eigh(W.T @ Ct @ W)
        order = np.argsort(lam)[::-1]
        self.eigenvalues_ = lam[order][:self.n_components]
        self.components_ = (W @ U[:, order])[:, :self.n_components]
        self.rank_ = int(keep.sum())
        # Tempi di rilassamento impliciti, in unita' di lag.  Diagnostica:
        # se il primo e' dell'ordine del lag, il lag e' troppo lungo.
        with np.errstate(divide="ignore", invalid="ignore"):
            self.timescales_ = -self.lag / np.log(np.abs(self.eigenvalues_))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_
