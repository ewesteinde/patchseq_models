from typing import Dict, Optional

import jax.numpy as jnp
from jax.lax import select
from jaxley.channels import Channel
from jaxley.solver_gate import (
    exponential_euler,
    save_exp,
    solve_gate_exponential,
    solve_inf_gate_exponential,
)

#from ..utils import efun

__all__ = [
    "NaV", # unverified, uses a 12-state Markov model instead of the simpler Hodgkin-Huxley model. translate w/ Claude need to check if this is correct
    "NaTs",
    "Nap",
    "Kv2like",
    "K_P",
    "K_T",
    "Kd", # functional influence is contained in K_P and K_T
    "SKE2", # SK
    "SKv3_1", # Kv3_1
    "M",
    "Im_v2",
    "CaHVA",
    "CaLVA",
    "CaPump_NEURON",
    "CaPump",
    "CaNernstReversal",
    "H",
]

# need to add the following channels:
# NaV - done but unverified, uses a 12-state Markov model instead of the simpler Hodgkin-Huxley model
# Kd - done, unverified
# Kv2like - done, unverified
# Im_v2 -done, unverified
############################
## Sodium channels:       ##
## Nats, Nap, NaV ##
############################


class NaV(Channel):
    ### TEST BUT PROB NOT USE AT THIS POINT ###
    """Mouse sodium current from Carter et al. (2012).
    Based on 37 degC recordings from mouse hippocampal CA1 pyramids.
    Implements a 12-state Markov model."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        
        # Parameters from PARAMETER block
        self.channel_params = {
            f"{prefix}_gNaV": 0.015,  # S/cm^2
            "eNa": 53.0,  # mV
            "celsius": 37.0,  # Temperature in Celsius
            
            # Kinetic parameters
            "Con": 0.01,   # /ms - closed -> inactivated transitions
            "Coff": 40.0,  # /ms - inactivated -> closed transitions
            "Oon": 8.0,    # /ms - open -> Ineg transition
            "Ooff": 0.05,  # /ms - Ineg -> open transition
            "alpha": 400.0, # /ms
            "beta": 12.0,  # /ms
            "gamma": 250.0, # /ms - opening
            "delta": 60.0, # /ms - closing
            
            # Factors
            "alfac": 2.51,
            "btfac": 5.32,
            
            # Voltage dependence
            "x1": 24.0,   # mV - Vdep of activation
            "x2": -24.0,  # mV - Vdep of deactivation
        }

        # Initialize all states (C1-C5, O, I1-I6)
        self.channel_states = {
            f"{prefix}_C1": 1.0,  # Initial value for first closed state
            f"{prefix}_C2": 0.0,
            f"{prefix}_C3": 0.0,
            f"{prefix}_C4": 0.0,
            f"{prefix}_C5": 0.0,
            f"{prefix}_O": 0.0,   # Open state
            f"{prefix}_I1": 0.0,
            f"{prefix}_I2": 0.0,
            f"{prefix}_I3": 0.0,
            f"{prefix}_I4": 0.0,
            f"{prefix}_I5": 0.0,
            f"{prefix}_I6": 0.0,  # Inactivated state connected to O
        }
        
        self.current_name = "i_Na"
        self.META = {
            "reference": "Carter et al. (2012)",
            "species": "mouse",
            "cell_type": "hippocampal CA1 pyramidal cell",
            "ion": "Na",
        }

    def compute_rates(self, v: float, params: Dict[str, jnp.ndarray]):
        """Compute all transition rates based on voltage and parameters."""
        qt = 2.3**((params["celsius"] - 37.0) / 10.0)
        
        # Forward rates (equivalent to f__ variables in mod file)
        f01 = qt * 4 * params["alpha"] * jnp.exp(v/params["x1"])
        f02 = qt * 3 * params["alpha"] * jnp.exp(v/params["x1"])
        f03 = qt * 2 * params["alpha"] * jnp.exp(v/params["x1"])
        f04 = qt * 1 * params["alpha"] * jnp.exp(v/params["x1"])
        f0O = qt * params["gamma"]
        
        f11 = qt * 4 * params["alpha"] * params["alfac"] * jnp.exp(v/params["x1"])
        f12 = qt * 3 * params["alpha"] * params["alfac"] * jnp.exp(v/params["x1"])
        f13 = qt * 2 * params["alpha"] * params["alfac"] * jnp.exp(v/params["x1"])
        f14 = qt * 1 * params["alpha"] * params["alfac"] * jnp.exp(v/params["x1"])
        f1n = qt * params["gamma"]
        
        fi1 = qt * params["Con"]
        fi2 = qt * params["Con"] * params["alfac"]
        fi3 = qt * params["Con"] * params["alfac"]**2
        fi4 = qt * params["Con"] * params["alfac"]**3
        fi5 = qt * params["Con"] * params["alfac"]**4
        fin = qt * params["Oon"]
        
        # Backward rates (equivalent to b__ variables in mod file)
        b01 = qt * 1 * params["beta"] * jnp.exp(v/params["x2"])
        b02 = qt * 2 * params["beta"] * jnp.exp(v/params["x2"])
        b03 = qt * 3 * params["beta"] * jnp.exp(v/params["x2"])
        b04 = qt * 4 * params["beta"] * jnp.exp(v/params["x2"])
        b0O = qt * params["delta"]
        
        b11 = qt * 1 * params["beta"] * jnp.exp(v/params["x2"]) / params["btfac"]
        b12 = qt * 2 * params["beta"] * jnp.exp(v/params["x2"]) / params["btfac"]
        b13 = qt * 3 * params["beta"] * jnp.exp(v/params["x2"]) / params["btfac"]
        b14 = qt * 4 * params["beta"] * jnp.exp(v/params["x2"]) / params["btfac"]
        b1n = qt * params["delta"]
        
        bi1 = qt * params["Coff"]
        bi2 = qt * params["Coff"] / params["btfac"]
        bi3 = qt * params["Coff"] / (params["btfac"]**2)
        bi4 = qt * params["Coff"] / (params["btfac"]**3)
        bi5 = qt * params["Coff"] / (params["btfac"]**4)
        bin = qt * params["Ooff"]
        
        return {
            "f01": f01, "f02": f02, "f03": f03, "f04": f04, "f0O": f0O,
            "f11": f11, "f12": f12, "f13": f13, "f14": f14, "f1n": f1n,
            "fi1": fi1, "fi2": fi2, "fi3": fi3, "fi4": fi4, "fi5": fi5, "fin": fin,
            "b01": b01, "b02": b02, "b03": b03, "b04": b04, "b0O": b0O,
            "b11": b11, "b12": b12, "b13": b13, "b14": b14, "b1n": b1n,
            "bi1": bi1, "bi2": bi2, "bi3": bi3, "bi4": bi4, "bi5": bi5, "bin": bin
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of the Markov model using exponential integration."""
        prefix = self._name
        rates = self.compute_rates(voltages, params)
        
        # Current states
        C1, C2, C3, C4, C5 = (
            u[f"{prefix}_C1"], u[f"{prefix}_C2"], u[f"{prefix}_C3"],
            u[f"{prefix}_C4"], u[f"{prefix}_C5"]
        )
        O = u[f"{prefix}_O"]
        I1, I2, I3, I4, I5, I6 = (
            u[f"{prefix}_I1"], u[f"{prefix}_I2"], u[f"{prefix}_I3"],
            u[f"{prefix}_I4"], u[f"{prefix}_I5"], u[f"{prefix}_I6"]
        )
        
        # Update each state based on incoming and outgoing fluxes
        # This is a simplified explicit Euler integration - could be improved
        dC1 = (
            -rates["f01"]*C1 + rates["b01"]*C2 
            -rates["fi1"]*C1 + rates["bi1"]*I1
        )
        dC2 = (
            rates["f01"]*C1 - rates["b01"]*C2 
            -rates["f02"]*C2 + rates["b02"]*C3
            -rates["fi2"]*C2 + rates["bi2"]*I2
        )
        dC3 = (
            rates["f02"]*C2 - rates["b02"]*C3
            -rates["f03"]*C3 + rates["b03"]*C4
            -rates["fi3"]*C3 + rates["bi3"]*I3
        )
        dC4 = (
            rates["f03"]*C3 - rates["b03"]*C4
            -rates["f04"]*C4 + rates["b04"]*C5
            -rates["fi4"]*C4 + rates["bi4"]*I4
        )
        dC5 = (
            rates["f04"]*C4 - rates["b04"]*C5
            -rates["f0O"]*C5 + rates["b0O"]*O
            -rates["fi5"]*C5 + rates["bi5"]*I5
        )
        dO = (
            rates["f0O"]*C5 - rates["b0O"]*O
            -rates["fin"]*O + rates["bin"]*I6
        )
        dI1 = (
            rates["fi1"]*C1 - rates["bi1"]*I1
            -rates["f11"]*I1 + rates["b11"]*I2
        )
        dI2 = (
            rates["fi2"]*C2 - rates["bi2"]*I2
            + rates["f11"]*I1 - rates["b11"]*I2
            - rates["f12"]*I2 + rates["b12"]*I3
        )
        dI3 = (
            rates["fi3"]*C3 - rates["bi3"]*I3
            + rates["f12"]*I2 - rates["b12"]*I3
            - rates["f13"]*I3 + rates["b13"]*I4
        )
        dI4 = (
            rates["fi4"]*C4 - rates["bi4"]*I4
            + rates["f13"]*I3 - rates["b13"]*I4
            - rates["f14"]*I4 + rates["b14"]*I5
        )
        dI5 = (
            rates["fi5"]*C5 - rates["bi5"]*I5
            + rates["f14"]*I4 - rates["b14"]*I5
            - rates["f1n"]*I5 + rates["b1n"]*I6
        )
        dI6 = (
            rates["fin"]*O - rates["bin"]*I6
            + rates["f1n"]*I5 - rates["b1n"]*I6
        )
        
        # Simple Euler integration
        return {
            f"{prefix}_C1": C1 + dt * dC1,
            f"{prefix}_C2": C2 + dt * dC2,
            f"{prefix}_C3": C3 + dt * dC3,
            f"{prefix}_C4": C4 + dt * dC4,
            f"{prefix}_C5": C5 + dt * dC5,
            f"{prefix}_O": O + dt * dO,
            f"{prefix}_I1": I1 + dt * dI1,
            f"{prefix}_I2": I2 + dt * dI2,
            f"{prefix}_I3": I3 + dt * dI3,
            f"{prefix}_I4": I4 + dt * dI4,
            f"{prefix}_I5": I5 + dt * dI5,
            f"{prefix}_I6": I6 + dt * dI6,
        }

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        O = u[f"{prefix}_O"]  # Open state probability
        na_cond = params[f"{prefix}_gNaV"]
        current = na_cond * O * (voltages - params["eNa"])
        return current

    # def init_state(self, states, voltages, params, delta_t):
    #     """Initialize the state at fixed point of Markov dynamics."""
    #     # For simplicity, start in C1 state (could be improved by solving steady-state)

    #     import jax.numpy as jnp
    #     from jax.numpy.linalg import solve
    #     prefix = self._name
    #     return {
    #         f"{prefix}_C1": 1.0,
    #         f"{prefix}_C2": 0.0,
    #         f"{prefix}_C3": 0.0,
    #         f"{prefix}_C4": 0.0,
    #         f"{prefix}_C5": 0.0,
    #         f"{prefix}_O": 0.0,
    #         f"{prefix}_I1": 0.0,
    #         f"{prefix}_I2": 0.0,
    #         f"{prefix}_I3": 0.0,
    #         f"{prefix}_I4": 0.0,
    #         f"{prefix}_I5": 0.0,
    #         f"{prefix}_I6": 0.0,
    #     }
    
    # # Alternative to the above, could use a function to solve the steady-state
    # def init_state(self, states, voltages, params, delta_t):
    #     """Initialize the state by solving for steady-state of the Markov chain."""

    #     prefix = self._name
    #     rates = self.compute_rates(voltages, params)
        
    #     # Build rate matrix Q (12x12) for states [C1,C2,C3,C4,C5,O,I1,I2,I3,I4,I5,I6]
    #     # Each row represents the differential equation for that state
    #     Q = jnp.zeros((12, 12))
        
    #     # C1 row
    #     Q[0,0] = -(rates["f01"] + rates["fi1"])
    #     Q[0,1] = rates["b01"]
    #     Q[0,6] = rates["bi1"]
        
    #     # C2 row
    #     Q[1,0] = rates["f01"]
    #     Q[1,1] = -(rates["b01"] + rates["f02"] + rates["fi2"])
    #     Q[1,2] = rates["b02"]
    #     Q[1,7] = rates["bi2"]
        
    #     # C3 row
    #     Q[2,1] = rates["f02"]
    #     Q[2,2] = -(rates["b02"] + rates["f03"] + rates["fi3"])
    #     Q[2,3] = rates["b03"]
    #     Q[2,8] = rates["bi3"]
        
    #     # C4 row
    #     Q[3,2] = rates["f03"]
    #     Q[3,3] = -(rates["b03"] + rates["f04"] + rates["fi4"])
    #     Q[3,4] = rates["b04"]
    #     Q[3,9] = rates["bi4"]
        
    #     # C5 row
    #     Q[4,3] = rates["f04"]
    #     Q[4,4] = -(rates["b04"] + rates["f0O"] + rates["fi5"])
    #     Q[4,5] = rates["b0O"]
    #     Q[4,10] = rates["bi5"]
        
    #     # O row
    #     Q[5,4] = rates["f0O"]
    #     Q[5,5] = -(rates["b0O"] + rates["fin"])
    #     Q[5,11] = rates["bin"]
        
    #     # I1 row
    #     Q[6,0] = rates["fi1"]
    #     Q[6,6] = -(rates["bi1"] + rates["f11"])
    #     Q[6,7] = rates["b11"]
        
    #     # I2 row
    #     Q[7,1] = rates["fi2"]
    #     Q[7,6] = rates["f11"]
    #     Q[7,7] = -(rates["bi2"] + rates["b11"] + rates["f12"])
    #     Q[7,8] = rates["b12"]
        
    #     # I3 row
    #     Q[8,2] = rates["fi3"]
    #     Q[8,7] = rates["f12"]
    #     Q[8,8] = -(rates["bi3"] + rates["b12"] + rates["f13"])
    #     Q[8,9] = rates["b13"]
        
    #     # I4 row
    #     Q[9,3] = rates["fi4"]
    #     Q[9,8] = rates["f13"]
    #     Q[9,9] = -(rates["bi4"] + rates["b13"] + rates["f14"])
    #     Q[9,10] = rates["b14"]
        
    #     # I5 row
    #     Q[10,4] = rates["fi5"]
    #     Q[10,9] = rates["f14"]
    #     Q[10,10] = -(rates["bi5"] + rates["b14"] + rates["f1n"])
    #     Q[10,11] = rates["b1n"]
        
    #     # I6 row
    #     Q[11,5] = rates["fin"]
    #     Q[11,10] = rates["f1n"]
    #     Q[11,11] = -(rates["bin"] + rates["b1n"])
        
    #     # Replace last row with conservation equation (sum of probabilities = 1)
    #     Q = Q.at[11].set(jnp.ones(12))
    #     b = jnp.zeros(12)
    #     b = b.at[11].set(1.0)
        
    #     # Solve Q*x = b for steady state probabilities
    #     x = solve(Q, b)
        
    #     # Return the steady state values
    #     return {
    #         f"{prefix}_C1": x[0],
    #         f"{prefix}_C2": x[1],
    #         f"{prefix}_C3": x[2],
    #         f"{prefix}_C4": x[3],
    #         f"{prefix}_C5": x[4],
    #         f"{prefix}_O": x[5],
    #         f"{prefix}_I1": x[6],
    #         f"{prefix}_I2": x[7],
    #         f"{prefix}_I3": x[8],
    #         f"{prefix}_I4": x[9],
    #         f"{prefix}_I5": x[10],
    #         f"{prefix}_I6": x[11]
    #     }

class NaTs(Channel):
    """Transient sodium current from Colbert and Pan, 2002. Based on mod file NaTs.mod from Allen perisomatic model."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNaTs": 0.00001,  # S/cm^2
            "eNa": 53.0,  # mV
            "celsius": 34.0,  # Temperature in Celsius
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_Na"
        self.META = {
            "reference": "Colbert and Pan (2002)",
            "doi": "https://doi.org/10.1038/nn0602-857",
            "species": "rat",
            "cell_type": "layer 5 pyramidal cell",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/NaTa_t.mod",
            "ion": "Na",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages, params))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages, params))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        # I think the same as BREAKPOINT in mod files
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        na_cond = params[f"{prefix}_gNaTs"]
        current = na_cond * (ms**3) * hs * (voltages - params["eNa"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages, params)
        h_inf, _ = self.h_gate(voltages, params)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }
    
    @staticmethod
    def vtrap(x, y):
        """Numerical function to avoid division by zero errors."""
        # if jnp.abs(x / y) < 1e-6:
        #     return y * (1 - x / (2 * y))
        # else:
        #     return x / (jnp.exp(x / y) - 1)
        # normal booleans don't work in jax, need to replace above if statement arguments with jnp functions
        return jnp.where(jnp.abs(x / y) < 1e-6, y * (1 - x / (2 * y)), x / (jnp.exp(x / y) - 1))
        
    # functions below are the PROCEDURE rates in the mod file
    # NaTa_t.mod does this differently than NaTs.mod 
    @staticmethod
    def m_gate(v, params: Dict[str, jnp.ndarray]):
        """Voltage-dependent dynamics for the m gating variable."""
        # alphaF, betaF, vhalf, k for m & h gating variables from mod file hardcoded here
        #qt = 2.3 ** ((34 - 21) / 10)
        qt = 2.3 ** ((params["celsius"]-23)/10)
        #alpha = (0.182 * (v + 38 + 1e-6)) / (1 - save_exp(-(v + 38 + 1e-6) / 6))
        #beta = (0.124 * (-v - 38 + 1e-6)) / (1 - save_exp(-(-v - 38 + 1e-6) / 6))
        alpha = 0.182 * NaTs.vtrap(-(v - -40), 6)
        beta = 0.124 * NaTs.vtrap((v - -40), 6)
        
        m_inf = alpha / (alpha + beta)
        tau_m = 1 / (alpha + beta) / qt
        return m_inf, tau_m

    @staticmethod
    def h_gate(v, params: Dict[str, jnp.ndarray]):
        """Voltage-dependent dynamics for the h gating variable."""
        #qt = 2.3 ** ((34 - 21) / 10)
        qt = 2.3 ** ((params["celsius"]-23)/10)
        #alpha = (-0.015 * (v + 66 + 1e-6)) / (1 - save_exp((v + 66 + 1e-6) / 6))
        #beta = (-0.015 * (-v - 66 + 1e-6)) / (1 - save_exp((-v - 66 + 1e-6) / 6))
        alpha = 0.015 * NaTs.vtrap(v - -66, 6)
        beta = 0.015 * NaTs.vtrap(-(v - -66), 6)

        h_inf = alpha / (alpha + beta)
        tau_h = 1 / (alpha + beta) / qt
        return h_inf, tau_h
    

class Nap(Channel):
    """Persistent sodium current from Magistretti & Alonso 1999.

    Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 23.
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNap": 0.00001,  # S/cm^2
            "eNa": 53,  # mV
            "celsius": 34
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_Na"
        self.META = {
            "reference": "Magistretti and Alonso (1999)",
            "doi": "https://doi.org/10.1085/jgp.114.4.491",
            "species": "rat",
            "cell_type": "entorhinal cortex layer-II principal neurons",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Nap_Et2.mod",
            "ion": "Na",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        #m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        m_new = self.m_gate(voltages)
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages, params))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        na_cond = params[f"{prefix}_gNap"]
        #current = na_cond * (ms**3) * hs * (voltages - params["eNa"])
        current = na_cond * ms * hs * (voltages - params["eNa"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages, params)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def vtrap(x, y):
        """Numerical function to avoid division by zero errors."""
        # normal booleans don't work in jax, need to replace above if statement arguments with jnp functions
        return jnp.where(jnp.abs(x / y) < 1e-6, y * (1 - x / y / 2), x / (jnp.exp(x / y) - 1))
    
    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        #qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        # alpha = (0.182 * (v + 38 + 1e-6)) / (1 - save_exp(-(v + 38 + 1e-6) / 6))
        # beta = (0.124 * (-v - 38 + 1e-6)) / (1 - save_exp(-(-v - 38 + 1e-6) / 6))
        # tau_m = 6 / (alpha + beta) / qt

        m_inf = 1.0/(1+jnp.exp((v- -52.6)/-4.6))
        #m_inf = 1.0 / (1 + save_exp((v + 52.7) / -4.6))
        return m_inf

    @staticmethod
    def h_gate(v, params: Dict[str, jnp.ndarray]):
        """Voltage-dependent dynamics for the h gating variable."""
        #qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        qt = 2.3 ** ((params["celsius"]-23)/10)
        # alpha = (-2.88e-6 * (v + 17 + 1e-6)) / (1 - save_exp((v + 17 + 1e-6) / 4.63))
        # beta = (6.94e-6 * (v + 64.4 + 1e-6)) / (1 - save_exp((-(v + 64.4) + 1e-6) / 6))
        # tau_h = 1 / (alpha + beta) / qt
        # h_inf = 1.0 / (1 + save_exp((v + 48.8) / 10))

        h_inf = 1.0/(1+jnp.exp((v- -48.8)/10))
        alpha = 2.88e-6 * Nap.vtrap(v + 17, 4.63)
        beta = 6.94e-6 * Nap.vtrap(-(v + 64.4), 2.63)
        tau_h = (1/(alpha + beta))/qt

        return h_inf, tau_h


###########################
## Potassium channels    ##
## K_P, K_T, SKE2, SKv3_1, M
###########################

class Kd(Channel):
    """Delayed rectifier potassium channel based on Foust et al. (2011)."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        
        # Parameters from PARAMETER block
        self.channel_params = {
            f"{prefix}_gKd": 0.00001,  # S/cm^2
            "ek": -80.0,  # mV
            "celsius": 23.0,  # Temperature in Celsius
        }
        
        # Initialize states (m, h)
        self.channel_states = {
            f"{prefix}_m": 0.0,  # Activation gate
            f"{prefix}_h": 0.0,  # Inactivation gate
        }
        
        self.current_name = "i_K"
        self.META = {
            "reference": "Foust et al. (2011)",
            "ion": "K",
        }

    def compute_rates(self, v: float, params: Dict[str, jnp.ndarray]):
        """Compute all rate parameters for channel gates."""
        qt = 2.3**((params["celsius"] - 23) / 10)
        
        # m gate (activation) rates
        mInf = 1 - 1 / (1 + jnp.exp((v - (-43)) / 8))
        mTau = 1.0  # constant
        
        # h gate (inactivation) rates
        hInf = 1 / (1 + jnp.exp((v - (-67)) / 7.3))
        hTau = 1500.0  # constant
        
        return {
            "mInf": mInf, "mTau": mTau,
            "hInf": hInf, "hTau": hTau
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables using exponential integration."""
        prefix = self._name
        rates = self.compute_rates(voltages, params)
        
        # Current states
        m = u[f"{prefix}_m"]
        h = u[f"{prefix}_h"]
        
        # Update using exponential integration
        m_new = solve_inf_gate_exponential(m, dt, rates["mInf"], rates["mTau"])
        h_new = solve_inf_gate_exponential(h, dt, rates["hInf"], rates["hTau"])
        
        return {
            f"{prefix}_m": m_new,
            f"{prefix}_h": h_new
        }

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        h = u[f"{prefix}_h"]
        
        # Compute conductance: g = gbar * m * h
        k_cond = params[f"{prefix}_gKd"] * m * h
        current = k_cond * (voltages - params["ek"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at steady-state values."""
        prefix = self._name
        rates = self.compute_rates(voltages, params)
        return {
            f"{prefix}_m": rates["mInf"],
            f"{prefix}_h": rates["hInf"]
        }

class Kv2like(Channel):
    """Kv2-like potassium channel adapted from Keren et al. 2005.
    Adjusted to match guangxitoxin-sensitive current in mouse CA1 pyramids (Liu and Bean 2014)."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        
        # Parameters from PARAMETER block
        self.channel_params = {
            f"{prefix}_gKv2like": 0.00001,  # S/cm^2
            "ek": -80.0,  # mV
            "celsius": 34.0,  # Temperature in Celsius
        }
        
        # Initialize states (m, h1, h2)
        self.channel_states = {
            f"{prefix}_m": 0.0,  # Activation gate
            f"{prefix}_h1": 0.0,  # First inactivation gate
            f"{prefix}_h2": 0.0,  # Second inactivation gate
        }
        
        self.current_name = "i_K"
        self.META = {
            "reference": ["Keren et al. (2005)", "Liu and Bean (2014)"],
            "species": "mouse",
            "cell_type": "CA1 pyramidal cell",
            "ion": "K",
        }
    
    @staticmethod
    def vtrap(x, y):
        """Numerical function to avoid division by zero errors in rate equations."""
        # if abs(x/y) < 1e-6:
        #     return y * (1 - x/y/2)
        # else:
        #     return x / (jnp.exp(x/y) - 1)
        return jnp.where(
            jnp.abs(x/y) < 1e-6,
            y * (1 - x/y/2),
            x / (jnp.exp(x/y) - 1)
        )

    def compute_rates(self, v: float, params: Dict[str, jnp.ndarray]):
        """Compute all rate parameters for channel gates."""
        qt = 2.3**((params["celsius"] - 21) / 10)
        
        # m gate (activation) rates
        mAlpha = 0.12 * self.vtrap(-(v - 43), 11.0)
        mBeta = 0.02 * jnp.exp(-(v + 1.27) / 120)
        mInf = mAlpha / (mAlpha + mBeta)
        mTau = 2.5 * (1 / (qt * (mAlpha + mBeta)))
        
        # h gate (inactivation) rates
        hInf = 1 / (1 + jnp.exp((v + 58) / 11))
        h1Tau = (360 + (1010 + 23.7 * (v + 54)) * jnp.exp(-((v + 75) / 48)**2)) / qt
        h2Tau_raw = (2350 + 1380 * jnp.exp(-0.011 * v) - 210 * jnp.exp(-0.03 * v)) / qt
        h2Tau = jnp.where(h2Tau_raw < 0, 1e-3, h2Tau_raw)
        
        return {
            "mInf": mInf, "mTau": mTau,
            "hInf": hInf, "h1Tau": h1Tau, "h2Tau": h2Tau
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables using exponential integration."""
        prefix = self._name
        rates = self.compute_rates(voltages, params)
        
        # Current states
        m = u[f"{prefix}_m"]
        h1 = u[f"{prefix}_h1"]
        h2 = u[f"{prefix}_h2"]
        
        # Update using exponential integration
        m_new = solve_inf_gate_exponential(m, dt, rates["mInf"], rates["mTau"])
        h1_new = solve_inf_gate_exponential(h1, dt, rates["hInf"], rates["h1Tau"])
        h2_new = solve_inf_gate_exponential(h2, dt, rates["hInf"], rates["h2Tau"])
        
        return {
            f"{prefix}_m": m_new,
            f"{prefix}_h1": h1_new,
            f"{prefix}_h2": h2_new
        }

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        h1 = u[f"{prefix}_h1"]
        h2 = u[f"{prefix}_h2"]
        
        # Compute conductance: g = gbar * m^2 * (0.5*h1 + 0.5*h2)
        k_cond = params[f"{prefix}_gKv2like"] * (m * m) * (0.5 * h1 + 0.5 * h2)
        current = k_cond * (voltages - params["ek"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at steady-state values."""
        prefix = self._name
        rates = self.compute_rates(voltages, params)
        return {
            f"{prefix}_m": rates["mInf"],
            f"{prefix}_h1": rates["hInf"],
            f"{prefix}_h2": rates["hInf"]
        }

class K_P(Channel):
    """Persistent component of the K current from Korngreen and Sakmann, 2000, adjusted for junction potential."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK_P": 0.00001,  # S/cm^2
            "eK": -107.0,  # mV, from l5pc/config/parameters.json
            'vshift': 0.0,
            'celsius': 34.0,
            'TauF': 1.0
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Korngreen and Sakmann (2000)",
            "doi": "https://doi.org/10.1111/j.1469-7793.2000.00621.x",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/K_Pst.mod",
            "species": "rat",
            "cell_type": "layer 5 pyramidal cell",
            "note": "Shifted -10 mV to correct for junction potential, rates corrected with Q10",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables using cnexp integration method."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages,params))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages,params))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gK_P"] * (ms**2) * hs
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages,params)
        h_inf, _ = self.h_gate(voltages,params)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v,params: Dict[str, jnp.ndarray]):
        """Voltage-dependent dynamics for the m gating variable, adjusted for junction potential."""
        #qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        qt = 2.3 ** ((params['celsius']-21)/10)
        #v_adjusted = v + 10  # Adjust for junction potential
        #m_inf = 1 / (1 + save_exp(-(v_adjusted + 1) / 12))
        m_inf =  1 / (1 + jnp.exp(-(v - (-14.3 + params['vshift'])) / 14.6))
        tau_m = jnp.where(v < (-50 + params['vshift']), params['TauF'] * (1.25+175.03*jnp.exp(-(v - params['vshift']) * -0.026))/qt, params['TauF'] * (1.25+13*jnp.exp(-(v - params['vshift']) * 0.026))/qt)
        # See here for documentation of `select` vs `cond`:
        # https://github.com/google/jax/issues/7934
        # tau_m = select(
        #     v_adjusted < jnp.asarray(-50.0),
        #     (1.25 + 175.03 * save_exp(v_adjusted * 0.026)) / qt,
        #     (1.25 + 13 * save_exp(-v_adjusted * 0.026)) / qt,
        # )
        return m_inf, tau_m

    @staticmethod
    def h_gate(v, params: Dict[str, jnp.ndarray]):
        """Voltage-dependent dynamics for the h gating variable, adjusted for junction potential."""
        #qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        qt = 2.3 ** ((params['celsius']-21)/10)
        # v_adjusted = v + 10  # Adjust for junction potential
        # h_inf = 1 / (1 + save_exp(-(v_adjusted + 54) / -11))
        # tau_h = (
        #     360
        #     + (1010 + 24 * (v_adjusted + 55))
        #     * save_exp(-(((v_adjusted + 75) / 48) ** 2))
        # ) / qt

        h_inf =  1/(1 + jnp.exp(-(v - (-54 + params['vshift']))/-11))
        tau_h =  (360+(1010+24*(v - (-55 + params['vshift'])))*jnp.exp(-((v - (-75 + params['vshift']))/48)**2))/qt
        return h_inf, tau_h


class K_T(Channel):
    """Transient component of the K current from Korngreen and Sakmann, 2000, adjusted for junction potential."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK_T": 0.00001,  # S/cm^2
            "eK": -107.0,  # mV
            "celsius": 34.0,
            "vshift": 0.0,
            "mTauF": 1.0,
            "hTauF": 1.0
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
            f"{prefix}_h": 0.1,  # Initial value for h gating variable
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Korngreen and Sakmann (2000)",
            "doi": "https://doi.org/10.1111/j.1469-7793.2000.00621.x",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/K_Tst.mod",
            "species": "rat",
            "cell_type": "layer 5 pyramidal cell",
            "note": "Shifted -10 mV to correct for junction potential, rates corrected with Q10",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables using cnexp integration method."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages,params))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages,params))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        k_cond = params[f"{prefix}_gK_T"] * (ms**4) * hs
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages,params)
        h_inf, _ = self.h_gate(voltages,params)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v,params: Dict[str, jnp.ndarray]):
        """Voltage-dependent dynamics for the m gating variable, adjusted for junction potential."""
        #qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        qt = 2.3**((params['celsius']-21)/10)
        # v_adjusted = v + 10  # Adjust for junction potential
        # m_inf = 1 / (1 + save_exp(-(v_adjusted + 0) / 19))
        # tau_m = (0.34 + 0.92 * save_exp(-(((v_adjusted + 71) / 59) ** 2))) / qt

        m_inf =  1/(1 + jnp.exp(-(v - (-47 + params['vshift'])) / 29))
        tau_m =  (0.34 + params['mTauF'] * 0.92*jnp.exp(-((v+71-params['vshift'])/59)**2))/qt
        return m_inf, tau_m

    @staticmethod
    def h_gate(v,params: Dict[str, jnp.ndarray]):
        """Voltage-dependent dynamics for the h gating variable, adjusted for junction potential."""
        #qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        qt = 2.3**((params['celsius']-21)/10)
        # v_adjusted = v + 10  # Adjust for junction potential
        # h_inf = 1 / (1 + save_exp(-(v_adjusted + 66) / -10))
        # tau_h = (8 + 49 * save_exp(-(((v_adjusted + 73) / 23) ** 2))) / qt

        h_inf =  1/(1 + jnp.exp(-(v+66-params['vshift'])/-10))
        tau_h =  (8 + params['hTauF'] * 49*jnp.exp(-((v+73-params['vshift'])/23)**2))/qt
        return h_inf, tau_h


class SKE2(Channel):
    """SK-type calcium-activated potassium current from Kohler et al., 1996."""
    # actually the same as the Allen SK channel

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gSKE2": 0.000001,  # mho/cm^2
            "eK": -107.0,  # mV, assuming eK for potassium
        }
        self.channel_states = {
            f"{prefix}_z": 0.0,  # Initial value for z gating variable
            f"CaCon_i": 5e-05,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Kohler et al., 1996",
            "doi": "https://doi.org/10.1126/science.273.5282.1709",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/SK_E2.mod",
            "species": ["rat", "human"],
            "cell_type": "unknown",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variable z."""
        prefix = self._name
        zs = u[f"{prefix}_z"]
        cai = u["CaCon_i"]  # intracellular calcium concentration, from CaPump
        z_new = solve_inf_gate_exponential(zs, dt, *self.z_gate(cai))
        return {f"{prefix}_z": z_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        z = u[f"{prefix}_z"]
        k_cond = params[f"{prefix}_gSKE2"] * z
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        cai = 5e-05  # Initial value for intracellular calcium concentration
        z_inf, _ = self.z_gate(cai)
        return {f"{prefix}_z": z_inf}

    @staticmethod
    def z_gate(cai):
        """Dynamics for the z gating variable, dependent on intracellular calcium concentration."""
        cai = select(
            cai < jnp.asarray(1e-7),
            cai + 1e-7,
            cai,
        )
        z_inf = 1 / (1 + (0.00043 / cai) ** 4.8)
        tau_z = 1.0  # tau_z is fixed at 1 ms
        return z_inf, tau_z


class SKv3_1(Channel):
    """Shaw-related potassium channel family SKv3_1 from The EMBO Journal, 1992."""
    # actually the same as the Allen Kv3_1 channel

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gSKv3_1": 0.00001,  # S/cm^2
            "eK": -107.0,  # mV, assuming eK for potassium
        }
        self.channel_states = {
            f"{prefix}_m": 0.1,  # Initial value for m gating variable
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Rettig, et al. (1992)",
            "doi": "https://doi.org/10.1002/j.1460-2075.1992.tb05312.x",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/SKv3_1.mod",
            "species": "rat",
            "cell_type": "unknown",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variable m."""
        prefix = self._name
        ms = u[f"{prefix}_m"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        k_cond = params[f"{prefix}_gSKv3_1"] * m
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        return {f"{prefix}_m": m_inf}

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        m_inf = 1 / (1 + save_exp((v - 18.7) / -9.7))
        tau_m = 0.2 * 20.0 / (1 + save_exp((v + 46.56) / -44.14))
        return m_inf, tau_m

class Im_v2(Channel):
    """M-type potassium channel based on Vervaeke et al. (2006)."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        
        # Parameters from PARAMETER block
        self.channel_params = {
            f"{prefix}_gIm_v2": 0.00001,  # S/cm^2
            "ek": -80.0,  # mV
            "celsius": 30.0,  # Temperature in Celsius
        }
        
        # Initialize single state (m)
        self.channel_states = {
            f"{prefix}_m": 0.0,  # Activation gate
        }
        
        self.current_name = "i_K"
        self.META = {
            "reference": "Vervaeke et al. (2006)",
            "ion": "K",
            "current_type": "M-type",
        }

    def compute_rates(self, v: float, params: Dict[str, jnp.ndarray]):
        """Compute rate parameters for the m gate."""
        qt = 2.3**((params["celsius"] - 30) / 10)
        
        # Constants from the mod file
        V_HALF = -48  # mV
        ALPHA_BETA_BASE = 0.007  # /ms
        GATE_CHARGE = 6
        GAMMA = 0.4  # Position of the energy barrier
        KT = 26.12  # mV
        
        # m gate (activation) rates
        mAlpha = ALPHA_BETA_BASE * jnp.exp(
            (GATE_CHARGE * GAMMA * (v - V_HALF)) / KT
        )
        mBeta = ALPHA_BETA_BASE * jnp.exp(
            (-GATE_CHARGE * (1 - GAMMA) * (v - V_HALF)) / KT
        )
        
        mInf = mAlpha / (mAlpha + mBeta)
        mTau = (15 + 1 / (mAlpha + mBeta)) / qt
        
        return {
            "mInf": mInf,
            "mTau": mTau
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of m gate using exponential integration."""
        prefix = self._name
        rates = self.compute_rates(voltages, params)
        
        # Current state
        m = u[f"{prefix}_m"]
        
        # Update using exponential integration
        m_new = solve_inf_gate_exponential(m, dt, rates["mInf"], rates["mTau"])
        
        return {
            f"{prefix}_m": m_new
        }

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        
        # Compute conductance: g = gbar * m
        k_cond = params[f"{prefix}_gIm_v2"] * m
        current = k_cond * (voltages - params["ek"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at steady-state value."""
        prefix = self._name
        rates = self.compute_rates(voltages, params)
        return {
            f"{prefix}_m": rates["mInf"]
        }

class M(Channel):
    """M-currents and other potassium currents in bullfrog sympathetic neurones from Adams et al., 1982, with temperature corrections."""
    # Same as Im in Allen mod files
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gM": 0.00001,  # S/cm^2
            "eK": -107.0,  # mV, assuming eK for potassium
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,  # Initial value for m gating variable
        }
        self.current_name = f"i_K"
        self.META = {
            "reference": "Adams et al. (1982)",
            "doi": "https://doi.org/10.1113/jphysiol.1982.sp014357",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Im.mod",
            "species": "bullfrog",
            "cell_type": "lumbar sympathetic neurones",
            "note": "Corrected rates using Q10 = 2.3, target temperature 34, original 21",
            "ion": "K",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variable m."""
        prefix = self._name
        ms = u[f"{prefix}_m"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the potassium current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        k_cond = params[f"{prefix}_gM"] * m
        current = k_cond * (voltages - params["eK"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        return {f"{prefix}_m": m_inf}

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable, with temperature correction."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        m_alpha = 3.3e-3 * save_exp(2.5 * 0.04 * (v + 35))
        m_beta = 3.3e-3 * save_exp(-2.5 * 0.04 * (v + 35))
        m_inf = m_alpha / (m_alpha + m_beta)
        tau_m = (1 / (m_alpha + m_beta)) / qt
        return m_inf, tau_m


############################
## Calcium channels:      ##
## CaHVA, CaLVA           ##
############################


class CaHVA(Channel):
    """High-Voltage-Activated (HVA) Ca2+ channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCaHVA": 0.00001,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.1,  # Initial value for m gating variable
            f"{self._name}_h": 0.1,  # Initial value for h gating variable
            "eCa": 120,  # mV, assuming eca for demonstration
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Reuveni, et al. (1993)",
            "doi": "https://doi.org/10.1523/jneurosci.13-11-04609.1993",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Ca_HVA.mod",
            "species": "rat",
            "cell_type": "layer 5 pyramidal cell",
            "ion": "Ca",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": u["eCa"]}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCaHVA"] * (ms**2) * hs
        current = ca_cond * (voltages - u["eCa"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltages)
        alpha_h, beta_h = self.h_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def vtrap(x, y):
        """Numerical function to avoid division by zero errors."""
        # normal booleans don't work in jax, need to replace above if statement arguments with jnp functions
        return jnp.where(jnp.abs(x / y) < 1e-6, y * (1 - x / y / 2), x / (jnp.exp(x / y) - 1))
    
    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        alpha = (0.055 * (-27 - v + 1e-6)) / (save_exp((-27.0 - v + 1e-6) / 3.8) - 1.0)
        beta = 0.94 * save_exp((-75.0 - v + 1e-6) / 17.0)
        # alpha = 0.055 * CaHVA.vtrap(-27 - v, 3.8)   
        # beta = 0.94*jnp.exp((-75-v)/17)
        # m_inf = alpha/(alpha + beta)
        # tau_m = 1/(alpha + beta)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = 0.000457 * save_exp((-13.0 - v) / 50.0)
        beta = 0.0065 / (save_exp((-v - 15.0) / 28.0) + 1.0)
        # alpha = (0.000457*jnp.exp((-13-v)/50))  
        # beta = (0.0065/(jnp.exp((-v-15)/28)+1))
        # m_inf = alpha/(alpha + beta)
        # tau_m = 1/(alpha + beta)
        return alpha, beta


class CaLVA(Channel):
    """Low-Voltage-Activated (LVA) Ca2+ channel, based on Avery and Johnston 1996 and Randall 1997"""
    # Same as the one used in the allen models
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCaLVA": 0.00001,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.0,  # Initial value for m gating variable
            f"{self._name}_h": 0.0,  # Initial value for h gating variable
            "eCa": 120,  # mV, assuming eCa for demonstration
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Avery and Johnston (1996)",
            "doi": "https://doi.org/10.1523/jneurosci.16-18-05567.1996",
            "species": "rat",
            "cell_type": "hippocampal CA3 pyramidal cell",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Ca_LVAst.mod",
            "ion": "Ca",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_inf_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": u["eCa"]}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCaLVA"] * (ms**2) * hs
        current = ca_cond * (voltages - u["eCa"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        h_inf, _ = self.h_gate(voltages)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h": h_inf,
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_shifted = v + 10  # Shift by 10 mV
        m_inf = 1.0 / (1 + save_exp((v_shifted + 30) / -6))
        tau_m = (5.0 + 20.0 / (1 + save_exp((v_shifted + 25) / 5))) / qt
        return m_inf, tau_m

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable, adjusted for junction potential."""
        qt = 2.3 ** ((34 - 21) / 10)  # Q10 temperature correction
        v_shifted = v + 10  # Shift by 10 mV
        h_inf = 1.0 / (1 + save_exp((v_shifted + 80) / 6.4))
        tau_h = (20.0 + 50.0 / (1 + save_exp((v_shifted + 40) / 7))) / qt
        return h_inf, tau_h


class CaPump_NEURON(Channel):
    # same as CaPump for now
    """Calcium dynamics tracking inside calcium concentration, modeled after Destexhe et al. 1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered)
            f"{self._name}_decay": 80,  # Rate of removal of calcium in ms
            f"{self._name}_depth": 0.1,  # Depth of shell in um
            f"{self._name}_minCai": 1e-4,  # Minimum intracellular calcium concentration in mM
        }
        self.channel_states = {
            f"CaCon_i": 1e-4,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Destexhe et al., (1994)",
            "doi": "https://doi.org/10.1152/jn.1994.72.2.803",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/CaDynamics_E2.mod",
            "species": "ferret",
            "cell_type": "thalamic reticular nucleus",
            "ion": "Ca",
        }

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = u["i_Ca"] #/ 1000.0
        cai = u["CaCon_i"]
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485.3329  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10000.0 * ica * gamma / (2 * FARADAY * depth)
    
        cai_tau = decay
        cai_inf = minCai + decay * drive_channel
        
        new_cai = exponential_euler(cai, dt, cai_inf, cai_tau)

        return {f"CaCon_i": new_cai}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}

class CaPump(Channel):
    """Calcium dynamics tracking inside calcium concentration, modeled after Destexhe et al. 1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered)
            f"{self._name}_decay": 80,  # Rate of removal of calcium in ms
            f"{self._name}_depth": 0.1,  # Depth of shell in um
            f"{self._name}_minCai": 1e-4,  # Minimum intracellular calcium concentration in mM
        }
        self.channel_states = {
            f"CaCon_i": 1e-04,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Destexhe et al., (1994)",
            "doi": "https://doi.org/10.1152/jn.1994.72.2.803",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/CaDynamics_E2.mod",
            "species": "ferret",
            "cell_type": "thalamic reticular nucleus",
            "ion": "Ca",
        }

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = u["i_Ca"] #/ 1000.0
        cai = u["CaCon_i"]
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485.3329  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10000.0 * ica * gamma / (2 * FARADAY * depth)
    
        cai_tau = decay
        cai_inf = minCai + decay * drive_channel
        
        new_cai = exponential_euler(cai, dt, cai_inf, cai_tau)

        return {f"CaCon_i": new_cai}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 279.45,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {}
        self.channel_states = {"eCa": 0, "CaCon_i": 1e-04, "CaCon_e": 2.0}
        self.current_name = f"i_Ca"
        self.META = {"ion": "Ca"}

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cai = u["CaCon_i"]
        Cao = u["CaCon_e"]
        C = R * T / (2 * F) * 1000  # mV
        vCa = C * jnp.log(Cao / Cai)
        #vCa = 120
        return {"eCa": vCa, "CaCon_i": Cai, "CaCon_e": Cao}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


#################################
## hyperpolarization-activated ##
## cation channel              ##
#################################

class Ih(Channel):
    """H-current (H) from Kole, Hallermann, and Stuart, J. Neurosci., 2006."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gIh": 0.00001,  # S/cm^2
            "eH": -45.0,  # mV, reversal potential for H
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,  # Initial value for m gating variable
        }
        self.current_name = f"i_H"
        self.META = {
            "reference": "Kole, et al. (2006)",
            "doi": "https://doi.org/10.1523/JNEUROSCI.3664-05.2006",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Ih.mod",
            "cell_type": "layer 5 pyramidal cell",
            "species": "rat",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variable m."""
        prefix = self._name
        ms = u[f"{prefix}_m"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the nonspecific current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        h_cond = params[f"{prefix}_gIh"] * m
        current = h_cond * (voltages - params["eH"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        return {f"{prefix}_m": m_inf}
    
    @staticmethod
    def vtrap(x, y):
        """Numerical function to avoid division by zero errors."""
        return jnp.where(jnp.abs(x / y) < 1e-6, y * (1 - x / y / 2), x / (jnp.exp(x / y) - 1))

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        # m_alpha = (
        #     0.001
        #     * 6.43
        #     * (v + 154.9 + 1e-6)
        #     / (save_exp((v + 154.9 + 1e-6) / 11.9) - 1)
        # )
        # m_beta = 0.001 * 193 * save_exp(v / 33.1)
        # m_inf = m_alpha / (m_alpha + m_beta)
        # tau_m = 1 / (m_alpha + m_beta)

        alpha = 0.001 * 6.43 * Ih.vtrap(v + 154.9, 11.9)
        beta  =  0.001*193*jnp.exp(v/33.1)
        m_inf = alpha/(alpha + beta)
        tau_m = 1/(alpha + beta)
        return m_inf, tau_m


class H(Channel):
    """H-current (H) from Kole, Hallermann, and Stuart, J. Neurosci., 2006."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gH": 0.00001,  # S/cm^2
            "eH": -45.0,  # mV, reversal potential for H
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,  # Initial value for m gating variable
        }
        self.current_name = f"i_H"
        self.META = {
            "reference": "Kole, et al. (2006)",
            "doi": "https://doi.org/10.1523/JNEUROSCI.3664-05.2006",
            "code": "https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/mechanisms/Ih.mod",
            "cell_type": "layer 5 pyramidal cell",
            "species": "rat",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variable m."""
        prefix = self._name
        ms = u[f"{prefix}_m"]
        m_new = solve_inf_gate_exponential(ms, dt, *self.m_gate(voltages))
        return {f"{prefix}_m": m_new}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the nonspecific current through the channel."""
        prefix = self._name
        m = u[f"{prefix}_m"]
        h_cond = params[f"{prefix}_gH"] * m
        current = h_cond * (voltages - params["eH"])
        return current

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        prefix = self._name
        m_inf, _ = self.m_gate(voltages)
        return {f"{prefix}_m": m_inf}

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        m_alpha = (
            0.001
            * 6.43
            * (v + 154.9 + 1e-6)
            / (save_exp((v + 154.9 + 1e-6) / 11.9) - 1)
        )
        m_beta = 0.001 * 193 * save_exp(v / 33.1)
        m_inf = m_alpha / (m_alpha + m_beta)
        tau_m = 1 / (m_alpha + m_beta)
        return m_inf, tau_m
        return m_inf, tau_m
