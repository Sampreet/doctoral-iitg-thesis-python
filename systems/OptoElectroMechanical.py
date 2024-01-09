#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""Class to simulate opto-electro-mechanical systems."""

__authors__ = ["Sampreet Kalita"]
__toolbox__ = 'qom-v1.0.1'
__created__ = "2021-06-14"
__updated__ = "2024-01-09"

# dependencies
import numpy as np

# qom modules
from qom.systems import BaseSystem

class OEM_20(BaseSystem):
    r"""Class to simulate an OEM system with multiple modulations in laser amplitude, voltage amplitude and mechanical spring constant.

    Parameters
    ----------
    params : dict
        Parameters for the system. The system parameters are:
        ========    ========================================================================
        key         meaning
        ========    ========================================================================
        A_ls        (*list*) amplitudes of the laser, in the format :math:`\left[ A_{l0}, A_{l-}, A_{l+} \right]`. Default is :math:`\left[ 100.0, 10.0, 10.0 \right]`.
        A_vs        (*list*) amplitudes of the voltage, in the format :math:`\left[ A_{v0}, A_{v-}, A_{v+} \right]`. Default is :math:`\left[ 50.0, 50.0, 50.0 \right]`.
        Delta_0     (*float*) detuning of the laser, :math:`\Delta_{0} = \omega_{a0} - \omega_{l}`. Default is :math:`1.0`.
        gammas      (*list*) optical decay and mechanical and electrical damping rates, in the format :math:`\left[ \gamma_{a}, \gamma_{b}, \gamma_{c} \right]`. Default is :math:`\left[ 0.1, 10^{-6}, 10^{-2} \right]`.
        gs          (*list*) optomechanical and electromechanical coupling constants, in the format :math:`\left[ g_{ab}, g_{bc} \right]`. Default is :math:`\left[ 10^{-3}, 2 \times 10^{-4} \right]`.
        n_ths       (*list*) number of mechanical and electrical thermal phonons, in the format :math:`\left[ n_{th_{b}}, n_{th_{c}} \right]`. Default is :math:`\left[ 0, 0 \right]`.
        Omegas      (*list*) laser and voltage modulation frequencies and the spring constant modulation frequency, in the format :math:`\left[ \Omega_{l}, \Omega_{v}, \Omega_{s} \right]`. Default is :math:`\left[ 2.0, 2.0, 2.0 \right]`.
        omega_c0    (*list*) LC circuit base resonance frequency, :math:`\omega_{c0}`. Default is :math:`1.1`.
        theta       (*list*) mechanical modulation amplitude, :math:`\theta`. Default is :math:`0.5`.
        t_mod       (*str*) type of modulation for the mechanical spring constant. Options are ``'cos'`` (fallback) for cosinusoidal, ``'sin'`` for sinusoidal. Default is ``'cos'``.
        t_pos       (*str*) position of the mechanical membrame in the parallel-plate capacitor. Options are ``'top'`` (fallback) and ``'bottom'``. Default is ``'top'``.
        ========    ========================================================================
    cb_update : callable, optional
        Callback function to update status and progress, formatted as ``cb_update(status, progress, reset)``, where ``status`` is a string, ``progress`` is a float and ``reset`` is a boolean.
    """

    # default system parameters
    system_defaults = {
        'A_ls'      : [100.0, 10.0, 10.0],
        'A_vs'      : [50.0, 50.0, 50.0],
        'Delta_0'   : 1.0,
        'gammas'    : [0.1, 1e-6, 1e-2],
        'gs'        : [1e-3, 2e-4],
        'n_ths'     : [0.0, 0.0],
        'Omegas'    : [2.0, 2.0, 2.0],
        'omega_c0'  : 1.1,
        'theta'     : 0.5,
        't_mod'     : 'cos',
        't_pos'     : 'top'
    }

    def __init__(self, params, cb_update=None):
        """Class constructor for OEM_20."""
        
        # initialize super class
        super().__init__(
            params=params,
            name='OEM_20',
            desc='Multi-modulated OEM System',
            num_modes=3,
            cb_update=cb_update
        )

        # validate strings
        assert self.params['t_mod'] in ['cos', 'sin'], "Parameter ``'t_mod'`` can only assume the values ``'cos'`` and ``'sin'``"
        assert self.params['t_pos'] in ['top', 'bottom'], "Parameter ``'t_pos'`` can only assume the values ``'top'`` and ``'bottom'``"

    def get_A(self, modes, c, t):
        """Method to obtain the drift matrix.

        Parameters
        ----------
        modes : numpy.ndarray
            Classical modes.
        c : numpy.ndarray
            Derived constants and controls.
        t : float
            Time at which the values are calculated.
        
        Returns
        -------
        A : numpy.ndarray
            Drift matrix.
        """

        # extract frequently used variables
        gamma_a, gamma_b, gamma_c = self.params['gammas']
        g_ab, g_bc = self.params['gs']
        _, _, Omega_s = self.params['Omegas']
        alpha, beta, chi = modes

        # effective values
        Delta = self.params['Delta_0'] - 2.0 * g_ab * np.real(beta)
        G_alpha = g_ab * alpha
        g_1 = - g_bc if self.params['t_pos'] == 'bottom' else g_bc
        G_beta = 2.0 * g_1 * np.real(beta)
        G_chi = 2.0 * g_1 * np.real(chi)
        
        # handle fixed point 
        t = 0.0 if t is None else t

        # update modulations
        omega_b = np.sqrt(1.0 + self.params['theta'] * getattr(np, self.params['t_mod'])(Omega_s * t))

        # optical position quadrature
        self.A[0][0] = - gamma_a 
        self.A[0][1] = Delta 
        self.A[0][2] = - 2.0 * np.imag(G_alpha) 
        # optical momentum quadrature
        self.A[1][0] = - Delta
        self.A[1][1] = - gamma_a
        self.A[1][2] = 2.0 * np.real(G_alpha)
        # mechanical position quadrature
        self.A[2][2] = - gamma_b
        self.A[2][3] = omega_b
        # mechanical momentum quadrature
        self.A[3][0] = 2.0 * np.real(G_alpha)
        self.A[3][1] = 2.0 * np.imag(G_alpha)
        self.A[3][2] = - omega_b
        self.A[3][3] = - gamma_b
        self.A[3][4] = 4.0 * G_chi
        # LC charge quadrature
        self.A[4][4] = - gamma_c
        self.A[4][5] = self.params['omega_c0']
        # LC flux quadrature
        self.A[5][2] = 4.0 * G_chi
        self.A[5][4] = - self.params['omega_c0'] + 4.0 * G_beta
        self.A[5][5] = - gamma_c

        return self.A

    def get_coeffs_A(self, modes, c, t):
        """Method to obtain the coefficients of the characteristic equation of the drift matrix.

        Parameters
        ----------
        modes : numpy.ndarray
            Classical modes.
        c : numpy.ndarray
            Derived constants and controls.
        t : float
            Time at which the values are calculated.

        Returns
        -------
        coeffs : int
            Coefficients of the characteristic equation of the drift matrix.
        """

        # extract frequently used variables
        gamma_a, gamma_b, gamma_c = self.params['gammas']
        g_ab, g_bc = self.params['gs']
        _, _, Omega_s = self.params['Omegas']
        omega_c0 = self.params['omega_c0']
        alpha, beta, chi = modes

        # effective values
        Delta = self.params['Delta_0'] - 2.0 * g_ab * np.real(beta)
        G_alpha = g_ab * alpha
        g_1 = - g_bc if self.params['t_pos'] == 'bottom' else g_bc
        G_beta = 2.0 * g_1 * np.real(beta)
        G_chi = 2.0 * g_1 * np.real(chi)

        # handle fixed point 
        t = 0.0 if t is None else t

        # update modulations
        omega_b = np.sqrt(1.0 + self.params['theta'] * getattr(np, self.params['t_mod'])(Omega_s * t))

        # substituted parameters
        D_2 = Delta**2 + gamma_a**2
        G_2 = np.real(np.conjugate(G_alpha) * G_alpha)
        Omega_b_2 = gamma_b**2 + omega_b**2
        Omega_c_2 = gamma_c**2 + omega_c0 * (omega_c0 - 4.0 * G_beta)

        # coefficients
        coeffs = np.array(2 * self.num_modes + 1, dtype=np.float_)
        # a_0
        coeffs[0] = 1.0
        # a_1 
        coeffs[1] = 2.0 * (gamma_a + gamma_b + gamma_c)
        # a_2
        coeffs[2] = D_2**2 + Omega_b_2**2 + Omega_c_2**2 + 4.0 * (gamma_b * gamma_c + gamma_b * gamma_a + gamma_c * gamma_a)
        # a_3
        coeffs[3] = 2.0 * (gamma_b + gamma_c) * D_2**2 + 2.0 * (Omega_b_2**2 * gamma_c + Omega_c_2**2 * gamma_b) + 2.0 * gamma_a * (Omega_b_2**2 + Omega_c_2**2) + 8.0 * gamma_b * gamma_c * gamma_a
        # a_4
        coeffs[4] = (Omega_b_2**2 + Omega_c_2**2 + 4.0 * gamma_b * gamma_c) * D_2**2 + 4.0 * (Omega_b_2**2 * gamma_c + Omega_c_2**2 * gamma_b) * gamma_a + Omega_b_2**2 * Omega_c_2**2 - 4.0 * G_2**2 * Delta * omega_b - 16.0 * G_chi**2 * omega_b * omega_c0
        # a_5
        coeffs[5] = 2.0 * (Omega_b_2**2 * gamma_c + Omega_c_2**2 * gamma_b) * D_2**2 + 2.0 * Omega_b_2**2 * Omega_c_2**2 * gamma_a - 8.0 * G_2**2 * Delta * gamma_c * omega_b - 32.0 * G_chi**2 * gamma_a * omega_b * omega_c0
        # a_6
        coeffs[6] = Omega_b_2**2 * Omega_c_2**2 * D_2**2 - 4.0 * G_2**2 * Omega_c_2**2 * Delta * omega_b - 16.0 * G_chi**2 * D_2**2 * omega_b * omega_c0

        return coeffs
    
    def get_coeffs_beta_sum(self, c):
        """Method to obtain coefficients of the polynomial in the sum of mechanical modes.
        
        Parameters
        ----------
        c : numpy.ndarray
            Derived constants and controls.
        
        Returns 
        -------
        coeffs : numpy.ndarray
            Coefficients of the polynomial in the sum of mechanical modes.
        """

        # extract frequently used variables
        A_l0, _, _ = self.params['A_ls']
        A_v0, _, _ = self.params['A_vs']
        Delta_0 = self.params['Delta_0']
        gamma_a, gamma_b, gamma_c = self.params['gammas']
        g_ab, g_bc = self.params['gs']
        omega_c0 = self.params['omega_c0']

        # effective values
        g_1 = - g_bc if self.params['t_pos'] == 'bottom' else g_bc
        omega_b = np.sqrt(1.0 + self.params['theta'])
        
        # get coefficients
        coeffs = np.zeros(2 * self.num_modes, dtype=np.float_)
        coeffs[0] = 16.0 * omega_c0**2 * g_ab**2 * g_1**2 * (gamma_b**2 + omega_b**2)
        coeffs[1] = - 8.0 * omega_c0 * g_ab * g_1 * (4.0 * Delta_0 * omega_c0 * g_1 + g_ab * (gamma_c**2 + omega_c0**2)) * (gamma_b**2 + omega_b**2)
        coeffs[2] = (16.0 * (gamma_a**2 + Delta_0**2) * omega_c0**2 * g_1**2 + g_ab * (gamma_c**2 + omega_c0**2) * (16.0 * Delta_0 * omega_c0 * g_1 + g_ab * (gamma_c**2 + omega_c0**2))) * (gamma_b**2 + omega_b**2)
        coeffs[3] = (- 2.0 * Delta_0 * g_ab * (gamma_c**2 + omega_c0**2)**2 - 8.0 * omega_c0 * g_1 * (gamma_c**2 + omega_c0**2) * (gamma_a**2 + Delta_0**2)) * (gamma_b**2 + omega_b**2) - 32.0 * omega_b * omega_c0**2 * g_ab * g_1**2 * A_l0**2 - 8.0 * omega_b * omega_c0**2 * g_ab**2 * g_1 * A_v0**2
        coeffs[4] = (gamma_a**2 + Delta_0**2) * (gamma_c**2 + omega_c0**2)**2 * (gamma_b**2 + omega_b**2) + 16.0 * omega_b * omega_c0 * g_ab * g_1 * A_l0**2 * (gamma_c**2 + omega_c0**2) + 16.0 * omega_b * omega_c0**2 * g_ab * g_1 * A_v0**2 * Delta_0
        coeffs[5] = - 2.0 * omega_b * g_ab * A_l0**2 * (gamma_c**2 + omega_c0**2) - 8.0 * omega_b * omega_c0**2 * g_1 * A_v0**2 * (gamma_a**2 + Delta_0**2)

        return coeffs

    def get_D(self, modes, corrs, c, t):
        """Method to obtain the noise matrix.
        
        Parameters
        ----------
        modes : numpy.ndarray
            Classical modes.
        corrs : numpy.ndarray
            Quantum correlations.
        c : numpy.ndarray
            Derived constants and controls.
        t : float
            Time at which the values are calculated.
        
        Returns
        -------
        D : numpy.ndarray
            Noise matrix.
        """

        # extract frequently used variables
        gammas = self.params['gammas']
        n_ths = self.params['n_ths']
        
        # optical mode
        self.D[0][0] = gammas[0]
        self.D[1][1] = gammas[0]
        # mechanical mode
        self.D[2][2] = gammas[1] * (2.0 * n_ths[0] + 1.0) 
        self.D[3][3] = gammas[1] * (2.0 * n_ths[0] + 1.0) 
        # LC mode
        self.D[4][4] = gammas[2] * (2.0 * n_ths[1] + 1.0) 
        self.D[5][5] = gammas[2] * (2.0 * n_ths[1] + 1.0) 

        return self.D
    
    def get_ivc(self):
        """Method to obtain the initial values of the modes, correlations and derived constants and controls.
        
        Returns
        -------
        iv_modes : numpy.ndarray
            Initial values of the classical modes.
        iv_corrs : numpy.ndarray
            Initial values of the quantum correlations.
        c : numpy.ndarray
            Derived constants and controls.
        """

        # extract frequently used variables
        n_ths = self.params['n_ths']
 
        # initial mode values
        iv_modes = np.zeros(self.num_modes, dtype=np.complex_)

        # initial quadrature correlations
        iv_corrs = np.zeros(self.dim_corrs, dtype=np.float_)
        iv_corrs[0][0] = 0.5 
        iv_corrs[1][1] = 0.5
        iv_corrs[2][2] = n_ths[0] + 0.5
        iv_corrs[3][3] = n_ths[0] + 0.5
        iv_corrs[4][4] = n_ths[1] + 0.5
        iv_corrs[5][5] = n_ths[1] + 0.5

        return iv_modes, iv_corrs, np.empty(0)

    def get_modes_steady_state(self, c):
        """Method to obtain the steady state modes.
        
        Parameters
        ----------
        c : numpy.ndarray
            Derived constants and controls.
        
        Returns 
        -------
        Modes : list
            Steady state modes.
        """

        # extract frequently used variables
        A_l0, _, _ = self.params['A_ls']
        A_v0, _, _ = self.params['A_vs']
        Delta_0 = self.params['Delta_0']
        gamma_a, gamma_b, gamma_c = self.params['gammas']
        g_ab, g_bc = self.params['gs']
        omega_c0 = self.params['omega_c0']

        # effective values
        omega_b = np.sqrt(1.0 + self.params['theta'])

        # get real roots for the sum of betas
        coeffs = self.get_coeffs_beta_0_sum(c)
        roots = np.roots(coeffs)
        beta_sums = np.real(roots[np.imag(roots) == 0.0])

        # initialize modes
        Modes = np.zeros((len(beta_sums), self.num_modes), dtype=np.complex_)

        # calculate mode amplitudes for each sum
        for i in range(len(beta_sums)):
            # optical mode
            alpha = A_l0 / (gamma_a + 1.0j * Delta_0 - 1.0j * g_ab * beta_sums[i])
            Modes[i][0] = alpha
            # sum of chi and its conjugate
            chi_sum = 2.0 * omega_c0 * A_v0 / (gamma_c**2 + omega_c0**2 - 4.0 * omega_c0 * g_bc * beta_sums[i])
            # mechanical mode
            beta = (1.0j * g_ab * np.conjugate(alpha) * alpha + 1.0j * g_bc * chi_sum**2) / (gamma_b + 1.0j * omega_b)
            Modes[i][1] = beta
            # LC circuit mode
            Modes[i][2] = (1.0j * A_v0 + 2.0j * g_bc * beta_sums[i] * chi_sum) / (gamma_c + 1.0j * omega_c0)

        return Modes
    
    def get_mode_rates(self, modes, c, t):
        """Method to obtain the rates of change of the modes.

        Parameters
        ----------
        modes : numpy.ndarray
            Classical modes.
        c : numpy.ndarray
            Derived constants and controls.
        t : float
            Time at which the values are calculated.
        
        Returns
        -------
        mode_rates : numpy.ndarray
            Rate of change of the modes.
        """

        # extract frequently used variables
        A_l0, A_lm, A_lp = self.params['A_ls']
        A_v0, A_vm, A_vp = self.params['A_vs']
        gamma_a, gamma_b, gamma_c = self.params['gammas']
        g_ab, g_bc = self.params['gs']
        Omega_l, Omega_v, Omega_s = self.params['Omegas']
        omega_c0 = self.params['omega_c0']
        alpha, beta, chi = modes

        # effective values
        Delta = self.params['Delta_0'] - 2.0 * g_ab * np.real(beta)
        g_1 = - g_bc if self.params['t_pos'] == 'bottom' else g_bc

        # handle fixed point 
        t = 0.0 if t is None else t

        # update modulations
        A_l = A_l0 + A_lm * np.exp(1j * Omega_l * t) + A_lp * np.exp(-1j * Omega_l * t)
        A_v = A_v0 + A_vm * np.exp(1j * Omega_v * t) + A_vp * np.exp(-1j * Omega_v * t)
        omega_b = np.sqrt(1.0 + self.params['theta'] * getattr(np, self.params['t_mod'])(Omega_s * t))

        # calculate mode rates
        # optical
        dalpha_dt = - (gamma_a + 1.0j * Delta) * alpha + A_l
        # mechanical
        dbeta_dt = 1.0j * g_ab * np.conjugate(alpha) * alpha - (gamma_b + 1.0j * omega_b) * beta + 4.0j * g_1 * np.real(chi)**2
        # circuit
        dchi_dt = 8.0j * g_1 * np.real(beta) * np.real(chi) - (gamma_c + 1.0j * omega_c0) * chi + 1.0j * A_v

        return np.array([dalpha_dt, dbeta_dt, dchi_dt], dtype=np.complex_)