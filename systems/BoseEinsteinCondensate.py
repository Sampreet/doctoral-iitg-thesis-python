#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Class to simulate a Bose-Einstein condensate systems."""

__authors__ = ["Sampreet Kalita"]
__toolbox__ = 'qom-v1.0.1'
__created__ = "2020-12-21"
__updated__ = "2024-01-09"

# dependencies
from sympy import Symbol
import numpy as np
import scipy.constants as sc

# qom modules
from qom.systems import BaseSystem

class BEC_10(BaseSystem):
    r"""Class to simulate a BEC-OM system with a weak probe laser and a strong control laser containing OAM.

    Parameters
    ----------
    params : dict
        Parameters for the system. The system parameters are:
        ================    ====================================================
        key                 meaning
        ================    ====================================================
        Delta_tilde         (float) laser detuning :math:`\tilde{\Delta}` with normalization determined by the value of ``t_Delta_norm`` and offset determined by the value of ``t_Delta_offset``. Default is :math:`-1.0` units with the value of ``t_Delta_norm`` set to "Omega_m" and ``t_Delta_offset`` set to "zero".
        delta               (float) probe detuning :math:`\delta` with normalization determined by the value of ``t_delta_norm`` and offset determined by the value of ``t_delta_offset``. Default is :math:`0.0` units with the value of ``t_delta_norm`` set to "Omega_m" and ``t_delta_offset`` set to "Omega_m".
        G                   (float) sidemode coupling constant :math:`G` in Hertz. Default is :math:`2 \pi \times 10^{3}` Hz.
        g_tilde_norm        (float) normalized atom-atom interaction strength :math:`4 I N \tilde{g} / \hbar`. Default is :math:`0.0`.
        gamma_m             (float) mechanical decay rate :math:`\gamma_{m}` in Hertz. Default is :math:`2 \pi \times 0.8` Hz.
        gamma_o             (float) optical decay rate :math:`\gamma_{o}` in Hertz. Default is :math:`2 \pi \times 10^{3}` Hz.
        k                   (float) multiplier for subtractive detuning :math:`k`. Default is :math:`1`.
        L_p                 (float) winding number :math:`L_{p}`. Default is :math:`1`.
        l                   (float) OAM number :math:`l`. Default is :math:`20`.
        lambda_lc           (float) wavelength of the control laser :math:`\lambda_{lc}` in metres. Default is :math:`589 \times 10^{-9}` m.
        m                   (float) mass of the atom :math:`m` in amu. Default is :math:`23` amu.
        mu                  (float) cavity coupling parameter :math:`\mu`. Default is :math:`0.5`.
        N                   (float) number of BEC atoms :math:`N`. Default is :math:`10^{4}`.
        P_lc                (float) power of the control laser :math:`P_{lc}` normalized by the critical power :math:`P_{lp_{cr}}` if the value of ``t_P_lc_norm`` is "cr", otherwise the fixed value in Watts. Default is :math:`1 \times 10^{-15}` Watts with the value of ``t_P_lc_norm`` set to "none".
        P_lp_norm           (float) power of the probe laser :math:`P_{lp}` normalized by :math:`P_{lc}`. Default is :math:`0.01`.
        R                   (float) radius of the ring :math:`R` in metres. Default is :math:`12 \times 10^{-6}` m.
        t_approx            (str) type of approximation. Options are "del" for approximation on delta, "res" for resolved sideband approximation, "del-res" for both and "none" (fallback). Default is "none".
        t_Delta_norm        (str) type of normalization for the laser detuning. Options are "cr" for critical detuning :math:`\tilde{Delta}_{cr}`, "gamma_m" for sidemode decay rate, "gamma_o" for optical decay rate, "Omega_c" for first sidemode, "Omega_d" for second sidemode, "Omega_m" for additive detuning, "Omega_n" for subtractive detuning, otherwise :math:`1.0` (fallback). Default is "none".
        t_Delta_offset      (str) type of offset for the laser detuning. Options are "cr" for critical detuning, "gamma_m" for sidemode decay rate, "gamma_o" for optical decay rate, "Omega_c" or "-Omega_c" for first sidemode, "Omega_d" or "-Omega_d" for second sidemode, "Omega_m" or "-Omega_m" for additive detuning, "Omega_n" or "-Omega_n" for subtractive detuning, otherwise no offset (fallback). Default is "zero".
        t_delta_norm        (str) type of normalization for the probe detuning. Options are "cr" for critical detuning :math:`\tilde{Delta}_{cr}`, "Delta" for the same value as :math:`\tilde{\Delta}`, "gamma_m" for sidemode decay rate, "gamma_o" for optical decay rate, "Omega_c" for first sidemode, "Omega_d" for second sidemode, "Omega_m" for additive detuning, "Omega_n" for subtractive detuning, otherwise :math:`1.0` (fallback). Default is "none".
        t_delta_offset      (str) type of offset for the probe detuning. Options are "cr" for critical detuning, "Delta" for the same value as :math:`\tilde{\Delta}`, "gamma_m" for sidemode decay rate, "gamma_o" for optical decay rate, "Omega_c" or "-Omega_c" for first sidemode, "Omega_d" or "-Omega_d" for second sidemode, "Omega_m" or "-Omega_m" for additive detuning, "Omega_n" or "-Omega_n" for subtractive detuning, otherwise offset (fallback). Default is "zero".
        t_line              (str) type of scattering line. Options are "s" or "S" (fallback) for Stokes and "as" or "aS" for anti-Stokes. Default is "s".
        t_oss_method        (str) type of method to use for calculating the optical steady state. Options are "basic" which ignores the coefficient of ``N_o`` (fallback) and "cubic" to solve the cubic equation. Default is "cubic".
        t_P_lc_norm         (str) type of normalization for :math:`P_{lc}`. Options are "cr" for critical power, otherwise :math:`1.0` (fallback). Default is "none".
        ================    ====================================================
    cb_update : callable, optional
        Callback function to update status and progress, formatted as ``cb_update(status, progress, reset)``, where ``status`` is a string, ``progress`` is an integer and ``reset`` is a boolean.
    """

    # default system parameters
    system_defaults = {
        'Delta_tilde'   : -1.0,
        'delta'         : 0.0,
        'G'             : 2 * np.pi * 1e3,
        'g_tilde_norm'  : 0.0,
        'gamma_m'       : 2 * np.pi * 0.8,
        'gamma_o'       : 2 * np.pi * 1e3,
        'k'             : 1,
        'L_p'           : 1,
        'l'             : 20,
        'lambda_lc'     : 589e-9,
        'm'             : 23,
        'mu'            : 0.5,
        'N'             : 1e4,
        'P_lc'          : 1e-15,
        'P_lp_norm'     : 0.01,
        'R'             : 10e-6,
        't_approx'      : 'none',
        't_Delta_norm'  : 'Omega_m',
        't_Delta_offset': 'zero',
        't_delta_norm'  : 'Omega_m',
        't_delta_offset': '-Omega_m',
        't_line'        : 's',
        't_oss_method'  : 'cubic',
        't_P_lc_norm'   : 'none'
    }

    def __init__(self, params, cb_update=None):
        """Class constructor for BEC_10."""
        
        # initialize super class
        super().__init__(
            params=params,
            name='BEC_00',
            desc="BEC-OM System with Control and Probe Lasers",
            num_modes=3,
            cb_update=cb_update
        )

    def get_A(self, modes, c, t=None):
        """Method to obtain the drift matrix.

        Parameters
        ----------
        modes : list
            Values of the modes.
        params : list
            Constant parameters.
        t : float
            Time at which the rates are calculated.
        
        Returns
        -------
        A : list
            Drift matrix.
        """

        # extract frequently used variables
        Delta_tilde = c[0]
        G = c[4]
        g_tilde = c[5]
        gamma_m = c[6]
        gamma_o = c[7]
        N = c[9]
        omegas = [c[10], c[11]]
        alpha = modes[0]
        betas = [modes[1], modes[2]]
        temp = 2 * g_tilde * N

        # derived frequencies
        Delta = Delta_tilde - 2 * G * np.real(np.sum(betas))

        # drift matrix
        if self.A is None or np.shape(self.A) != (6, 6):
            self.A = np.zeros([6, 6], dtype=np.float_)
        # optical mode
        self.A[0][0] = - gamma_o / 2
        self.A[0][1] = - Delta
        self.A[0][2] = np.sqrt(2) * G * np.imag(alpha)
        self.A[0][4] = np.sqrt(2) * G * np.imag(alpha)
        self.A[1][0] = Delta
        self.A[1][1] = - gamma_o / 2
        self.A[1][2] = - np.sqrt(2) * G * np.real(alpha)
        self.A[1][4] = - np.sqrt(2) * G * np.real(alpha)
        # first mechanical mode
        self.A[2][3] = omegas[0] + 2 * temp
        self.A[2][5] = - temp
        self.A[3][0] = - np.sqrt(2) * G * np.real(alpha)
        self.A[3][1] = - np.sqrt(2) * G * np.imag(alpha)
        self.A[3][2] = - omegas[0] + 2 * temp
        self.A[3][3] = - gamma_m
        self.A[3][4] = - temp
        # second mechanical mode
        self.A[4][3] = - temp
        self.A[4][5] = omegas[1] + 2 * temp
        self.A[5][0] = - np.sqrt(2) * G * np.real(alpha)
        self.A[5][1] = - np.sqrt(2) * G * np.imag(alpha)
        self.A[5][2] = - temp
        self.A[5][4] = - omegas[1] + 2 * temp
        self.A[5][5] = - gamma_m

        return self.A

    def get_absorption(self, c):
        """Method to obtain the absorption.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.

        Returns
        -------
        absorp : float
            Absorption.
        """

        # get transmission coefficient
        # anti-Stokes field
        if self.params['t_line'] == 'as':
            _, _t = self.get_transmission_coeffs(
                c=c
            )
        # Stokes field
        else:
            _t, _ = self.get_transmission_coeffs(
                c=c
            )

        # extract absorption
        absorp = np.real(1 - _t)

        return absorp

    def get_coeffs(self, modes, c, t=None):
        """Method to obtain the coefficients of the characteristic equation.

        Parameters
        ----------
        modes : list
            Values of the modes.
        params : list
            Constants parameters.
        t : float
            Time at which the coefficients are calculated.

        Returns
        -------
        coeffs : int
            Coefficients of the characteristic equation.
        """

        # extract frequently used variables
        Delta_tilde = c[0]
        G = c[4]
        gamma_m = c[6]
        gamma_o = c[7]
        t_oss_method = self.params['t_oss_method']
        alpha = modes[0]

        # get effective values
        A_mathcal, Omegas, omega_tildes, C = self.get_effective_values(
            c=c
        )
        # calculate mean optical occupancy
        N_o = np.real(np.conjugate(alpha) * alpha)
        # calculate effiective detuning
        Delta = Delta_tilde + (C * N_o if 'cubic' in t_oss_method else 0)

        # frequently used expressions
        A_2 = A_mathcal**2 + Omegas[0]**2 * Omegas[1]**2
        D_2 = Delta**2 + gamma_o**2 / 4
        D_N = 2 * Delta * G**2 * N_o * (omega_tildes[0] + omega_tildes[1])
        O_2 = Omegas[0]**2 + Omegas[1]**2

        # coefficients
        coeffs = list()
        # a_0
        coeffs.append(1)
        # a_1
        coeffs.append(2 * gamma_m + gamma_o)
        # a_2
        coeffs.append(D_2 + O_2 + gamma_m**2 + 2 * gamma_m * gamma_o)
        # a_3
        coeffs.append(2 * D_2 * gamma_m + O_2 * (gamma_m + gamma_o) + gamma_m**2 * gamma_o)
        # a_4
        coeffs.append(A_2 + D_2 * (O_2 + gamma_m**2) + D_N + O_2 * gamma_m * gamma_o)
        # a_5
        coeffs.append(A_2 * gamma_o + D_2 * O_2 * gamma_m + D_N * gamma_m)
        # a_6
        coeffs.append(A_2 * D_2 + np.sqrt(A_2) * D_N)
        # # alternate expression
        # coeffs.append(A_2 * D_2 + 2 * A_mathcal * Delta * G**2 * N_o * (omega_tildes[0] - omega_tildes[1]) + 2 * Delta * G**2 * N_o * (Omegas[0]**2 * omega_tildes[1] + Omegas[1]**2 * omega_tildes[0]))

        return coeffs

    def get_cooperativity(self, c):
        """Method to obtain the cooperativity of the system.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
            
        Returns
        -------
        C_mathcal : float
            Cooperativity.
        """

        # extract frequently used variables
        Delta_tilde = c[0]
        eta_lc = c[2]
        G = c[4]
        gamma_m = c[6]
        gamma_o = c[7]
        t_oss_method = self.params['t_oss_method']

        # get mean occupancy amplitude
        alpha_ss, _ = self.get_mean_optical_amplitudes(method='cubic' if 'cubic' in t_oss_method else 'basic')
        alpha_s = alpha_ss[0]

        # # display steady-state values
        # logger.debug('alpha_s={} alpha_s_basic={}\n'.format(alpha_s, eta_lc / (gamma_o / 2 - 1j * Delta_tilde)))

        # mean optical occupancy
        N_o = np.real(np.conjugate(alpha_s) * alpha_s)

        # calculate cooperativity
        C_mathcal = 4 * G**2 * N_o / gamma_o / gamma_m

        return C_mathcal

    def get_deltas(self, c):
        """Method to obtain the complex solution of the denominator of the output amplitudes.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
        
        Returns
        -------
        deltas : list
            Complex roots of probe detuning.
        """

        # extract frequently used variables
        Delta_tilde = c[0]
        G = c[4]
        gamma_m = c[6]
        gamma_o = c[7]

        # get effective values
        A_mathcal, Omegas, omega_tildes, C = self.get_effective_values(
            c=c
        )
        # calculate mean optical occupancy
        N_o, _ = self.get_mean_optical_occupancies()
        N_o = N_o[0]
        # calculate effiective detuning
        Delta = Delta_tilde + C * N_o

        # probe detuning
        delta = Symbol('delta', complex=True)
        # effective values
        Gamma_m = gamma_o / 2 - 1j * (Delta + delta)
        Gamma_p = gamma_o / 2 + 1j * (Delta - delta)
        chis =  [1 / (Omegas[i]**2 - 1j * delta * gamma_m - delta**2) for i in range(2)]
        # Lambda
        _num = A_mathcal * chis[0] * chis[1] * (omega_tildes[0] - omega_tildes[1])
        _num += sum([chis[i] * omega_tildes[i] for i in range(2)])
        _den = A_mathcal**2 * chis[0] * chis[1] + 1
        Lambda = _num / _den

        # equation for delta
        eqtn_delta = Gamma_m * Gamma_p + 2 * Delta * G**2 * Lambda * N_o
        eqtn_delta *= _den / chis[0] / chis[1]
        eqtn_delta = eqtn_delta.factor().expand()

        # solve for delta
        coeff_1 = eqtn_delta.coeff(delta)
        coeff_2 = eqtn_delta.coeff(delta**2)
        coeff_3 = eqtn_delta.coeff(delta**3)
        coeff_4 = eqtn_delta.coeff(delta**4)
        coeff_5 = eqtn_delta.coeff(delta**5)
        coeff_6 = eqtn_delta.coeff(delta**6)
        coeff_0 = eqtn_delta - coeff_6 * delta**6 - coeff_5 * delta**5 - coeff_4 * delta**4 - coeff_3 * delta**3 - coeff_2 * delta**2 - coeff_1 * delta
        coeff_0 = coeff_0.expand()
        # list of coefficients
        coeffs = [coeff_6, coeff_5, coeff_4, coeff_3, coeff_2, coeff_1, coeff_0]

        # convert to numpy constants
        for i in range(len(coeffs)):
            coeffs[i] = np.complex(np.real(coeffs[i]), np.imag(coeffs[i]))

        # convert to list
        deltas = np.roots(coeffs).tolist()

        return deltas

    def get_dispersion(self, c):
        """Method to obtain the dispersion.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.

        Returns
        -------
        disper : float
            Dispersion.
        """

        # get transmission coefficient
        # anti-Stokes field
        if self.params['t_line'] == 'as':
            _, _t = self.get_transmission_coeffs(
                c=c
            )
        # Stokes field
        else:
            _t, _ = self.get_transmission_coeffs(
                c=c
            )

        # extract dispersion
        disper = np.imag(1 - _t)

        return disper

    def get_effective_values(self, c):
        """Method to obtain the effective values.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
        
        Returns
        -------
        A_mathcal : float
            The effective interaction.
        Omegas : list
            The effective sidemode frequencies.
        omega_tildes : list
            The modified sidemode frequencies.
        C : float
            The coefficient of the mean optical occupancy.
        """

        # extract frequently used variables
        G = c[4]
        g_tilde = c[5]
        N = c[9]
        omegas = [c[10], c[11]]
        temp = 2 * g_tilde * N
        
        # first element
        A_mathcal = temp * (omegas[0] - omegas[1])
        # second element
        Omega_c = np.sqrt((omegas[0] + 2 * temp)**2 - temp**2)
        Omega_d = np.sqrt((omegas[1] + 2 * temp)**2 - temp**2)
        Omegas = [Omega_c, Omega_d]
        # third element
        omega_c_tilde = omegas[0] + temp
        omega_d_tilde = omegas[1] + temp
        omega_tildes = [omega_c_tilde, omega_d_tilde]
        # fourth element
        A_2 = A_mathcal**2 + Omega_c**2 * Omega_d**2
        C = G**2 * (omega_c_tilde + omega_d_tilde) / np.sqrt(A_2)
        # # alternate expression
        # C = G**2 * (A_mathcal * (omega_c_tilde - omega_d_tilde) + Omega_c**2 * omega_d_tilde + Omega_d**2 * omega_c_tilde) / A2

        return A_mathcal, Omegas, omega_tildes, C
    
    def get_fwhm_norm_resonance(self, c):
        """Method to obtain the normalized FWHM at resonance using the analytical expression.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
            
        Returns
        -------
        fwhm : float
            Full width at half maxima.
        """

        # extract frequently used variables
        gamma_m = c[6]
        _, Omegas, _, _ = self.get_effective_values(
            c=c
        )

        # get cooperativity
        C_mathcal = self.get_cooperativity(
            c=c
        )

        # calculate FWHM
        fwhm = gamma_m * (1 + C_mathcal) / Omegas[0]

        return fwhm

    def get_ivc(self):
        r"""Method to obtain the initial values and constants required for the IVP.
        
        Returns
        -------
        iv : list
            Initial values of variables.
            First element contains the optical mode.
            Second element contains the first mechanical sidemode.
            Third element contains the second mechanical sidemode.
            Remaining :math:`4 \times 3^{2}` elements contain the quadrature correlations.

        c : list
            Constants of the IVP.
            First :math:`4 \times 3^{2}` elements contain the noise matrix.
            Rest of the elements contain the system parameters ``params`` in the following order:
            ========    =============================================
            index       parameter
            ========    =============================================
            0           cavity detuning :math:`\tilde{\Delta}`.
            1           probe detuning :math:`\delta`.
            2           control laser amplitude :math:`\eta_{lc}`.
            3           probe laser amplitude :math:`\eta_{lp}`.
            4           sidemode coupling constant :math:`G`.
            5           atom-atom interaction strength :math:`\tilde{g}`.
            6           mechanical decay rate :math:`\gamma_{m}`.
            7           optical decay rate :math:`\gamma_{o}`.
            8           cavity coupling parameter :math:`\mu`.
            9           number of BEC atoms :math:`N`.
            10          first sidemode frequency :math:`\omega_{c}`.
            11          second sidemode frequency :math:`\omega_{d}`.
            ========    =============================================
        """

        # extract frequently used variables
        Delta_tilde = self.params['Delta_tilde']
        delta = self.params['delta']
        G = self.params['G']
        g_tilde_norm = self.params['g_tilde_norm']
        gamma_m = self.params['gamma_m']
        gamma_o = self.params['gamma_o']
        k = self.params['k']
        L_p = self.params['L_p']
        l = self.params['l']
        lambda_lc = self.params['lambda_lc']
        m = self.params['m']
        mu = self.params['mu']
        N = self.params['N']
        P_lc = self.params['P_lc']
        P_lp_norm = self.params['P_lp_norm']
        R = self.params['R']
        t_Delta_norm = self.params['t_Delta_norm']
        t_Delta_offset = self.params['t_Delta_offset']
        t_delta_norm = self.params['t_delta_norm']
        t_delta_offset = self.params['t_delta_offset']
        t_P_lc_norm = self.params['t_P_lc_norm']

        # moment of inertia
        I = m * sc.physical_constants['atomic mass constant'][0] * R**2

        # atomic interaction strengths
        g_tilde = g_tilde_norm * sc.hbar / 4 / I / N

        # frequently used variable
        temp = 2 * g_tilde * N

        # first sidemode
        omega_c = sc.hbar * (L_p + 2 * l)**2 / 2 / I
        omega_c_tilde = omega_c + temp
        Omega_c = np.sqrt((omega_c + 2 * temp)**2 - temp**2)
        # second sidemode
        omega_d = sc.hbar * (L_p - 2 * l)**2 / 2 / I
        omega_d_tilde = omega_d + temp
        Omega_d = np.sqrt((omega_d + 2 * temp)**2 - temp**2)
        
        # frequently used variables
        A_mathcal = temp * (omega_c - omega_d)
        A_2 = A_mathcal**2 + Omega_c**2 * Omega_d**2
        C = G**2 * (omega_c_tilde + omega_d_tilde) / np.sqrt(A_2)

        # critical detuning
        Delta_tilde_cr = - np.sqrt(3) * gamma_o / 2
        # frequency of the control laser
        omega_lc = 2 * np.pi * sc.c / lambda_lc
        # critical power of the control laser
        P_cr = gamma_o**2 * sc.hbar * omega_lc / 3 / np.sqrt(3) / C / mu

        # mean and difference of frequencies
        Omega_m = (Omega_c + Omega_d) / 2
        Omega_n = (Omega_c - Omega_d) * k

        # selector for detuning and offset
        _selector = {
            'cr': Delta_tilde_cr,
            'gamma_m': gamma_m,
            'gamma_o': gamma_o,
            'Omega_c': Omega_c,
            '-Omega_c': -Omega_c,
            'Omega_d': Omega_d,
            '-Omega_d': -Omega_d,
            'Omega_m': Omega_m,
            '-Omega_m': -Omega_m,
            'Omega_n': Omega_n,
            '-Omega_n': -Omega_n
        }
        # laser detuning
        Delta_tilde = Delta_tilde * np.abs(_selector.get(t_Delta_norm, 1.0)) + _selector.get(t_Delta_offset, 0.0)
        # probe detuning
        delta = delta * np.abs(_selector.get(t_delta_norm, 1.0)) + _selector.get(t_delta_offset, Delta_tilde if t_delta_offset == 'Delta' else 0.0)
        
        # selector for control power
        _selector_P_lc = {
            'cr': P_cr
        }
        # power of the control laser
        P_lc = P_lc * _selector_P_lc.get(t_P_lc_norm, 1.0)
        # amplitude of the control laser
        eta_lc = np.sqrt(mu * gamma_o * P_lc / sc.hbar / omega_lc)

        # power of the probe laser
        P_lp = P_lp_norm * P_lc
        # frequency of the probe laser
        omega_lp = delta + omega_lc
        # amplitude of the probe laser
        eta_lp = np.sqrt(mu * gamma_o * P_lp / sc.hbar / omega_lp)

        # scattering line
        self.params['t_line'] = 'as' if self.params['t_line'] == 'as' or self.params['t_line'] == 'aS' else 's'
 
        # initial mode values as 1D list
        u_0 = np.zeros(3, dtype=np.complex_).tolist()
        # initial quadrature correlations
        V_0 = np.zeros([6, 6], dtype=np.float_)
        V_0[0][0] = 0.5
        V_0[1][1] = 0.5
        V_0[3][3] = 0.5
        V_0[5][5] = 0.5
        # convert to 1D list and concatenate all variables
        iv = u_0 + [np.complex_(element) for element in V_0.flatten()]

        # noise correlation matrix
        D = np.zeros([6, 6], dtype=np.float_)
        # optical mode
        D[0][0] = eta_lp + mu * gamma_o / 2
        D[1][1] = eta_lp + mu * gamma_o / 2
        D[3][3] = gamma_m
        D[5][5] = gamma_m
        
        # constant parameters
        c = [Delta_tilde, delta] + \
            [eta_lc, eta_lp] + \
            [G, g_tilde] + \
            [gamma_m, gamma_o] + \
            [mu, N] + \
            [omega_c, omega_d]

        # # display parameter values
        # logger.debug('G={}, g_tilde_norm={}, gamma_m={}, gamma_o={}, L_p={}, l={}, lamnda_lc={}, m={}, mu={}, N={}\n'.format(G, g_tilde_norm, gamma_m, gamma_o, L_p, l, lambda_lc, m, mu, N))
        # # display critical values values
        # logger.debug('omega_lc={}, Delta_tilde_cr={}, P_cr={}, Omegas={}, Delta_tilde={}, C={}\n'.format(omega_lc, Delta_tilde_cr, P_cr, [Omega_c, Omega_d], Delta_tilde, C))
        # # display condition for weak perturbations
        # logger.debug('hbar L_p^2 / 2 I + 2 g_tilde N={}, g_a^2 / Delta_a={}, 4 g_tilde N={}\n'.format(sc.hbar * L_p**2 / 2 / I + 2 * g_tilde * N, 2 * np.sqrt(2) * params[4] / np.sqrt(params[9]), 4 * g_tilde * N))

        return iv, None, c

    def get_lifetimes(self, c):
        """Method to obtain the normalized lifetimes.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.

        Returns
        -------
        lieftimes : list
            Normalized lifetimes.
        """

        # get normalized roots of offset delta
        deltas = self.get_deltas(
            c=c
        )

        # get normalized lifetimes
        lieftimes = np.imag(deltas).tolist()

        return lieftimes

    def get_mode_rates(self, modes, c, t=None):
        """Method to obtain the rates of the optical and mechanical modes.

        Parameters
        ----------
        modes : list
            Values of the modes.
        params : list
            Constants parameters.
        t : float
            Time at which the rates are calculated.

        Returns
        -------
        mode_rates : list
            Rates of each mode.
        """

        # extract frequently used variables
        Delta_tilde = c[0]
        delta = c[1]
        eta_lc = c[2]
        eta_lp = c[3]
        G = c[4]
        g_tilde = c[5]
        gamma_m = c[6]
        gamma_o = c[7]
        N = c[9]
        omegas = [c[10], c[11]]
        alpha = modes[0]
        betas = [modes[1], modes[2]]
        
        # frequently used expressions
        temp = 2 * g_tilde * N

        # derived frequencies
        Delta = Delta_tilde - np.sqrt(2) * G * np.real(np.sum(betas))

        # optical mode
        dalpha_dt = (- gamma_o / 2 + 1j * Delta) * alpha + eta_lc
        # probe term
        if t is not None:
            dalpha_dt += eta_lp * np.exp(- 1j * delta * t)
            
        # first mechanical mode
        dq_c_dt = 2 * (omegas[0] + 2 * temp) * np.imag(betas[0]) - 2 * temp * np.imag(betas[1])
        dp_c_dt = - 2 * (omegas[0] + 2 * temp) * np.real(betas[0]) - 2 * gamma_m * np.imag(betas[0]) - 2 * temp * np.imag(betas[1]) - G * np.real(np.conjugate(alpha) * alpha)
        dbeta_c_dt = (dq_c_dt + 1j * dp_c_dt) / 2
        # # alternate expression
        # dbeta_c_dt = - 1j * G / np.sqrt(2) * np.conjugate(alpha) * alpha - gamma_m * betas[0] - 1j * (omegas[0] + 2 * temp) * betas[0] - 2j * g_tilde * N * np.conjugate(betas[1])

        # second mechanical mode
        dq_d_dt = - 2 * temp * np.imag(betas[0]) + 2 * (omegas[1] + 2 * temp) * np.imag(betas[1])
        dp_d_dt = - 2 * temp * np.imag(betas[0]) - 2 * (omegas[1] + 2 * temp) * np.real(betas[1]) - 2 * gamma_m * np.imag(betas[1]) - G * np.real(np.conjugate(alpha) * alpha)
        dbeta_d_dt = (dq_d_dt + 1j * dp_d_dt) / 2
        # # alternate expression
        # dbeta_d_dt = - 1j * G / np.sqrt(2) * np.conjugate(alpha) * alpha - 2j * g_tilde * N * np.conjugate(betas[0]) - gamma_m * betas[1] - 1j * (omegas[1] + 2 * temp) * betas[1]

        # mode rates
        mode_rates = [dalpha_dt, dbeta_c_dt, dbeta_d_dt]

        return mode_rates
        
    def get_N_o_norms(self, params):
        """Method to obtain the mean optical occupancy per oscillation.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
        
        Returns
        -------
        N_o_norms : list
            Mean optical occupancies per oscillation.
        """

        # get mean optical occupancies
        N_os, _ = self.get_mean_optical_occupancies(method='cubic' if 'cubic' in params[14] else 'basic')

        # calculate mean optical occupancies per oscillation
        N_o_norms = [N_o * params[7] / params[6] for N_o in N_os]

        return N_o_norms
        
    def get_N_os(self, c):
        """Method to obtain the mean optical occupancy.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
        
        Returns
        -------
        N_os : list
            Mean optical occupancies.
        """

        # get mean optical occupancies
        N_os, _ = self.get_mean_optical_occupancies()

        return N_os

    def get_params_steady_state(self, c):
        r"""Method to obtain the arguments required to calculate the optical steady state.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
        
        Returns 
        -------
        eta_lc : float
            Amplitude of the control laser.
        Delta_tilde : float
            Detuning of the laser.
        gamma_o : float
            Optical decay rate.
        C : float
            Coefficient of :math:`|\alpha_{s}|^{2}`.
        """

        # get effective values
        _, _, _, C = self.get_effective_values(
            c=c
        )

        return c[2], c[0], c[7], C
    
    def get_coeffs_N_o(self, c):
        """Method to obtain coefficients of the polynomial in mean optical occupancy.
        
        Parameters
        ----------
        c : numpy.ndarray
            Derived constants and controls.
        
        Returns 
        -------
        coeffs : numpy.ndarray
            Coefficients of the polynomial in mean optical occupancy.
        """

        # frequently used variables
        A_l_norm, Delta_0_norm, kappa_norm, C = self.get_params_steady_state(
            c=c
        )
        
        # get coefficients
        coeffs = np.zeros(2 * self.num_modes, dtype=np.float_)
        coeffs[0] = 4.0 * C**2
        coeffs[1] = 8.0 * C * Delta_0_norm
        coeffs[2] = 4.0 * Delta_0_norm**2 + kappa_norm**2
        coeffs[3] = - 4.0 * np.real(np.conjugate(A_l_norm) * A_l_norm)

        return coeffs

    def get_oss_modes(self, c):
        """Method to obtain the steady state optical and mechanical position mode apmlitudes.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
        
        Returns 
        -------
        Modes : list
            List of optical and mechanical mode amplitudes.
        """

        # extract frequently used variables
        Delta_tilde = c[0]
        eta_lc = c[2]
        G = c[4]
        gamma_o = c[7]
        t_oss_method = self.params['t_oss_method']

        # get effective values
        A_mathcal, Omegas, omega_tildes, C = self.get_effective_values(
            c=c
        )

        # frequently used expressions
        A2 = A_mathcal**2 + Omegas[0]**2 * Omegas[1]**2
        A2_sqrt = np.sqrt(A2)

        # initialize lists
        Modes = list()

        # get mean optical occupancies
        N_os, _ = self.get_mean_optical_occupancies()
        # for each mean optical occupancy
        for N_o in N_os:
            # calculate mode amplitudes
            alpha = eta_lc / (gamma_o / 2 - 1j * (Delta_tilde + (C * N_o if 'cubic' in t_oss_method else 0)))
            q_c = - G * omega_tildes[1] * N_o / A2_sqrt
            q_d = - G * omega_tildes[0] * N_o / A2_sqrt
            # # alternate expressions
            # q_c = - G * (Omegas[1]**2 * omega_tildes[0] - A_mathcal * omega_tildes[1]) * N_o / A2
            # q_d = - G * (Omegas[0]**2 * omega_tildes[1] + A_mathcal * omega_tildes[0]) * N_o / A2

            # append to list
            Modes.append([alpha, q_c, q_d])

        return Modes

    def get_peaks(self, c):
        """Method to obtain the normalized peak positions.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.

        Returns
        -------
        peaks : list
            Normalized peak positions.
        """

        # get normalized roots of offset delta
        deltas = self.get_deltas(
            c=c
        )

        # get normalized peak positions
        peaks = np.real(deltas).tolist()

        return peaks

    def get_Ss(self, c):
        """Method to obtain the shot noise, radiaton pressure noise and thermal noise.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.

        Returns
        -------
        S : list
            Shot noise, radiaton pressure noise and thermal noise.
        """
        # frequently used variables
        delta = c[1]
        G = c[4]
        gamma_m = c[6]
        gamma_o = c[7]
        t_oss_method = self.params['t_oss_method']
        T = 20e-9

        # get effective values
        A_mathcal, Omegas, omega_tildes, C = self.get_effective_values(
            c=c
        )

        # get mean occupancy amplitude
        alpha_ss, _ = self.get_mean_optical_amplitudes(method='cubic' if 'cubic' in t_oss_method else 'basic')
        alpha_s_sq = np.real(np.conjugate(alpha_ss[0]) * alpha_ss[0])

        # shot noise
        S_sn = (delta**2 + gamma_o**2 / 4) / 4 / G**2 / gamma_o / alpha_s_sq

        # radiation pressure noise
        _chis = [1 / (Omegas[i]**2 - 1j * delta * gamma_m - delta**2) for i in range(2)]
        _F = C**2 / G**4 * np.real(np.conjugate(Omegas[0] * _chis[0]) * Omegas[0] * _chis[0]) * np.real(np.conjugate(Omegas[1] * _chis[1]) * Omegas[1] * _chis[1]) * ((delta**2 - Omegas[0] * Omegas[1])**2 + gamma_m**2 * delta**2)
        S_rp = gamma_o * G**2 * _F * alpha_s_sq / (delta**2 + gamma_o**2 / 4)

        # thermal noise
        S_th = gamma_m * delta * (Omegas[0] * np.real(np.conjugate(_chis[0]) * _chis[0]) + Omegas[1] * np.real(np.conjugate(_chis[1]) * _chis[1])) / np.tanh(sc.hbar * delta / 2 / sc.k / T)
        
        return [S_sn, S_rp, S_th]

    def get_transmission(self, c):
        """Method to obtain the transmission.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.

        Returns
        -------
        T : float
            Transmission.
        """

        # get transmission coefficient
        # anti-Stokes field
        if self.params['t_line'] == 'as':
            _, _t = self.get_transmission_coeffs(
                c=c
            )
        # Stokes field
        else:
            _t, _ = self.get_transmission_coeffs(
                c=c
            )

        # calculate transmission
        T = np.real(np.conjugate(_t) * (_t))

        return T

    def get_transmission_resonance(self, params):
        """Method to obtain the transmission at resonance using the analytical expression.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
            
        Returns
        -------
        T : float
            Transmission.
        """

        # get cooperativity
        C_mathcal = self.get_cooperativity(params)

        # calculate transmission
        T = (C_mathcal / (1 + C_mathcal))**2

        return T

    def get_transmission_coeffs(self, c):
        """Method to obtain the transmission coefficient for the probe field of the system.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.
        
        Returns
        -------
        t_s : complex
            Transmission coefficient for Stokes field.
        t_as : complex
            Transmission coefficient for anti-Stokes field.
        """

        # extract frequently used variables
        Delta_tilde = c[0]
        delta = c[1]
        eta_lc = c[2]
        G = c[4]
        gamma_m = c[6]
        gamma_o = c[7]
        mu = c[8]
        t_approx = self.params['t_approx']
        t_oss_method = self.params['t_oss_method']

        # get effective values
        A_mathcal, Omegas, omega_tildes, C = self.get_effective_values(
            c=c
        )

        # get mean occupancy amplitude
        if 'cubic' in t_oss_method:
            N_os = self.get_mean_optical_occupancies()
            alpha_s = eta_lc / (gamma_o / 2.0 - 1.0j * (Delta_tilde + C * N_os[0]))
        else:
            alpha_s = eta_lc / (gamma_o / 2.0 - 1.0j * Delta_tilde)

        # # display steady-state values
        # logger.debug('alpha_s={} alpha_s_basic={}\n'.format(alpha_s, eta_lc / (gamma_o / 2 - 1j * Delta_tilde)))

        # mean optical occupancy
        N_o = np.real(np.conjugate(alpha_s) * alpha_s)

        # effiective detuning
        Delta = Delta_tilde + (C * N_o if 'cubic' in t_oss_method else 0)

        # effective decays
        Gamma_m = gamma_o / 2 - 1j * (Delta + delta)
        Gamma_p = gamma_o / 2 + 1j * (Delta - delta)

        # approximation for delta
        if 'del' in t_approx:
            delta_approx = - Delta_tilde
            if delta_approx >= 0:
                chis =  [1 / ((Omegas[i] - delta) * (Omegas[i] + delta_approx) - 1j * delta_approx * gamma_m) for i in range(2)]
            else:
                chis =  [1 / ((Omegas[i] - delta_approx) * (Omegas[i] + delta) - 1j * delta_approx * gamma_m) for i in range(2)]
        else:
            chis =  [1 / (Omegas[i]**2 - 1j * delta * gamma_m - delta**2) for i in range(2)]

        # substitution term
        _num = A_mathcal * chis[0] * chis[1] * (omega_tildes[0] - omega_tildes[1])
        _num += sum([chis[i] * omega_tildes[i] for i in range(2)])
        _den = A_mathcal**2 * chis[0] * chis[1] + 1
        Lambda = _num / _den

        # resolved sideband approximation
        if 'res' in t_approx:
            _num_s = mu * gamma_o
            _num_as = 1
            _den = Gamma_m - 1j * G**2 * Lambda * N_o
        else:
            _num_s = mu * gamma_o * (Gamma_p + 1j * G**2 * Lambda * N_o)
            _num_as = mu * gamma_o * (- 1j * G**2 * Lambda * np.conjugate(alpha_s)**2)
            _den = Gamma_m * Gamma_p + 2 * Delta * G**2 * Lambda * N_o

        # transmission coefficients
        t_s = 1 - _num_s / _den
        t_as = - np.conjugate(_num_as / _den)

        return t_s, t_as

    def get_transmission_phase(self, c):
        """Method to obtain the phase of transmission phase.
        
        Parameters
        ----------
        params : list
            Constant parameters of the system.

        Returns
        -------
        phi : float
            Transmission phase.
        """

        # get transmission coefficient
        # anti-Stokes field
        if self.params['t_line'] == 'as':
            _, _t = self.get_transmission_coeffs(
                c=c
            )
        # Stokes field
        else:
            _t, _ = self.get_transmission_coeffs(
                c=c
            )

        # calculate transmission phase
        phi = np.angle(_t)

        return phi