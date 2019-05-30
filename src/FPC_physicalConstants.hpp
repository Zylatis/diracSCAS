#pragma once
/*
Constains physical constants, and units conversions.
Taken mostly from 2014 CODATA values.
https://physics.nist.gov/cuu/Constants/
https://en.wikipedia.org/wiki/Atomic_units
*/

namespace FPC {

// speed of light in a.u., and fine-structure constant
const double c = 137.035999139; // CODATA 2014: 137.035999139(31)
const double c2 = c * c;
const double alpha = 1. / c; // fine structure constant
const double alpha2 = alpha * alpha;

// speed of light, in m/s
const double c_SI = 299792458.;

// Proton mass (mp/me)
const double m_p = 1836.15267389;     // CODATA 2014: 1836.152 673 89(17)
const double m_e_MeV = 0.5109989461;  // MeV/c^2 - electron mass
const double m_e_kg = 9.10938356e-31; // CODATA 2014: 9.109 383 56(11) e-31

//"unified atomic mass" unit; nuclear mass unit; Dalton; u
const double u_NMU = 1822.888486192; // CODATA 2014: 1822.888 486 192(53)

// Length:
const double aB_m = 0.52917721067e-10; // CODATA 2014: 0.52917721067(12)e-10 m
const double aB_cm = 0.52917721067e-8;
const double aB_fm = 0.52917721067e+5;

// Time:
const double time_s = 2.418884326505e-17; // wiki: 2.418884326505(16)×10−17 s

// Energy:
const double Hartree_eV = 27.21138602;        // CODATA 2014: 27.21138602(17) eV
const double Hartree_Hz = 6.579683920711e+15; // 6.579683920711(39)e15 Hz
const double Hartree_MHz = 6.579683920711e+9;
const double Hartree_GHz = 6.579683920711e+6;
// wave-number (inverse cm):
const double Hartree_invcm = 2.194746313702e+5; // 2.194746313702(13)e7 m-1
// wavelength (nm):
const double HartreeWL_nm = 45.56335252767;

// Fermi weak constant (au)
const double GFe11 = 2.2225e-3;
const double GF = GFe11 * (1e-11);

// Bohr magneton (in atomic units):
const double muB_SI = 0.5;          // SI-derived
const double muB_CGS = 0.5 * alpha; // Gaussian CGS-derived
// Nulcear magneton (in atomic units):
const double muN_SI = muB_SI / m_p;
const double muN_CGS = muB_CGS / m_p;

} // namespace FPC