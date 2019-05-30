#pragma once
#include "FPC_physicalConstants.hpp"
#include "NumCalc_quadIntegrate.hpp"
#include <cmath>
#include <gsl/gsl_sf_fermi_dirac.h>
#include <vector>
class Grid;

/*
Add data tables of change radii ? etc.
e.g.: https://www-nds.iaea.org/radii/ (or Mathematrica?)
*/

namespace Nucleus {

//******************************************************************************
inline double approximate_r_rms(int A)
// Returns approximate root-mean-square charge radius in fm [1.e-15 m]
// https://www.sciencedirect.com/science/article/pii/S0092640X12000265
// https://www-nds.iaea.org/radii/
// Few light elements (H, He, Li) are hard-coded for common A
// Also: Cs-133 is hard-coded.
{
  double rN;
  if (A == 1)
    rN = 0.8791; // 1-H
  else if (A == 4)
    rN = 1.6757; // 4-He
  else if (A == 7)
    rN = 2.4312; // 7-Li
  else if (A < 10)
    rN = 1.15 * pow(A, 0.333);
  else if (A == 133) // 133-Cs
    rN = 4.8041;
  else
    rN = 0.836 * pow(A, 0.333) + 0.570;

  return rN;
}

//******************************************************************************
inline double c_hdr_formula_rrms_t(double rrms, double t = 2.3)
// Calculates half-density radius, given rms charge radius, and t.
// Formula from Ginges, Volotka, Fritzsche, Phys. Rev. A 96, 1 (2017).
// 4 ln(3) = 4.39445, pi^2 = 9.87
{
  double a = t / 4.39445;
  return sqrt((5. / 3) * rrms * rrms - (7. / 3) * (9.87 * a * a));
}

//******************************************************************************
inline double approximate_c_hdr(int A, double t = 2.3)
// approximate formula for half-density radius (c) - used for Fermi distro
// nb: returns in Fermi
{
  double r = approximate_r_rms(A);
  return c_hdr_formula_rrms_t(r, t);
}

//******************************************************************************
inline double approximate_t_skin(int)
// skin-thickness. Always same?
{
  return 2.30;
}

//******************************************************************************
inline std::vector<double>
sphericalNuclearPotential(double Z, double rnuc,
                          const std::vector<double> &rgrid)
// Potential due to a spherical nucleus, with (charge) radius, rnuc.
// Note: rnuc must be given in "fermi" (fm, femto-metres).
// rnuc = 0 corresponds to zeroNucleus
{
  std::vector<double> vnuc;
  vnuc.reserve(rgrid.size());

  double rN = rnuc / FPC::aB_fm;

  // Fill the vnuc array with spherical nuclear potantial
  double rn2 = pow(rN, 2);
  double rn3 = pow(rN, 3);
  for (auto r : rgrid) {
    double temp_v = (r < rN) ? Z * (r * r - 3. * rn2) / (2. * rn3) : -Z / r;
    vnuc.push_back(temp_v);
  }

  return vnuc;
}

//******************************************************************************
inline std::vector<double>
fermiNuclearPotential(double Z, double t, double c,
                      const std::vector<double> &rgrid)
// Uses a Fermi-Dirac distribution for the nuclear potential.
//
// rho(r) = rho_0 {1 + Exf[(r-c)/a]}^-1
// V(r) = -(4 Pi)/r [A+B]
//   A = Int[ rho(x) x^2 , {x,0,r}]
//   B = r * Int[ rho(x) x , {x,r,infty}]
// rho_0 is found by either:
//   * V(infinity) = -Z/r , or equivilantly
//   * \int rho(r) d^3r = Z
//
// Depends on:
//   * t: skin thickness [90 to 10% fall-off range]
//     note: t = a[4 ln(3)]
//   * c: half-density raius [rho(c)=0.5 rho0]
//
// t and c are input values. In 'fermi' of fm (femto metres)
// If provided with 0, will use 'default' values, approx. formula.
//
// V(r) is expressed in terms of Complete Fermi-Dirac intagrals.
// These are computed using the GSL libraries.
// gnu.org/software/gsl/manual/html_node/Complete-Fermi_002dDirac-Integrals
{
  std::vector<double> vnuc;
  vnuc.reserve(rgrid.size());

  double a = 0.22756 * t; // t = a*[4 ln(3)]
  double coa = c / a;
  // Use GSL for the Complete Fermi-Dirac Integrals:
  double F2 = gsl_sf_fermi_dirac_2(coa);
  double pi2 = pow(M_PI, 2);
  for (auto r : rgrid) {
    double t_v = -Z / r;
    double roa = FPC::aB_fm * r / a; // convert fm <-> atomic
    if (roa < 30. + coa) {
      double roa = FPC::aB_fm * r / a; // convert fm <-> atomic
      double coa2 = pow(coa, 2);
      double xF1 = gsl_sf_fermi_dirac_1(roa - coa);
      double xF2 = gsl_sf_fermi_dirac_2(roa - coa);
      double tX = -pow(roa, 3) - 2 * coa * (pi2 + coa2) +
                  roa * (pi2 + 3 * coa2) + 6 * roa * xF1 - 12 * xF2;
      t_v += t_v * tX / (12. * F2);
    }
    vnuc.push_back(t_v);
  }

  return vnuc;
} // namespace Nucleus

//******************************************************************************
inline std::vector<double>
fermiNuclearDensity_tcN(double t, double c, double Z_norm, const Grid &grid)
// Integrate[ rho(r) , dV ] = Integrate[ 4pi * r^2 * rho(r) , dr ] = Z_norm
// Znorm = Z for nuclear chare density; Z_norm = 1 for nuclear density.
{
  std::vector<double> rho;
  rho.reserve(grid.ngp);

  double a = 0.22756 * t;
  double coa = c / a;
  for (auto r : grid.r) {
    double roa = FPC::aB_fm * r / a;
    if (roa < 30. + coa) {
      rho.emplace_back(1. / (1. + exp(roa - coa)));
    } else {
      rho.push_back(0.);
    }
  }

  double Norm =
      NumCalc::integrate(grid.r, grid.r, rho, grid.drdu, grid.du) * 4. * M_PI;
  double rho0 = Z_norm / Norm;

  for (auto &rhoi : rho) {
    rhoi *= rho0;
  }

  return rho;
}

} // namespace Nucleus