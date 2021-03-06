#pragma once
#include "NuclearData.hpp" //for Isotope
#include <string>
#include <vector>
class Grid;

namespace Nuclear {

enum class Type { Fermi, spherical, point };

Type parseType(const std::string &str_type);
std::string parseType(Type type);

struct Parameters {
  int z, a;
  Nuclear::Type type;
  double t;
  double r_rms;
  Parameters(const std::string &z_str, int in_a,
             const std::string &str_type = "Fermi", double in_rrms = -1.0,
             double in_t = -1.0);
  Parameters(int in_z, int in_a, const std::string &str_type = "Fermi",
             double in_rrms = -1.0, double in_t = -1.0);
};

std::vector<double> sphericalNuclearPotential(double Z, double rnuc,
                                              const std::vector<double> &rgrid);

std::vector<double> fermiNuclearPotential(double Z, double t, double c,
                                          const std::vector<double> &rgrid);

std::vector<double> fermiNuclearDensity_tcN(double t, double c, double Z_norm,
                                            const Grid &grid);

std::vector<double> formPotential(Parameters params, int z, int,
                                  const std::vector<double> &r);

} // namespace Nuclear
