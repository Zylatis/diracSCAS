#include "Dirac/Wavefunction.hpp"
#include "IO/ChronoTimer.hpp"
#include "IO/UserInput.hpp"
#include "Maths/Grid.hpp"
#include "Modules/Module_runModules.hpp"
#include "Physics/AtomInfo.hpp"
#include "Physics/Nuclear.hpp"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  ChronoTimer timer("\nhartreeFock");
  std::string input_file = (argc > 1) ? argv[1] : "hartreeFock.in";
  std::cout << "Reading input from: " << input_file << "\n";

  // Input options
  UserInput input(input_file);

  // Get + setup atom parameters
  auto atom = input.get<std::string>("Atom", "Z");
  auto Z = AtomInfo::get_z(atom);
  auto A = input.get("Atom", "A", -1);
  auto varAlpha2 = input.get("Atom", "varAlpha2", 1.0);
  if (varAlpha2 <= 0)
    varAlpha2 = 1.0e-25;
  auto varalpha = std::sqrt(varAlpha2);

  // Get + setup Grid parameters
  auto r0 = input.get("Grid", "r0", 1.0e-5);
  auto rmax = input.get("Grid", "rmax", 150.0);
  auto ngp = input.get("Grid", "ngp", 1600ul);
  auto b = input.get("Grid", "b", 4.0);
  auto grid_type = GridParameters::parseType(
      input.get<std::string>("Grid", "type", "loglinear"));
  GridParameters grid_params(ngp, r0, rmax, b, grid_type);

  // Get + setup nuclear parameters
  A = input.get("Nucleus", "A", A); // over-writes "atom" A
  auto nuc_type =
      Nuclear::parseType(input.get<std::string>("Nucleus", "type", "Fermi"));
  auto rrms = input.get("Nucleus", "rrms", -1.0); /*<0 means lookup default*/
  auto skint = input.get("Nucleus", "skin_t", -1.0);
  Nuclear::Parameters nuc_params(Z, A, nuc_type, rrms, skint);

  // create wavefunction object
  Wavefunction wf(Z, grid_params, nuc_params, varalpha);

  std::cout << "\nRunning for " << wf.atom() << "\n"
            << wf.nuclearParams() << "\n"
            << wf.rgrid.gridParameters() << "\n"
            << "********************************************************\n";

  // Parse input for HF method
  auto str_core = input.get<std::string>("HartreeFock", "core");
  auto eps_HF = input.get("HartreeFock", "convergence", 1.0e-12);
  auto HF_method = HartreeFock::parseMethod(
      input.get<std::string>("HartreeFock", "method", "HartreeFock"));

  // Solve Hartree equations for the core:
  wf.hartreeFockCore(HF_method, str_core, eps_HF);

  // ***************************************************************************
  // ***************************************************************************
  // Need to add to effective polarisation potential to V_dir here
  // V_dir is stored in wf.vdir[i]  (ith array element)
  // V_dir is the "direct" part of the electrostatic potential
  // (not including nuclear potential)
  // The points along radius are stored in wf.rgrid.r[i]
  //  r[i] is value of r at ith point along the grid

  // example reading in input option:
  double input1 = input.get("PolarisationOperator", "myInput1", 0.0);
  double input2 = input.get("PolarisationOperator", "myInput2", 0.0);
  std::cout << "input1 = " << input1 << ", input2 = " << input2 << "\n";

  // auto alpha_d = ...; // this needs to be set
  for (unsigned int i = 0; i < wf.rgrid.ngp; i++) {
    // auto r = wf.rgrid.r[i];
    // wf.vdir[i] += .....; // add polarisation operator here
  }

  // ***************************************************************************
  // ***************************************************************************

  // Solve for the valence states:
  auto valence_list = (wf.Ncore() < wf.Znuc())
                          ? input.get<std::string>("HartreeFock", "valence", "")
                          : "";
  wf.hartreeFockValence(valence_list);

  // Output results:
  std::cout << "\nHartree Fock: " << wf.atom() << "\n";
  auto sorted = input.get("HartreeFock", "sortOutput", true);
  wf.printCore(sorted);
  wf.printValence(sorted);

  // run each of the modules
  Module::runModules(input, wf);

  return 0;
}

//******************************************************************************
