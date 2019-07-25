#pragma once
#include <string>
class Wavefunction;
class UserInputBlock;
class HartreeFock;

namespace Module {

void runModule(const UserInputBlock &input, const Wavefunction &wf,
               const HartreeFock &hf);

void Module_tests(const UserInputBlock &input, const Wavefunction &wf,
                  const HartreeFock &hf);
void Module_Tests_orthonormality(const Wavefunction &wf);
void Module_Tests_Hamiltonian(const Wavefunction &wf, const HartreeFock &hf);

void Module_WriteOrbitals(const UserInputBlock &input, const Wavefunction &wf,
                          const HartreeFock &hf);

} // namespace Module