#include "AtomInfo.hpp"
#include "ChronoTimer.hpp"
#include "DiracOperator.hpp"
#include "ElectronOrbitals.hpp"
#include "FileIO_fileReadWrite.hpp"
#include "HartreeFockClass.hpp"
#include "Nucleus.hpp"
#include "NumCalc_quadIntegrate.hpp"
#include "Operators.hpp"
#include "Parametric_potentials.hpp"
#include "PhysConst_constants.hpp"
#include <cmath>
#include <iostream>
#include <tuple>
//
#include "CoulombIntegrals.hpp" //for testing

int main(int argc, char *argv[]) {
  ChronoTimer timer; // start the overall timer

  std::string input_file = (argc > 1) ? argv[1] : "hartreeFock.in";
  std::cout << "Reading input from: " << input_file << "\n";

  // Input options
  std::string Z_str;
  int A;
  std::string str_core;
  double r0, rmax;
  int ngp;
  double eps_HF;      // HF convergance
  int num_val, l_max; // valence states to calc
  double varalpha, varalpha2;
  bool exclude_exchange;
  bool run_test;

  { // Open and read the input file:
    int i_excl_ex, i_tests;
    auto tp =
        std::forward_as_tuple(Z_str, A, str_core, r0, rmax, ngp, eps_HF,
                              num_val, l_max, varalpha2, i_excl_ex, i_tests);
    FileIO::setInputParameters(input_file, tp);
    exclude_exchange = i_excl_ex == 1 ? true : false;
    run_test = i_tests == 1 ? true : false;
  }

  // Change varAlph^2 to varalph
  if (varalpha2 == 0)
    varalpha2 = 1.e-10;
  varalpha = sqrt(varalpha2);
  int Z = AtomInfo::get_z(Z_str);

  // Generate the orbitals object:
  ElectronOrbitals wf(Z, A, ngp, r0, rmax, varalpha);
  // Print info to screen:
  if (exclude_exchange)
    std::cout << "\nRunning Hartree (excluding exchange) for ";
  else
    std::cout << "\nRunning Hartree-Fock for " << wf.atom() << "\n";
  std::cout << wf.nuclearParams() << "\n"
            << wf.rgrid.gridParameters() << "\n"
            << "********************************************************\n";

  // Solve Hartree equations for the core:
  timer.start(); // start the timer for HF
  HartreeFock hf(wf, str_core, eps_HF, exclude_exchange);
  double core_energy = hf.calculateCoreEnergy();
  std::cout << "core: " << timer.lap_reading_str() << "\n";

  // Create list of valence states to solve for
  if ((int)wf.Ncore() >= wf.Znuc())
    num_val = 0;
  auto val_lst = wf.listOfStates_nk(num_val, l_max);

  // Solve for the valence states:
  timer.start();
  for (const auto &nk : val_lst) {
    int n = nk[0];
    int k = nk[1];
    hf.solveNewValence(n, k);
  }
  // wf.orthonormaliseOrbitals(wf.valence_orbitals, 2);
  // Note: it's not correct to orthogonalise valence states like this,
  // since if we do, the orbitals will depend (weakly) on which other valence
  // states we calculate!
  if (val_lst.size() > 0)
    std::cout << "Valence: " << timer.lap_reading_str() << "\n";

  // Output results:
  std::cout << "\nHartree Fock: " << wf.atom() << "\n";
  // wf.printAtom();
  bool sorted = true;
  wf.printCore(sorted);
  std::cout << "E_core = " << core_energy
            << " au;  = " << core_energy * PhysConst::Hartree_invcm << "/cm\n";
  wf.printValence(sorted);

  std::cout << "\n Total time: " << timer.reading_str() << "\n";

  //*********************************************************
  //               TESTS
  //*********************************************************

  if (run_test) {
    std::cout << "Test orthonormality [log-scale, should all read 0]:\n";
    for (int i = 0; i < 3; i++) {
      const auto &tmp_b = (i == 2) ? wf.valence_orbitals : wf.core_orbitals;
      const auto &tmp_a = (i == 0) ? wf.core_orbitals : wf.valence_orbitals;
      // Core-Core:
      if (i == 0)
        std::cout << "\nCore-Core\n    ";
      else if (i == 1)
        std::cout << "\nValence-Core\n    ";
      else
        std::cout << "\nValence-Valence\n    ";
      for (auto &psi_b : tmp_b)
        printf("%2i%2i", psi_b.n, psi_b.k);
      std::cout << "\n";
      for (auto &psi_a : tmp_a) {
        printf("%2i%2i", psi_a.n, psi_a.k);
        for (auto &psi_b : tmp_b) {
          if (psi_b > psi_a) {
            std::cout << "    ";
            continue;
          }
          if (psi_a.k != psi_b.k) {
            std::cout << "    ";
            continue;
          }
          double xo = (psi_a * psi_b);
          if (psi_a.n == psi_b.n)
            xo -= 1;
          if (xo == 0)
            printf("   0");
          else
            printf(" %+3.0f", log10(fabs(xo)));
        }
        std::cout << "\n";
      }
    }
    std::cout << "\n(Note: Core orbitals are orthogonalised explicitely, as is "
                 "each valence state [with respect to the core]. However, "
                 "valence states are not explicitely orthogonalised wrt each "
                 "other, since there's no self-consistent way to do this with "
                 "a finite set of valence orbitals).\n";

    std::cout << "\nTesting wavefunctions: <n|H|n>  (numerical error)\n";
    double c = 1. / wf.get_alpha();
    DiracOperator w(c, GammaMatrix::g5, 1, true);
    RadialOperator x_a(wf.rgrid, -1);
    DiracOperator y(c * c, DiracMatrix(0, 0, 0, -2));
    DiracOperator z1(wf.vnuc);
    DiracOperator z2(wf.vdir);
    bool tmp_valence = false;
    for (auto &tmp_orbs : {wf.core_orbitals, wf.valence_orbitals}) {
      for (auto &psi : tmp_orbs) {
        auto k = psi.k;
        DiracOperator z3(hf.get_vex(psi, tmp_valence));
        DiracOperator x_b(c, DiracMatrix(0, 1 - k, 1 + k, 0), 0, true);
        auto rhs = (w * psi) + (x_a * (x_b * psi)) + (y * psi) + (z1 * psi) +
                   (z2 * psi) + (z3 * psi);
        double R = psi * rhs;
        double ens = psi.en;
        double fracdiff = (R - ens) / ens;
        printf("<%i% i|H|%i% i> = %17.11f, E = %17.11f; % .0e\n", psi.n, psi.k,
               psi.n, psi.k, R, ens, fracdiff);
      }
      tmp_valence = true;
    }
    std::cout << "\n Total time: " << timer.reading_str() << "\n";
  }

  bool print_wfs = false;
  if (print_wfs) {
    std::ofstream of("hf-orbitals.txt");
    of << "r ";
    for (auto &psi : wf.core_orbitals)
      of << "\"" << psi.symbol(true) << "\" ";
    for (auto &psi : wf.valence_orbitals)
      of << "\"" << psi.symbol(true) << "\" ";
    of << "\n";
    for (std::size_t i = 0; i < wf.rgrid.ngp; i++) {
      of << wf.rgrid.r[i] << " ";
      for (auto &psi : wf.core_orbitals)
        of << psi.f[i] << " ";
      for (auto &psi : wf.valence_orbitals)
        of << psi.f[i] << " ";
      of << "\n";
    }
  }

  bool testpnc = false;
  if (testpnc) {
    double t = 2.3;
    double c = Nucleus::approximate_c_hdr(wf.Anuc());
    PNCnsiOperator hpnc(c, t, wf.rgrid, -wf.Nnuc());
    E1Operator he1(wf.rgrid);

    double Ac = 2. / 6.; // angular coef
    auto a6s_i = wf.getStateIndex(6, -1);
    auto a7s_i = wf.getStateIndex(7, -1);
    auto &a6s = wf.valence_orbitals[a6s_i];
    auto &a7s = wf.valence_orbitals[a7s_i];
    std::cout << "E_pnc: " << wf.Anuc() << "-"
              << AtomInfo::atomicSymbol(wf.Znuc()) << " " << a6s.symbol()
              << " -> " << a7s.symbol() << "\n";

    double pnc = 0;
    for (int i = 0; i < 2; i++) {
      auto &tmp_orbs = (i == 0) ? wf.core_orbitals : wf.valence_orbitals;
      for (auto &np : tmp_orbs) {
        if (np.k != 1)
          continue; // p_1/2 only
        // <7s|d|np><np|hw|6s>/dE6s + <7s|hw|np><np|d|6s>/dE7s
        double pnc1 =
            Ac * (a7s * (he1 * np)) * (np * (hpnc * a6s)) / (a6s.en - np.en);
        double pnc2 =
            Ac * (a7s * (hpnc * np)) * (np * (he1 * a6s)) / (a7s.en - np.en);
        std::cout << "n=" << np.n << " pnc= " << pnc1 << " + " << pnc2 << " = "
                  << pnc1 + pnc2 << "\n";
        pnc += pnc1 + pnc2;
      }
    }
    std::cout << "Total= " << pnc << "\n";
    std::cout << "\n Total time: " << timer.reading_str() << "\n";
  }

  bool test_hfs = false;
  if (test_hfs) {
    // Test hfs and Operator [hard-coded for Rb]
    double muN = 2.751818;                  // XXX Rb
    double IN = (3. / 2.);                  // XXX Rb
    auto r_rms = 4.1989 / PhysConst::aB_fm; // XXX Rb
    // auto r_rms = Nucleus::approximate_r_rms(wf.Anuc());
    std::cout << "Gridpoints below Rrms: " << wf.rgrid.getIndex(r_rms) << "\n";

    // example for using lambda
    auto l1 = [](double r, double) { return 1. / (r * r); };
    // auto l1 = [](double r, double rN) { return r > rN ? 1. / (r * r) : 0.;
    // };
    HyperfineOperator vhfs(muN, IN, r_rms, wf.rgrid, l1);
    for (auto phi : wf.valence_orbitals) {
      auto A_tmp = phi * (vhfs * phi);
      double j = phi.j();
      auto factor = PhysConst::Hartree_MHz * phi.k / (j * (j + 1.));
      std::cout << phi.symbol() << ": ";
      std::cout << A_tmp * factor << "\n";
    }
  }

  //**************************************************************************
  // Playing with RPA.
  // a) Extremely slow.
  //    Many ineficiencies, including Z calculation.
  //    Not taking advatage of any symmetries, and calculating Z EVERY TIME!
  // b) Is the sign wrong?
  // c) Also: correction is order-of-magnitude too small
  //    - is this due to shitty basis? Or actually incorrect somewhere?

  bool test_hf_basis = true;
  std::vector<DiracSpinor> v_basis;
  if (test_hf_basis) {
    auto basis_lst = wf.listOfStates_nk(15, 2);
    for (const auto &nk : basis_lst) {
      v_basis.emplace_back(DiracSpinor(nk[0], nk[1], wf.rgrid));
      auto tmp_vex = std::vector<double>{};
      hf.solveValence(v_basis.back(), tmp_vex);
    }
    wf.orthonormaliseOrbitals(v_basis, 2);
    wf.printValence(false, v_basis);
    std::cout << "\n Total time: " << timer.reading_str() << "\n";
  }

  Coulomb cint(wf.core_orbitals, v_basis);
  cint.form_core_valence();

  ChronoTimer sw;
  sw.start();
  // const auto &psi_a = wf.core_orbitals.front();
  E1Operator he1(wf.rgrid);

  std::vector<std::vector<double>> t0_am;
  t0_am.reserve(wf.core_orbitals.size());
  for (const auto &psi_a : wf.core_orbitals) {
    std::vector<double> t0a_m;
    t0a_m.reserve(v_basis.size());
    for (const auto &psi_m : v_basis) {
      auto radInt = psi_a * (he1 * psi_m);
      auto Cc = Wigner::Ck_kk(1, psi_a.k, psi_m.k);
      t0a_m.push_back(radInt * Cc);
    }
    t0_am.push_back(t0a_m);
  }
  auto t_am = t0_am;

  const auto &psi_v = wf.valence_orbitals[0];
  const auto &psi_w = wf.valence_orbitals[1];
  double omega = fabs(psi_v.en - psi_w.en);
  double tvw_0 = Wigner::Ck_kk(1, psi_v.k, psi_w.k) * (psi_v * (he1 * psi_w));
  double tvw_rpa = tvw_0;

  for (int i = 0; i < 5; i++) {
    double max = 0;
    for (std::size_t ia = 0; ia < wf.core_orbitals.size(); ia++) {
      const auto &psi_a = wf.core_orbitals[ia];
      for (std::size_t im = 0; im < v_basis.size(); im++) {
        const auto &psi_m = v_basis[im];
        double sum = 0;
        std::size_t ib = 0;
        for (const auto &psi_b : wf.core_orbitals) {
          std::size_t in = 0;
          for (const auto &psi_n : v_basis) {
            auto f =
                -(1. / 3.) * pow(-1, (psi_b.twoj() - psi_n.twoj()) / 2 + 1);
            auto Zanmb = cint.calculate_Z_abcdk(psi_a, psi_n, psi_m, psi_b, 1);
            auto Zabmn = cint.calculate_Z_abcdk(psi_a, psi_b, psi_m, psi_n, 1);
            auto t_bn = t_am[ib][in];
            auto t_nb = pow(-1, (psi_n.twoj() - psi_b.twoj()) / 2) * t_bn;
            auto A = t_bn * Zanmb / (psi_b.en - psi_n.en - omega);
            auto B = t_nb * Zabmn / (psi_b.en - psi_n.en + omega);
            // std::cout << A << " " << B << "\n";
            sum += f * (A + B);
            ++in;
          }
          ++ib;
        }
        // auto tam_old = t_am[ia][im];
        // t_am[ia][im] = tam_old + 0.5 * sum;
        t_am[ia][im] = t0_am[ia][im] + sum;
        if (fabs(sum) > max)
          max = fabs(sum);
        // std::cout << tam_old << " " << tam_old + sum << " " << sum / tam_old
        //           << "\n";
      }
    }
    std::cout << max << "\n";

    // valence:
    {
      double sum = 0;
      std::size_t ia = 0;
      for (const auto &psi_a : wf.core_orbitals) {
        std::size_t im = 0;
        for (const auto &psi_m : v_basis) {
          auto f = -(1. / 3.) * pow(-1, (psi_a.twoj() - psi_m.twoj()) / 2 + 1);
          auto Zwmva = cint.calculate_Z_abcdk(psi_w, psi_m, psi_v, psi_a, 1);
          auto Zwavm = cint.calculate_Z_abcdk(psi_w, psi_a, psi_v, psi_m, 1);
          auto tt_am = t_am[ia][im];
          auto tt_ma = pow(-1, (psi_m.twoj() - psi_a.twoj()) / 2) * tt_am;
          auto A = tt_am * Zwmva / (psi_a.en - psi_m.en - omega);
          auto B = tt_ma * Zwavm / (psi_a.en - psi_m.en + omega);
          // std::cout << A << " " << B << "\n";
          sum += f * (A + B);
          ++im;
        }
        ++ia;
      }
      tvw_rpa = tvw_0 + sum;
      std::cout << tvw_0 << " " << tvw_rpa << " " << sum / tvw_0 << "\n";
    }
  }
  std::cout << sw.lap_reading_str() << "\n";

  // // cint.form_core_core();

  // // cint.form_valence_valence();
  //
  // for (const auto &psi_c : wf.core_orbitals) {
  //   for (const auto &psi_v : wf.valence_orbitals) {
  //     std::cout << psi_v.symbol() << "|" << psi_c.symbol() << ": ";
  //     auto R = cint.calculate_R_abcd_k(psi_c, psi_v, psi_v, psi_c);
  //     for (auto r : R)
  //       printf("%9.4e  ", r);
  //     std::cout << "\n";
  //   }
  //   std::cout << "\n";
  // }

  return 0;
}
