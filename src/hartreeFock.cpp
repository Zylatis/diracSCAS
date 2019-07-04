#include "AtomInfo.hpp"
#include "ChronoTimer.hpp"
#include "DiracOperator.hpp"
#include "FileIO_fileReadWrite.hpp"
#include "HartreeFockClass.hpp"
#include "Nucleus.hpp"
#include "NumCalc_quadIntegrate.hpp"
#include "Operators.hpp"
#include "Parametric_potentials.hpp"
#include "PhysConst_constants.hpp"
#include "Wavefunction.hpp"
#include <cmath>
#include <iostream>
#include <tuple>

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

  Wavefunction wf(Z, A, ngp, r0, rmax, varalpha);

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
  if (val_lst.size() > 0)
    std::cout << "Valence: " << timer.lap_reading_str() << "\n";

  // Output results:
  std::cout << "\nHartree Fock: " << wf.atom() << "\n";
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
    for (auto &tmp_orbs : {wf.core_orbitals, wf.valence_orbitals}) {
      for (auto &psi : tmp_orbs) {
        auto k = psi.k;
        DiracOperator z3(hf.get_vex(psi));
        // auto vexPsi = (z3 * psi);
        auto vexPsi = hf.vex_psia(psi);
        DiracOperator x_b(c, DiracMatrix(0, 1 - k, 1 + k, 0), 0, true);
        auto rhs = (w * psi) + (x_a * (x_b * psi)) + (y * psi) + (z1 * psi) +
                   (z2 * psi) + vexPsi;
        double R = psi * rhs;
        double ens = psi.en;
        double fracdiff = (R - ens) / ens;
        printf("<%i% i|H|%i% i> = %17.11f, E = %17.11f; % .0e\n", psi.n, psi.k,
               psi.n, psi.k, R, ens, fracdiff);
      }
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
  // b) Is the sign wrong?
  // c) Also: correction is order-of-magnitude too small
  //    - is this due to shitty basis? Or actually incorrect somewhere?

  bool test_hf_basis = true;
  std::vector<DiracSpinor> v_basis; // = wf.core_orbitals;
  if (test_hf_basis) {
    HartreeFock hfbasis(wf, v_basis, eps_HF);
    hfbasis.verbose = false;
    auto basis_lst = wf.listOfStates_nk(30, 4);
    for (const auto &nk : basis_lst) {
      v_basis.emplace_back(DiracSpinor(nk[0], nk[1], wf.rgrid));
      auto tmp_vex = std::vector<double>{};
      hfbasis.solveValence(v_basis.back(), tmp_vex);
    }
    wf.orthonormaliseOrbitals(v_basis, 2);
    wf.printValence(false, v_basis);
    std::cout << "\n Basis time: " << timer.lap_reading_str() << "\n";
  }

  ChronoTimer sw;
  sw.start();

  std::cout << "Core-valence time: " << sw.lap_reading_str() << "\n";
  sw.start();
  std::cout << "Valence-valence time: " << sw.lap_reading_str() << "\n";

  // const auto &psi_a = wf.core_orbitals.front();
  const auto &psi_v = wf.valence_orbitals[0];
  const auto &psi_w = wf.valence_orbitals[1];
  double omega = 0; // fabs(psi_v.en - psi_w.en);

  E1Operator he1(wf.rgrid);
  // E1Operator_VG he1(wf.rgrid, omega);

  std::vector<std::vector<double>> t0_am;
  std::vector<std::vector<double>> t0_ma;
  for (const auto &psi_a : wf.core_orbitals) {
    std::vector<double> t0a_m;
    for (const auto &psi_m : v_basis) {
      auto radInt = psi_a * (he1 * psi_m);
      auto Cc = Wigner::Ck_kk(1, psi_a.k, psi_m.k);
      t0a_m.push_back(radInt * Cc);
    }
    t0_am.push_back(t0a_m);
  }
  for (const auto &psi_m : v_basis) {
    std::vector<double> t0m_a;
    for (const auto &psi_a : wf.core_orbitals) {
      auto radInt = psi_a * (he1 * psi_m);
      auto C2 = Wigner::Ck_kk(1, psi_m.k, psi_a.k);
      t0m_a.push_back(radInt * C2);
    }
    t0_ma.push_back(t0m_a);
  }
  auto t_am = t0_am;
  auto t_ma = t0_ma;

  // RPA: store Z Coulomb integrals (used only for Core RPA its)
  sw.start();
  std::vector<std::vector<std::vector<std::vector<double>>>> Zanmb;
  std::vector<std::vector<std::vector<std::vector<double>>>> Zabmn;
  Zanmb.resize(wf.core_orbitals.size());
  Zabmn.resize(wf.core_orbitals.size());
#pragma omp parallel for
  for (std::size_t i = 0; i < wf.core_orbitals.size(); i++) {
    const auto &psi_a = wf.core_orbitals[i];
    std::vector<std::vector<std::vector<double>>> Za_nmb;
    std::vector<std::vector<std::vector<double>>> Za_bmn;
    Za_nmb.reserve(v_basis.size());
    Za_bmn.reserve(v_basis.size());
    for (const auto &psi_n : v_basis) {
      std::vector<std::vector<double>> Zan_mb;
      std::vector<std::vector<double>> Zab_mn;
      Zan_mb.reserve(v_basis.size());
      Zab_mn.reserve(v_basis.size());
      for (const auto &psi_m : v_basis) {
        std::vector<double> Zanm_b;
        std::vector<double> Zabm_n;
        Zanm_b.reserve(wf.core_orbitals.size());
        Zabm_n.reserve(wf.core_orbitals.size());
        for (const auto &psi_b : wf.core_orbitals) {
          // auto x = cint.calculate_Z_abcdk(psi_a, psi_n, psi_m, psi_b, 1);
          // auto y = cint.calculate_Z_abcdk(psi_a, psi_b, psi_m, psi_n, 1);
          auto x = Coulomb::calc_Zabcdk_scratch(psi_a, psi_n, psi_m, psi_b, 1);
          auto y = Coulomb::calc_Zabcdk_scratch(psi_a, psi_b, psi_m, psi_n, 1);
          Zanm_b.push_back(x);
          Zabm_n.push_back(y);
        }
        Zan_mb.push_back(Zanm_b);
        Zab_mn.push_back(Zabm_n);
      }
      Za_nmb.push_back(Zan_mb);
      Za_bmn.push_back(Zab_mn);
    }
    Zanmb[i] = Za_nmb;
    Zabmn[i] = Za_bmn;
  }
  std::vector<std::vector<std::vector<std::vector<double>>>> Zmnab;
  std::vector<std::vector<std::vector<std::vector<double>>>> Zmban;
  Zmnab.resize(v_basis.size());
  Zmban.resize(v_basis.size());
#pragma omp parallel for
  for (std::size_t i = 0; i < v_basis.size(); i++) {
    const auto &psi_m = v_basis[i];
    std::vector<std::vector<std::vector<double>>> Za_nmb;
    std::vector<std::vector<std::vector<double>>> Za_bmn;
    for (const auto &psi_n : v_basis) {
      std::vector<std::vector<double>> Zan_mb;
      std::vector<std::vector<double>> Zab_mn;
      for (const auto &psi_a : wf.core_orbitals) {
        std::vector<double> Zanm_b;
        std::vector<double> Zabm_n;
        for (const auto &psi_b : wf.core_orbitals) {
          // auto x = cint.calculate_Z_abcdk(psi_m, psi_n, psi_a, psi_b, 1);
          // auto y = cint.calculate_Z_abcdk(psi_m, psi_b, psi_a, psi_n, 1);
          //
          auto x = Coulomb::calc_Zabcdk_scratch(psi_m, psi_n, psi_a, psi_b, 1);
          auto y = Coulomb::calc_Zabcdk_scratch(psi_m, psi_b, psi_a, psi_n, 1);
          //
          Zanm_b.push_back(x);
          Zabm_n.push_back(y);
        }
        Zan_mb.push_back(Zanm_b);
        Zab_mn.push_back(Zabm_n);
      }
      Za_nmb.push_back(Zan_mb);
      Za_bmn.push_back(Zab_mn);
    }
    Zmnab[i] = Za_nmb;
    Zmban[i] = Za_bmn;
  }
  std::cout << "Populate Z time: " << sw.lap_reading_str() << "\n\n";

  double tvw_0 = Wigner::Ck_kk(1, psi_v.k, psi_w.k) * (psi_v * (he1 * psi_w));
  double tvw_rpa = tvw_0;

  // RPA itterations for core:
  sw.start();
  const int num_its = 99;
  for (int i = 0; i < num_its; i++) {

    double max = 0;
    for (std::size_t ia = 0; ia < wf.core_orbitals.size(); ia++) {
      for (std::size_t im = 0; im < v_basis.size(); im++) {
        double sum_am = 0;
        double sum_ma = 0;
        auto f = (1. / (2 * 1 + 1));
        std::size_t ib = 0;
        for (const auto &psi_b : wf.core_orbitals) {
          std::size_t in = 0;
          for (const auto &psi_n : v_basis) {
            auto s1 =
                ((abs(psi_b.twoj() - psi_n.twoj()) + 2) % 4 == 0) ? 1 : -1;
            auto zanmb = Zanmb[ia][in][im][ib];
            auto zabmn = Zabmn[ia][in][im][ib];
            auto zmnab = Zmnab[im][in][ia][ib];
            auto zmban = Zmban[im][in][ia][ib];

            // const auto &psi_a = wf.core_orbitals[ia];
            // const auto &psi_m = v_basis[im];
            // zmban = Coulomb::calc_Zabcdk_scratch(psi_m, psi_b, psi_a, psi_n,
            // 1);

            //// XXX HERE !
            /*
            const auto &psi_a = wf.core_orbitals[ia];
            const auto &psi_m = v_basis[im];
            zmban = cint.calculate_Z_abcdk(psi_m, psi_b, psi_a, psi_n, 1);
            auto zmban2 = Coulomb::calc_Zabcdk_scratch(psi_m, psi_b,
                                                             psi_a, psi_n, 1);
            if (zmban - zmban2 != 0) {
              std::cout << zmban - zmban2 << "\n";
            }
            */

            auto t_bn = t_am[ib][in];
            auto t_nb = t_ma[in][ib];
            auto A = t_bn * zanmb / (psi_b.en - psi_n.en - omega);
            auto B = t_nb * zabmn / (psi_b.en - psi_n.en + omega);
            auto C = t_bn * zmnab / (psi_b.en - psi_n.en - omega);
            auto D = t_nb * zmban / (psi_b.en - psi_n.en + omega);
            sum_am += s1 * (A + B);
            sum_ma += s1 * (C + D);

            // const auto &psi_m = v_basis[im];
            // const auto &psi_a = wf.core_orbitals[ia];
            // if (psi_m == psi_v || psi_m == psi_w || psi_n == psi_v ||
            //     psi_n == psi_w) {
            //   if (A + B != 0 || C + D != 0) {
            //     std::cout << psi_b.symbol() << "|" << psi_n.symbol() << " ";
            //     std::cout << psi_m.symbol() << "|" << psi_a.symbol() << " ";
            //     std::cout << A + B << " " << C + D << "\n";
            //     std::cin.get();
            //   }
            // }

            ++in;
          }
          ++ib;
        }
        // std::cin.get();
        auto prev = t_am[ia][im];
        t_am[ia][im] = t0_am[ia][im] + f * sum_am;
        t_ma[im][ia] = t0_ma[im][ia] + f * sum_ma;
        // {
        //   auto &psi_a = wf.core_orbitals[ia];
        //   auto &psi_m = v_basis[im];
        //   auto s2t = ((psi_a.twoj() - psi_m.twoj()) % 4 == 0) ? 1 : -1;
        //   t_ma[im][ia] = t_am[ia][im] * s2t;
        //   // t_am[ia][im] = t_ma[im][ia] * s2t;
        // }
        auto delta = fabs(t_am[ia][im] - prev) / t_am[ia][im];
        if (delta > max)
          max = fabs(delta);
      }
    }
    std::cout << "it=" << i << ", eps=" << max << "\n";
    if (max < 1.0e-18)
      break;
  }
  std::cout << "Iterations time: " << sw.lap_reading_str() << "\n\n";

  // valence:
  sw.start();
  {
    double sum = 0;
    auto f = (1. / (2 * 1 + 1));
    std::size_t ia = 0;
    for (const auto &psi_a : wf.core_orbitals) {
      std::size_t im = 0;
      for (const auto &psi_m : v_basis) {
        if (psi_m == psi_a)
          continue;
        auto s1 = ((abs(psi_a.twoj() - psi_m.twoj()) + 2) % 4 == 0) ? 1 : -1;

        // auto Zwmva = cint.calculate_Z_abcdk(psi_w, psi_m, psi_v, psi_a, 1);
        // auto Zwavm = cint.calculate_Z_abcdk(psi_w, psi_a, psi_v, psi_m, 1);
        auto Zwmva =
            Coulomb::calc_Zabcdk_scratch(psi_w, psi_m, psi_v, psi_a, 1);
        auto Zwavm =
            Coulomb::calc_Zabcdk_scratch(psi_w, psi_a, psi_v, psi_m, 1);

        // auto Zwmva2 =
        //     Coulomb::calc_Zabcdk_scratch(psi_w, psi_m, psi_v, psi_a,
        //     1);
        // auto Zwavm2 =
        //     Coulomb::calc_Zabcdk_scratch(psi_w, psi_a, psi_v, psi_m,
        //     1);
        // if (Zwavm - Zwavm2 != 0) {
        //   std::cout << Zwavm - Zwavm2 << "\n";
        // }

        auto tt_am = t_am[ia][im];
        auto tt_ma = t_ma[im][ia];

        auto A = tt_am * Zwmva / (psi_a.en - psi_m.en - omega);
        auto B = Zwavm * tt_ma / (psi_a.en - psi_m.en + omega);
        if (t_am[ia][im] != 0 && t_ma[im][ia] != 0) {
          std::cout << psi_m.symbol() << "|" << psi_a.symbol() << ": ";
          auto s2t = ((psi_a.twoj() - psi_m.twoj()) % 4 == 0) ? 1 : -1;
          printf("%11.4e, %11.4e, %11.4e, d=%8.1e\n", t0_ma[im][ia],
                 t_am[ia][im] * s2t, t_ma[im][ia],
                 (t_am[ia][im] * s2t - t_ma[im][ia]) / t_am[ia][im]);
        }
        sum += s1 * (A + B);
        // {
        //   std::cout << psi_a.symbol() << "|" << psi_m.symbol() << " ";
        //   printf("+%10.3e =%10.3e|| %9.2e+%9.2e\n", f * s1 * (A + B), sum,
        //          f * s1 * A, f * s1 * B);
        // }
        ++im;
      }
      std::cout << "\n";
      ++ia;
    }
    std::cout << "\n";
    tvw_rpa = tvw_0 + f * sum;
    std::cout << tvw_0 << " " << tvw_rpa << " " << f * sum << "\n";
  }

  std::cout << sw.lap_reading_str() << "\n";

  /*
  21/6/19
   - There seems to be an issue, <a||r||m> != s <m||r||a>
   - (Is true for most, but not all!)
   - At least in some cases, seems to be a sign error!
   - Wrong sign for s/d_3/2 integrals? But these so small anyway
  */

  /*
    //TEST BASIS Completeness
    std::cout << "\n\nTEST BASIS\n\n";
    RadialOperator r2hat(wf.rgrid, 2);
    RadialOperator r1hat(wf.rgrid, 1);

    std::vector<DiracSpinor> v_basis2 = wf.core_orbitals;
    auto basis_lst = wf.listOfStates_nk(30, 1);
    for (const auto &nk : basis_lst) {
      v_basis2.emplace_back(DiracSpinor(nk[0], nk[1], wf.rgrid));
      auto tmp_vex = std::vector<double>{};
      hf.solveValence(v_basis2.back(), tmp_vex);
    }
    wf.orthonormaliseOrbitals(v_basis2, 2);

    const auto &psi = wf.valence_orbitals[2];
    double value = psi * (r2hat * psi);
    std::cout << psi * (r2hat * psi) << "\n";
    double sum = 0;
    for (const auto &phi : v_basis2) {
      if (phi.k != psi.k)
        continue;
      auto tmp = (psi * (r1hat * phi)) * (phi * (r1hat * psi));
      sum += tmp;
      std::cout << phi.symbol(); // << " " << tmp << " " << sum << "\n";
      printf(": +%11.4e =%9.5f, eps=%9.2e\n", tmp, sum,
             fabs((sum - value) / value));
    }
  */

  return 0;
}
