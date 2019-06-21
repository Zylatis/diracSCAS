#include "CoulombIntegrals.hpp"
#include "DiracSpinor.hpp"
#include "NumCalc_quadIntegrate.hpp"
#include <vector>

//******************************************************************************
Coulomb::Coulomb(const std::vector<DiracSpinor> &in_core,
                 const std::vector<DiracSpinor> &in_valence = {})
    : c_orbs_ptr(&in_core), v_orbs_ptr(&in_valence),
      rgrid_ptr((in_core.size() > 0) ? in_core.front().p_rgrid
                                     : in_valence.front().p_rgrid) {
  initialise_core_core(); // must be only time this is called
  // Core orbitals must not change after this has been called!
  // [orbital functions themselves can change, but not the vector]
  // Valence orbitals may change (new orbitals added) - but pointer to val.
  // orbitals remains const
}

//******************************************************************************
void Coulomb::initialise_core_core()
// Initialises memory (sizes arays) used to store core-core y^k_ab C ints
// Must only be called once! (from constructor)
{
  // XXX should only be able to call this once?
  auto ngp = rgrid_ptr->ngp;
  m_y_abkr.reserve(c_orbs_ptr->size());
  for (std::size_t ia = 0; ia < c_orbs_ptr->size(); ia++) {
    auto tja = (*c_orbs_ptr)[ia].twoj();
    std::vector<std::vector<std::vector<double>>> ya_bkr;
    for (std::size_t ib = 0; ib <= ia; ib++) {
      auto tjb = (*c_orbs_ptr)[ib].twoj();
      std::size_t num_k = (tja > tjb) ? (tjb + 1) : (tja + 1);
      ya_bkr.emplace_back(
          std::vector<std::vector<double>>(num_k, std::vector<double>(ngp)));
    }
    m_y_abkr.push_back(ya_bkr);
    calculate_angular((*c_orbs_ptr)[ia].k_index());
  }
}
//------------------------------------------------------------------------------
void Coulomb::initialise_core_valence()
// Initialises memory (sizes arays) used to store core-valence y^k_vc C ints
// May be called many times.
// Must be called after new valence orbitals are constructed before can
// calculate the new integrals (for now, this happens automatically)
{
  auto ngp = rgrid_ptr->ngp;
  for (std::size_t iv = num_initialised_vc; iv < v_orbs_ptr->size(); iv++) {
    auto tjv = (*v_orbs_ptr)[iv].twoj();
    std::vector<std::vector<std::vector<double>>> va_bkr;
    for (const auto &phi_c : *c_orbs_ptr) {
      auto tjc = phi_c.twoj();
      std::size_t num_k = (tjv > tjc) ? (tjc + 1) : (tjv + 1);
      va_bkr.emplace_back(
          std::vector<std::vector<double>>(num_k, std::vector<double>(ngp)));
    }
    m_y_vckr.push_back(va_bkr);
    calculate_angular((*v_orbs_ptr)[iv].k_index());
  }
  num_initialised_vc = v_orbs_ptr->size();
}
//------------------------------------------------------------------------------
void Coulomb::initialise_valence_valence()
// Initialises memory (sizes arays) used to store valence-valence y^k_vw C ints
// May be called many times.
// Must be called after new valence orbitals are constructed before can
// calculate the new integrals (for now, this happens automatically)
{
  auto ngp = rgrid_ptr->ngp;
  for (std::size_t iv = num_initialised_vv; iv < v_orbs_ptr->size(); iv++) {
    auto tjv = (*v_orbs_ptr)[iv].twoj();
    std::vector<std::vector<std::vector<double>>> vv_wkr;
    for (std::size_t iw = 0; iw <= iv; iw++) {
      auto tjw = (*v_orbs_ptr)[iw].twoj();
      std::size_t num_k = (tjv > tjw) ? (tjw + 1) : (tjv + 1);
      vv_wkr.emplace_back(
          std::vector<std::vector<double>>(num_k, std::vector<double>(ngp)));
    }
    m_y_vwkr.push_back(vv_wkr);
    calculate_angular((*v_orbs_ptr)[iv].k_index());
  }
  num_initialised_vv = v_orbs_ptr->size();
}

//******************************************************************************
std::size_t Coulomb::find_core_index(const DiracSpinor &phi) const {
  // This finds the array index of a particular orbital
  // Note: if orbital is not in the list (e.g., if called with a valence orbital
  // instead of a core orbital), will return [size_of_orbitals], which is an
  // invalid index (can cause undefined behaviour!) This is very slow..is there
  // another way?
  auto ia = std::find(c_orbs_ptr->begin(), c_orbs_ptr->end(), phi);
  return (std::size_t)std::distance(c_orbs_ptr->begin(), ia);
}
//------------------------------------------------------------------------------
std::size_t Coulomb::find_valence_index(const DiracSpinor &phi) const {
  auto ia = std::find(v_orbs_ptr->begin(), v_orbs_ptr->end(), phi);
  return (std::size_t)std::distance(v_orbs_ptr->begin(), ia);
}

//******************************************************************************
void Coulomb::form_core_core()
// Calls calculate_y_ijk, fills the core-core C int arrays
// Note: symmety: y_ij = y_ji, therefore only calculates y_ij with i >= j
{
  auto Ncore = c_orbs_ptr->size();
#pragma omp parallel for
  for (std::size_t ia = 0; ia < Ncore; ia++) {
    const auto &phi_a = (*c_orbs_ptr)[ia];
    auto tja = phi_a.twoj();
    auto kia = phi_a.k_index();
    for (std::size_t ib = 0; ib <= ia; ib++) {
      const auto &phi_b = (*c_orbs_ptr)[ib];
      auto tjb = phi_b.twoj();
      auto kib = phi_b.k_index();
      auto kmin = abs(tja - tjb) / 2;
      auto num_k = (tja > tjb) ? (tjb + 1) : (tja + 1);
      const auto &Lk = get_angular_L_kiakib_k(kia, kib);
      for (int ik = 0; ik < num_k; ik++) {
        if (Lk[ik] == 0)
          continue;
        calculate_y_ijk(phi_a, phi_b, kmin + ik, m_y_abkr[ia][ib][ik]);
      }
    }
  }
}
//******************************************************************************
void Coulomb::form_valence_valence()
// Calls calculate_y_ijk, fills the valence-valence C int arrays
// Note: symmety: y_ij = y_ji, therefore only calculates y_ij with i >= j
{
  initialise_valence_valence(); // call this each time?
  auto Nval = v_orbs_ptr->size();
#pragma omp parallel for
  for (std::size_t iv = 0; iv < Nval; iv++) {
    const auto &phi_v = (*v_orbs_ptr)[iv];
    auto tjv = phi_v.twoj();
    auto kiv = phi_v.k_index();
    for (std::size_t iw = 0; iw <= iv; iw++) {
      const auto &phi_w = (*v_orbs_ptr)[iw];
      auto tjw = phi_w.twoj();
      auto kiw = phi_w.k_index();
      auto kmin = abs(tjv - tjw) / 2;
      auto num_k = (tjv > tjw) ? (tjw + 1) : (tjv + 1);
      const auto &Lk = get_angular_L_kiakib_k(kiv, kiw);
      for (int ik = 0; ik < num_k; ik++) {
        if (Lk[ik] == 0)
          continue;
        calculate_y_ijk(phi_v, phi_w, kmin + ik, m_y_vwkr[iv][iw][ik]);
      }
    }
  }
}
//******************************************************************************
void Coulomb::form_core_valence()
// Calls calculate_y_ijk, fills the core-valence C int arrays
// Note: no symmetry here! y_ij != y_ji [j and i same index, NOT same orbital!]
{
  initialise_core_valence(); // call this each time?
  auto Nval = v_orbs_ptr->size();
#pragma omp parallel for // two-level?
  for (std::size_t iv = 0; iv < Nval; iv++) {
    const auto &phi_v = (*v_orbs_ptr)[iv];
    auto tjv = phi_v.twoj();
    auto kiv = phi_v.k_index();
    for (std::size_t ic = 0; ic < c_orbs_ptr->size(); ic++) {
      const auto &phi_c = (*c_orbs_ptr)[ic];
      auto tjc = phi_c.twoj();
      auto kic = phi_c.k_index();
      auto kmin = abs(tjc - tjv) / 2;
      auto num_k = (tjc > tjv) ? (tjv + 1) : (tjc + 1);
      const auto &Lk = get_angular_L_kiakib_k(kic, kiv);
      for (int ik = 0; ik < num_k; ik++) {
        if (Lk[ik] == 0)
          continue;
        calculate_y_ijk(phi_c, phi_v, kmin + ik, m_y_vckr[iv][ic][ik]);
      }
    }
  }
}

//******************************************************************************
const std::vector<std::vector<double>> &
Coulomb::get_y_abk(std::size_t a, std::size_t b) const {
  return (a > b) ? m_y_abkr[a][b] : m_y_abkr[a][b];
}
//------------------------------------------------------------------------------
const std::vector<std::vector<double>> &
Coulomb::get_y_vwk(std::size_t v, std::size_t w) const {
  return (v > w) ? m_y_abkr[v][w] : m_y_abkr[w][v];
}
//------------------------------------------------------------------------------
const std::vector<std::vector<double>> &
Coulomb::get_y_vck(std::size_t v, std::size_t c) const {
  return m_y_vckr[v][c];
}
//******************************************************************************
const std::vector<std::vector<double>> &
Coulomb::get_y_ijk(const DiracSpinor &phi_i, const DiracSpinor &phi_j) const {

  auto i = find_core_index(phi_i);
  bool ival = false;
  if (i == c_orbs_ptr->size()) {
    i = find_valence_index(phi_i);
    ival = true;
  }
  auto j = find_core_index(phi_j);
  bool jval = false;
  if (j == c_orbs_ptr->size()) {
    j = find_valence_index(phi_j);
    jval = true;
  }
  if (!ival && !jval)
    return (i > j) ? m_y_abkr[i][j] : m_y_abkr[i][j];
  if (ival && !jval)
    return m_y_vckr[i][j];
  if (!ival && jval)
    return m_y_vckr[j][i];
  return (i > j) ? m_y_vckr[i][j] : m_y_vckr[i][j];
}
//------------------------------------------------------------------------------
const std::vector<double> &Coulomb::get_y_ijk(const DiracSpinor &phi_i,
                                              const DiracSpinor &phi_j,
                                              int k) const {
  auto tji = phi_i.twoj();
  auto tjj = phi_j.twoj();
  auto kmin = abs(tji - tjj) / 2; // kmin
  auto kmax = (tji + tjj) / 2;    // kmax
  if (k > kmax || k < kmin) {
    std::cerr << "FAIL 214 in CI; bad k\n";
    std::abort();
  }
  const auto &tmp = get_y_ijk(phi_i, phi_j);
  return tmp[k - kmin];
}

//******************************************************************************
void Coulomb::calculate_angular(int ki)
// Calculated the angular coeficients C and L.
// Automatically allocates memory (sizes arrays); can be called any number of
// times. [Called automatically]
// Stores them in arrays, indexed directly by kappa_index (ki) for 'k',
// only store between kmin and kmax, so off-set by kmin For kappa_index (kia,
// kib), and k: C[kia][kib][k-kmin] Due to symmetry, C_ab = C_ba only store for
// kia >= kib Always use getter functions to access array. Definitions: L =
// Lambda^k_ij := 3js((ji,jj,k),(-1/2,1/2,0))^2 * parity(li+lj+k) C = |
// <k||C^k||k'> |
//   = Sqrt([ji][jj]) * 3js((ji,jj,k),(-1/2,1/2,0)) * parity(li+lj+k)
// Note: C is abs value) - if sign needed, do seperately [sign NOT symmetric!]
// Also:
// k_min = |j - j'|; k_max = |j + j'|
// num_k = (j' + 1) if j>j', (j + 1) if j'>j;
{
  if (ki <= m_largest_ki)
    return;
  auto prev_largest_ki = m_largest_ki;
  m_largest_ki = ki;
  for (auto kia = prev_largest_ki + 1; kia <= m_largest_ki; kia++) {
    auto tja = AtomInfo::twojFromIndex(kia);
    auto la = AtomInfo::lFromIndex(kia);
    std::vector<std::vector<double>> C_ka_kbk;
    std::vector<std::vector<double>> L_ka_kbk;
    for (auto kib = 0; kib <= kia; kib++) {
      auto tjb = AtomInfo::twojFromIndex(kib);
      auto lb = AtomInfo::lFromIndex(kib);
      auto kmin = (tja - tjb) / 2; // don't need abs, as b\leq a => ja\geq jb
      auto kmax = (tja + tjb) / 2;
      std::vector<double> C_k(kmax - kmin + 1, 0);
      std::vector<double> L_k(kmax - kmin + 1, 0);
      for (auto k = kmin; k <= kmax; k++) {
        if (Wigner::parity(la, lb, k) == 0)
          continue;
        int ik = k - kmin;
        auto tjs = Wigner::threej_2(tja, tjb, 2 * k, -1, 1, 0);
        C_k[ik] = sqrt((tja + 1) * (tjb + 1)) * tjs; // nb: no sign!
        L_k[ik] = tjs * tjs;
      } // k
      C_ka_kbk.push_back(C_k);
      L_ka_kbk.push_back(L_k);
    }
    m_C_kakbk.push_back(C_ka_kbk);
    m_L_kakbk.push_back(L_ka_kbk);
  }
}

//******************************************************************************
double Coulomb::get_angular_C_kiakibk(const DiracSpinor &phi_a,
                                      const DiracSpinor &phi_b, int k) const {
  auto kia = phi_a.k_index();
  auto kib = phi_b.k_index();
  int kmin =
      abs(AtomInfo::twojFromIndex(kia) - AtomInfo::twojFromIndex(kib)) / 2;
  int kmax =
      abs(AtomInfo::twojFromIndex(kia) + AtomInfo::twojFromIndex(kib)) / 2;
  if (k < kmin || k > kmax)
    return 0;
  return kia > kib ? m_C_kakbk[kia][kib][k - kmin]
                   : m_C_kakbk[kib][kia][k - kmin];
}
//******************************************************************************
const std::vector<double> &Coulomb::get_angular_C_kiakib_k(int kia,
                                                           int kib) const {
  // note:output is of-set by k_min!
  return kia > kib ? m_C_kakbk[kia][kib] : m_C_kakbk[kib][kia];
}

//******************************************************************************
double Coulomb::get_angular_L_kiakibk(const DiracSpinor &phi_a,
                                      const DiracSpinor &phi_b, int k) const {
  auto kia = phi_a.k_index();
  auto kib = phi_b.k_index();
  int kmin =
      abs(AtomInfo::twojFromIndex(kia) - AtomInfo::twojFromIndex(kib)) / 2;
  int kmax =
      abs(AtomInfo::twojFromIndex(kia) + AtomInfo::twojFromIndex(kib)) / 2;
  if (k < kmin || k > kmax)
    return 0;
  return kia > kib ? m_L_kakbk[kia][kib][k - kmin]
                   : m_L_kakbk[kib][kia][k - kmin];
}
//******************************************************************************
const std::vector<double> &Coulomb::get_angular_L_kiakib_k(int kia,
                                                           int kib) const {
  // note:output is off-set by k_min!
  return kia > kib ? m_L_kakbk[kia][kib] : m_L_kakbk[kib][kia];
}

//******************************************************************************
std::vector<double> Coulomb::calculate_R_abcd_k(const DiracSpinor &psi_a,
                                                const DiracSpinor &psi_b,
                                                const DiracSpinor &psi_c,
                                                const DiracSpinor &psi_d) const
// R^k_abcd = Int_0^inf [fa*fc + ga*gc]*y^k_bd(r) dr
// Symmetry: a<->c, and b<->d
// NOTE: NOT offset by k_min, so will calculate for k=0,1,2,...,k_max
{
  auto kmin = abs(psi_b.twoj() - psi_d.twoj()) / 2;
  auto kmax = abs(psi_b.twoj() + psi_d.twoj()) / 2;
  const auto &drdu = psi_a.p_rgrid->drdu; // save typing
  const auto du = psi_a.p_rgrid->du;

  auto pinf = std::min(psi_a.pinf, psi_c.pinf);

  // For now, this returns. Later, might be faster to swap to in/out param!
  // (To avoid huge amount of re-alocating memory)
  // Actually, typically only need to call this once (for each a,b,c,d)
  // So, will be equally as fast with nRVO
  std::vector<double> Rabcd(kmax + 1, 0);

  const auto &ybd_kr = get_y_ijk(psi_b, psi_d);
  for (int k = kmin; k <= kmax; k++) {
    const auto &ybdk_r = ybd_kr[k - kmin];
    auto ffy = NumCalc::integrate(psi_a.f, psi_c.f, ybdk_r, drdu, 1, 0, pinf);
    auto ggy = NumCalc::integrate(psi_a.g, psi_c.g, ybdk_r, drdu, 1, 0, pinf);
    Rabcd[k] = (ffy + ggy) * du;
  }
  return Rabcd;
}

//******************************************************************************
std::vector<double>
Coulomb::calculate_X_abcd_k(const DiracSpinor &psi_a, const DiracSpinor &psi_b,
                            const DiracSpinor &psi_c,
                            const DiracSpinor &psi_d) const {
  // tmp_X is NOT offset by kmin
  auto tmp_X = calculate_R_abcd_k(psi_a, psi_b, psi_c, psi_d);
  auto kmax = abs(psi_b.twoj() + psi_d.twoj()) / 2;
  // C IS offset by kmin
  auto kmin = abs(psi_b.twoj() - psi_d.twoj()) / 2;
  auto C_ac = get_angular_C_kiakib_k(psi_a.k_index(), psi_c.k_index());
  auto C_bd = get_angular_C_kiakib_k(psi_b.k_index(), psi_d.k_index());

  auto japjbp1 = (psi_a.twoj() + psi_b.twoj()) / 2 + 1;
  for (int k = kmin; k <= kmax; k++) {
    // sign: (-1)^{ja+0.5} * (-1)^{jb+0.5} * (-1)^k
    //       = (-1)^{ja + jb + k + 1}
    auto sign = ((japjbp1 + k) % 2 == 0) ? 1 : -1;
    tmp_X[k] *= (sign * C_ac[k - kmin] * C_bd[k - kmin]);
    ++k;
  }
  return tmp_X;
}
//******************************************************************************
double Coulomb::calculate_Z_abcdk(const DiracSpinor &psi_a,
                                  const DiracSpinor &psi_b,
                                  const DiracSpinor &psi_c,
                                  const DiracSpinor &psi_d, int k) const {
  auto tja = psi_a.twoj();
  auto tjb = psi_b.twoj();
  auto tjc = psi_c.twoj();
  auto tjd = psi_d.twoj();
  // auto kkp1 = k * (k + 1); // DOESN"T WORK FOR k=0 !!!
  auto kkp1 = 2 * k + 1; // DOESN"T WORK FOR k=0 !!!

  auto Xabcd = calculate_X_abcd_k(psi_a, psi_b, psi_c, psi_d);
  auto kmax = abs(tjb + tjd) / 2;
  if (k > kmax)
    return 0;

  double sum = 0.0;
  int ll = 0; // XXX X is NOT offset (prob should be..?)
  for (const auto &x : Xabcd) {
    auto sj = Wigner::sixj_2(tjc, tja, 2 * k, tjd, tjb, 2 * ll);
    sum += sj * x;
    ++ll;
  }

  return Xabcd[k] + kkp1 * sum;
}
//******************************************************************************
void Coulomb::calculate_y_ijk(const DiracSpinor &phi_a,
                              const DiracSpinor &phi_b, const int k,
                              std::vector<double> &vabk)
// This is static
// Calculalates y^k_ab screening function.
// Note: should only call for a>=b, and for k's with non-zero angular coefs
// (nothing bad will happen otherwise, but no point!)
// Since y_ab = y_ba
//
// Stores in vabk (in/out parameter, reference to whatever)
//
// r_min := min(r,r')
// rho(r') := fa(r')*fb(r') + ga(r')gb(r')
// y^k_ab(r) = Int_0^inf [r_min^k/r_max^(k+1)]*rho(f') dr'
//           = Int_0^r [r'^k/r^(k+1)]*rho(r') dr'
//             + Int_r^inf [r^k/r'^(k+1)]*rho(r') dr'
//          := A(r)/r^(k+1) + B(r)*r^k
// A(r0)  = 0
// B(r0)  = Int_0^inf [r^k/r'^(k+1)]*rho(r') dr'
// A(r_n) = A(r_{n-1}) + (rho(r_{n-1})*r_{n-1}^k)*dr
// B(r_n) = A(r_{n-1}) + (rho(r_{n-1})/r_{n-1}^(k+1))*dr
// y^k_ab(rn) = A(rn)/rn^(k+1) + B(rn)*rn^k
{
  auto &grid = phi_a.p_rgrid; // just save typing
  auto du = grid->du;
  auto ngp = grid->ngp;
  vabk.resize(ngp); // for safety

  auto irmax = std::min(phi_a.pinf, phi_b.pinf);

  double Ax = 0, Bx = 0; // A, B defined in equations/comments above
  for (std::size_t i = 0; i < irmax; i++) {
    Bx += grid->drdu[i] * (phi_a.f[i] * phi_b.f[i] + phi_a.g[i] * phi_b.g[i]) /
          pow(grid->r[i], k + 1);
  }

  // For "direct" part, can't cut!
  if (phi_a == phi_b)
    irmax = ngp;

  vabk[0] = Bx * du;
  for (std::size_t i = 1; i < irmax; i++) {
    auto Fdr = grid->drdu[i - 1] * (phi_a.f[i - 1] * phi_b.f[i - 1] +
                                    phi_a.g[i - 1] * phi_b.g[i - 1]);
    Ax = Ax + Fdr * pow(grid->r[i - 1], k);
    Bx = Bx - Fdr / pow(grid->r[i - 1], k + 1);
    vabk[i] = du * (Ax / pow(grid->r[i], k + 1) + Bx * pow(grid->r[i], k));
  }
  for (std::size_t i = irmax; i < ngp; i++) {
    vabk[i] = 0; // this doesn't happen in psi_a = psi_b
  }
}
