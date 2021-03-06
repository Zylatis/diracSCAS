#pragma once
#include <vector>
class DiracSpinor;
class DiracOperator;

// Allow it to do RPA daigram-style as well!
//(needs a basis)

enum class dPsiType { X, Y };

class ExternalField {
public:
  ExternalField(const DiracOperator *const h,
                const std::vector<DiracSpinor> &core,
                const std::vector<double> &vl, const double alpha,
                const double omega = 0);

private:
  // dPhi = X exp(-iwt) + Y exp(+iwt)
  // (H - e - w)X = -(h + dV - de)Phi
  // (H - e + w)Y = -(h* + dV* - de)Phi
  // X_c = sum_x X_x,
  // j(x)=j(c)-k,...,j(c)+k.  And: pi(x) = pi(c)*pi(h)
  std::vector<std::vector<DiracSpinor>> m_X; // X[core_state][kappa_x]
  std::vector<std::vector<DiracSpinor>> m_Y;

  const DiracOperator *const m_h; //??
  const std::vector<DiracSpinor> *const p_core;
  const std::vector<double> m_vl;
  const double m_alpha;
  const double m_omega;
  const bool static_fieldQ;
  const int m_rank;
  const int m_pi;
  const bool m_imag;

public:
  const std::vector<DiracSpinor> &get_dPsis(const DiracSpinor &phic,
                                            dPsiType XorY);
  const DiracSpinor &get_dPsi_x(const DiracSpinor &phic, dPsiType XorY,
                                const int kappa_x);

  void solve_TDHFcore();

  // does it matter if a or b is in the core?
  double dV_ab(const DiracSpinor &phia, const DiracSpinor &phib);
  DiracSpinor dV_ab_rhs(const DiracSpinor &phia, const DiracSpinor &phib);

private:
  std::size_t core_index(const DiracSpinor &phic);
};
