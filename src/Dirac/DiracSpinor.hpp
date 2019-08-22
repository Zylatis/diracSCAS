#pragma once
#include "Maths/Grid.hpp"
#include "Maths/NumCalc_quadIntegrate.hpp"
#include "Physics/AtomInfo.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

//******************************************************************************
class DiracSpinor {

public: // Data
  DiracSpinor(int in_n, int in_k, const Grid &rgrid, bool imaginary_g = true)
      : p_rgrid(&rgrid),                      //
        n(in_n), k(in_k), en(0.0),            //
        f(std::vector<double>(rgrid.ngp, 0)), //
        g(f),                                 //
        pinf(rgrid.ngp - 1),                  //
        imaginary_g(imaginary_g),             //
        its(-1), eps(-1), occ_frac(0),        //
        m_twoj(AtomInfo::twoj_k(in_k)),       //
        m_l(AtomInfo::l_k(in_k)),             //
        m_parity(AtomInfo::parity_k(in_k)),   //
        m_k_index(AtomInfo::indexFromKappa(in_k)) {}
  const Grid *const p_rgrid;

  const int n;
  const int k;

  double en = 0;
  std::vector<double> f;
  std::vector<double> g;
  std::size_t pinf;

  // determines relative sign in radial integral:
  // true by default. If false, means upper comp is i
  const bool imaginary_g;

  int its;
  double eps;
  double occ_frac;

private:
  const int m_twoj;
  const int m_l;
  const int m_parity;
  const int m_k_index;

public: // Methods
  int l() const { return m_l; }
  double j() const { return 0.5 * double(m_twoj); }
  double jjp1() const { return j() * (j() + 1); }
  int twoj() const { return m_twoj; }
  int twojp1() const { return m_twoj + 1; }
  int parity() const { return m_parity; }
  int k_index() const { return m_k_index; }

  std::string symbol(bool gnuplot = false) const {
    // Readable symbol (s_1/2, p_{3/2} etc.).
    // gnuplot-firndly '{}' braces optional.
    std::string ostring1 = std::to_string(n) + AtomInfo::l_symbol(m_l);
    std::string ostring2 = gnuplot ? "_{" + std::to_string(m_twoj) + "/2}"
                                   : "_" + std::to_string(m_twoj) + "/2";
    return ostring1 + ostring2;
  }
  std::string shortSymbol() const {
    std::string pm = (k < 0) ? "+" : "-";
    return std::to_string(n) + AtomInfo::l_symbol(m_l) + pm;
  }

  double norm() const { return std::sqrt((*this) * (*this)); }

  void scale(const double factor) {
    for (auto &f_r : f)
      f_r *= factor;
    for (auto &g_r : g)
      g_r *= factor;
    // for (std::size_t i = 0; i < pinf; ++i)
    //   f[i] *= factor;
    // for (std::size_t i = 0; i < pinf; ++i)
    //   g[i] *= factor;
    // for (std::size_t i = pinf; i < p_rgrid->ngp; ++i) {
    //   f[i] = 0.0;
    //   g[i] = 0.0;
    // }
  }

  void normalise(double norm_to = 1.0) {
    double rescale_factor = norm_to / norm();
    scale(rescale_factor);
  }

  auto r0pinfratio() const {
    auto max_abs_compare = [](double a, double b) {
      return std::fabs(a) < std::fabs(b);
    };
    auto max_pos =
        std::max_element(f.begin(), f.begin() + pinf, max_abs_compare);
    auto r0_ratio = f[0] / *max_pos;
    auto pinf_ratio = f[pinf - 1] / *max_pos;
    return std::make_pair(r0_ratio, pinf_ratio);
    // nb: do i care about ratio to max? or just value?
  }

public: // Operator overloads
  double operator*(const DiracSpinor &rhs) const {
    // XXX This is slow??? And one of the most critial parts!
    // Change the relative sign based in Complex f or g component
    // (includes complex conjugation of lhs)
    int ffs = ((!imaginary_g) && rhs.imaginary_g) ? -1 : 1;
    int ggs = (imaginary_g && !rhs.imaginary_g) ? -1 : 1;
    auto imax = p_rgrid->ngp; // std::min(pinf, rhs.pinf); //XXX
    // XX Actually makes a difference here!
    // auto ff = NumCalc::integrate({&f, &rhs.f, &p_rgrid->drdu}, 1.0, 0, imax);
    // auto gg = NumCalc::integrate({&g, &rhs.g, &p_rgrid->drdu}, 1.0, 0, imax);
    auto ff = NumCalc::integrate(f, rhs.f, p_rgrid->drdu, 1.0, 0, imax);
    auto gg = NumCalc::integrate(g, rhs.g, p_rgrid->drdu, 1.0, 0, imax);
    return (ffs * ff + ggs * gg) * p_rgrid->du;
  }

  DiracSpinor &operator+=(const DiracSpinor &rhs) {
    auto imax = p_rgrid->ngp; // std::min(pinf, rhs.pinf); //XXX
    // auto imax = std::max(pinf, rhs.pinf); // XXX
    // pinf = imax;
    for (std::size_t i = 0; i < imax; i++)
      f[i] += rhs.f[i];
    for (std::size_t i = 0; i < imax; i++)
      g[i] += rhs.g[i];
    return *this;
  }
  friend DiracSpinor operator+(DiracSpinor lhs, const DiracSpinor &rhs) {
    lhs += rhs;
    return lhs;
  }
  DiracSpinor &operator-=(const DiracSpinor &rhs) {
    auto imax = p_rgrid->ngp; // std::min(pinf, rhs.pinf); //XXX
    // auto imax = std::max(pinf, rhs.pinf); // XXX WHY this make slow??
    // auto imax = (pinf > rhs.pinf) ? pinf : rhs.pinf;
    // pinf = imax;
    for (std::size_t i = 0; i < imax; i++)
      f[i] -= rhs.f[i];
    for (std::size_t i = 0; i < imax; i++)
      g[i] -= rhs.g[i];
    return *this;
  }
  friend DiracSpinor operator-(DiracSpinor lhs, const DiracSpinor &rhs) {
    lhs -= rhs;
    return lhs;
  }

  DiracSpinor &operator*=(const double x) {
    scale(x);
    return *this;
  }
  friend DiracSpinor operator*(DiracSpinor lhs, const double x) {
    lhs *= x;
    return lhs;
  }
  friend DiracSpinor operator*(const double x, DiracSpinor rhs) {
    rhs *= x;
    return rhs;
  }
  friend DiracSpinor operator*(const std::vector<double> &v, DiracSpinor rhs) {
    // friend?
    auto size = rhs.p_rgrid->ngp;
    for (auto i = 0ul; i < size; i++) {
      rhs.f[i] *= v[i];
      rhs.g[i] *= v[i];
    }
    return rhs;
  }
  //
  DiracSpinor &operator=(const DiracSpinor &other) {
    if (this != &other) {
      en = other.en;
      f = other.f;
      g = other.g;
      pinf = other.pinf;
    }
    return *this;
  }
};

//******************************************************************************
// comparitor overloads:

inline bool operator==(const DiracSpinor &lhs, const DiracSpinor &rhs) {
  return lhs.n == rhs.n && lhs.k == rhs.k;
}
inline bool operator!=(const DiracSpinor &lhs, const DiracSpinor &rhs) {
  return !(lhs == rhs);
}
inline bool operator<(const DiracSpinor &lhs, const DiracSpinor &rhs) {
  if (lhs.n == rhs.n)
    return AtomInfo::indexFromKappa(lhs.k) < AtomInfo::indexFromKappa(rhs.k);
  return lhs.n < rhs.n;
}
inline bool operator>(const DiracSpinor &lhs, const DiracSpinor &rhs) {
  return rhs < lhs;
}
inline bool operator<=(const DiracSpinor &lhs, const DiracSpinor &rhs) {
  return !(lhs > rhs);
}
inline bool operator>=(const DiracSpinor &lhs, const DiracSpinor &rhs) {
  return !(lhs < rhs);
}