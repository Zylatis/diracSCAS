#include "Wigner_369j.hpp"
#include <vector>

constexpr int twoj(int ji) { return 2 * ji + 1; }
constexpr int jindex(int twoj) { return (twoj - 1) / 2; }

inline int max4(int a, int b, int c, int d) {
  return std::max(std::max(a, b), std::max(c, d));
}

inline int min_lambda_ji(int a, int b, int c, int d) {
  // specialised version! a must be max, rest must be in order!
  auto tmp = std::max((a - b), abs(c - d));
  return twoj(tmp); // XXX does this work??
  // lambdamin(twoj(index)) = twoj(lambdamin(index)) ???
}

class SixJTable { // XXX Make this a "const k" version. Later: full version!
public:
  SixJTable(int in_k, int tj_max = -1)
      : k(in_k), max_ji_sofar(-1) //, m_k_a_bcdl({})
  {
    fill(tj_max);
  }

private:
  const int k;
  int max_ji_sofar;
  std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>
      m_k_a_bcdl;
  // mk[a][b][c][d][l];

public:
  double value(int tja, int tjb, int tjc, int tjd, int l) {
    auto a = jindex(tja);
    auto b = jindex(tjb);
    auto c = jindex(tjc);
    auto d = jindex(tjd);
    auto max = max4(a, b, c, d); //

    auto sj = //[&m_k_a_bcdl, = l]
        [&](int a, int b, int c, int d) {
          const auto lmin = min_lambda_ji(a, b, c, d);
          auto &sj_tmp = m_k_a_bcdl[a][b][c][d];
          const auto lmax = lmin + (int)sj_tmp.size() - 1;
          if (l < lmin || l > lmax)
            return 0.0;
          return sj_tmp[l - lmin];
        };

    if (max == a) {
      std::cout << "a!\n";
      return sj(a, b, c, d);
    } else if (max == b) {
      std::cout << "b!\n";
      return sj(b, a, d, c);
    } else if (max == c) {
      return sj(c, d, a, b);
    } else {
      return sj(d, c, b, a);
    }
  }

  double value(int tja, int tjb, int in_k, int tjc, int tjd, int l) {
    if (in_k == k)
      return value(tja, tjb, tjc, tjd, l);
    return 0;
  }

public:
  void fill(const int tj_max) {

    const int ji_max = jindex(tj_max);
    if (ji_max <= max_ji_sofar)
      return;

    m_k_a_bcdl.reserve(ji_max + 1);
    for (int a = max_ji_sofar + 1; a <= ji_max; a++) {
      auto tja = twoj(a);
      std::vector<std::vector<std::vector<std::vector<double>>>> ka_bcdl;
      ka_bcdl.reserve(a + 1);
      for (int b = 0; b <= a; b++) {
        auto tjb = twoj(b);
        const auto amb = tja - tjb; // no abs, always +ve
        const auto apb = tja + tjb;
        std::vector<std::vector<std::vector<double>>> ka_b_cdl;
        ka_b_cdl.reserve(a + 1);
        for (int c = 0; c <= a; c++) {
          auto tjc = twoj(c);
          std::vector<std::vector<double>> ka_bc_dl;
          ka_bc_dl.reserve(a + 1);
          for (int d = 0; d <= a; d++) {
            auto tjd = twoj(d);
            const auto cmd = abs(tjc - tjd);
            const auto cpd = tjc + tjd;
            auto lambda_min = std::max(amb, cmd) / 2;
            auto lambda_max = std::min(apb, cpd) / 2;
            std::vector<double> ka_bcd_l;
            ka_bcd_l.reserve(abs(lambda_max - lambda_min + 1));
            for (auto l = lambda_min; l <= lambda_max; l++) {
              ka_bcd_l.push_back(
                  Wigner::sixj_2(tja, tjb, 2 * k, tjc, tjd, 2 * l));
            }
            ka_bc_dl.push_back(ka_bcd_l);
          }
          ka_b_cdl.push_back(ka_bc_dl);
        }
        ka_bcdl.push_back(ka_b_cdl);
      }
      m_k_a_bcdl.push_back(ka_bcdl);
    }

    if (ji_max > max_ji_sofar)
      max_ji_sofar = ji_max;
  }

  //
};
