#include "AtomInfo.hpp"
#include <algorithm>
#include <array>
#include <cctype> //char from string
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <utility> // pair
#include <vector>

//******************************************************************************
std::string NonRelSEConfig::symbol() {
  return std::to_string(n) + AtomInfo::l_symbol(l) + std::to_string(num);
}

bool NonRelSEConfig::ok() {
  if (l + 1 > n || l < 0 || n < 1 || num < 0 || num > 4 * l + 2)
    return false;
  return true;
}

//******************************************************************************
namespace AtomInfo {

int defaultA(int Z)
// c++14: can make constexpr ?
{
  static auto match_Z = [Z](const Element &atom) { return atom.Z == Z; };
  auto atom =
      std::find_if(periodic_table.begin(), periodic_table.end(), match_Z);
  if (atom == periodic_table.end())
    return 0;
  return atom->A;
}

std::string atomicSymbol(int Z) {
  static auto match_Z = [Z](const Element &atom) { return atom.Z == Z; };
  auto atom =
      std::find_if(periodic_table.begin(), periodic_table.end(), match_Z);
  if (atom == periodic_table.end())
    return std::to_string(Z);
  return atom->symbol;
}

// Given an atomic symbol (H, He, etc.), will return Z
// Note: Symbol must be exact, including capitalisation
int get_z(const std::string &at) {

  static auto match_At = [=](const Element &atom) { return atom.symbol == at; };
  auto atom =
      std::find_if(periodic_table.begin(), periodic_table.end(), match_At);
  if (atom != periodic_table.end())
    return atom->Z;

  int z = 0;
  try {
    z = std::stoi(at);
  } catch (...) {
  }
  if (z <= 0) {
    std::cerr << "Invalid atom/Z: " << at << "\n";
    std::abort();
  }
  return z;
}

//******************************************************************************
// Short function that returns orbital term given l
std::string l_symbol(int l) {
  if (l < (int)spectroscopic_notation.length() && l >= 0)
    return spectroscopic_notation.substr(l, 1);
  else
    return "[" + std::to_string(l) + "]";
}

int symbol_to_l(const std::string &l_str) {
  for (int i = 0; i < (int)spectroscopic_notation.length(); i++) {
    if (spectroscopic_notation.substr(i, 1) == l_str)
      return i;
  }
  int l = -1;
  try {
    // Can work if given an int as a string:
    l = std::stoi(l_str);
  } catch (...) { // don't abort here (might get nice error message later)
    std::cerr << "\nFAIL AtomInfo::69 Invalid l: " << l_str << "?\n";
  }
  return l;
}

//******************************************************************************
std::string coreConfig(const std::string &in_ng) {
  // Note: must return SAME string if no matching Nobel Gas found
  // (so that this doesn't break if I give it a full term list)
  using StringPair = std::pair<std::string, std::string>;
  auto match_ng = [&](const StringPair &ng) { return ng.first == in_ng; };
  auto ng_config =
      std::find_if(nobelGasses.begin(), nobelGasses.end(), match_ng);
  if (ng_config == nobelGasses.end())
    return in_ng;
  return ng_config->second;
}

std::string niceCoreOutput(const std::string &full_core) {
  // nb: there are 8 _actual_ nobel gasses (including H-like).
  // Only want actual nobel gasses in 'nice' output
  std::string nice_core = full_core;
  for (int i = 7; i >= 0; i--) { // loop backwards (so can break)
    auto &ng_fullterm = nobelGasses[i].second;
    if (full_core.rfind(ng_fullterm, 0) == 0) {
      nice_core = nobelGasses[i].first + full_core.substr(ng_fullterm.length());
      break;
    }
  }
  return nice_core;
}

//******************************************************************************
double diracen(double z, double n, int k, double alpha) {
  double a2 = alpha * alpha;
  double c2 = 1. / a2;
  double za2 = z * z * a2;
  double g = std::sqrt(k * k - za2);
  double w2 = z * z / std::pow(g + n - fabs((double)k), 2);
  double d = 1. + a2 * w2;
  return -w2 / (2 * d) -
         (0.5 * a2 * w2 + 1. - std::sqrt(1. + a2 * w2)) * (c2 / d);
}

//******************************************************************************
std::vector<NonRelSEConfig> core_parser(const std::string &str_core_in)
// Heler function for below.
// Move to Atom Info ?
{
  // If there's a 'Noble-Gas' term, replace it with full config
  // Otherwise, 'first-term' remains unchanges
  auto found = str_core_in.find(",");
  if (found > str_core_in.length())
    found = str_core_in.length();
  auto first_term = str_core_in.substr(0, found);
  auto rest = str_core_in.substr(found);
  auto str_core = AtomInfo::coreConfig(first_term) + rest;

  // Move comma-seperated string into an array (vector)
  std::vector<std::string> term_str_list;
  {
    std::stringstream ss(str_core);
    while (ss.good()) {
      std::string substr;
      getline(ss, substr, ',');
      term_str_list.push_back(substr);
    }
  }

  std::vector<NonRelSEConfig> core_configs;
  for (const auto &term : term_str_list) {
    if (term == "")
      continue;
    bool term_ok = true;
    // find position of 'l'
    auto l_ptr = std::find_if(term.begin(), term.end(),
                              [](const char &c) { return !std::isdigit(c); });
    auto l_position = std::size_t(l_ptr - term.begin());
    int n{0}, num{0}, l{-1};
    try {
      n = std::stoi(term.substr(0, l_position - 0));
      num = std::stoi(term.substr(l_position + 1));
      if (l_position == term.size())
        throw;
      l = AtomInfo::symbol_to_l(term.substr(l_position, 1));
    } catch (...) {
      term_ok = false;
    }
    NonRelSEConfig new_config(n, l, num);

    if (!term_ok || n <= 0 || l < 0) {
      std::cout << "Problem with core: " << str_core_in << "\n";
      std::cerr << "invalid core term: " << term << "\n";
      // continue;
      std::abort();
    }

    if (num == 0)
      continue;
    auto ia = std::find(core_configs.begin(), core_configs.end(), new_config);
    if (ia == core_configs.end()) {
      core_configs.push_back(new_config);
    } else {
      *ia += new_config;
    }
  }
  while (!core_configs.empty() && core_configs.back().num == 0)
    core_configs.pop_back();

  return core_configs;
}

//******************************************************************************
std::vector<DiracSEnken> listOfStates_nk(const std::string &in_list) {
  std::vector<DiracSEnken> state_list;

  std::string n_str_previous = "";
  std::string n_str = "";
  for (char c : in_list) {
    if (std::isdigit(c)) {
      n_str += c;
    } else {
      if (n_str == "")
        n_str = n_str_previous;
      int n_max = -1;
      try {
        n_max = std::stoi(n_str);
      } catch (...) {
      }
      auto l_str = std::string(1, c);
      auto l = AtomInfo::symbol_to_l(l_str);

      // for (int n = l + 1; n <= n_max; ++n) {
      //   if (l != 0)
      //     state_list.emplace_back(n, l);
      //   state_list.emplace_back(n, -l - 1);
      // }

      if (l != 0) {
        for (int n = l + 1; n <= n_max; ++n)
          state_list.emplace_back(n, l);
      }
      for (int n = l + 1; n <= n_max; ++n)
        state_list.emplace_back(n, -l - 1);

      n_str_previous = n_str;
      n_str = "";
    }
  }
  return state_list;
}

} // namespace AtomInfo
