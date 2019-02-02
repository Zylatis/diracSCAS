#pragma once
#include <iostream>
#include <string>
#include <vector>

/*
 * Add nuclear properties!
 */

namespace ATI {

// Default values for A for each atom.
// Note: array index matches Z, so first entry is blank.
// Goes up to E120 (Z=120)
const int A[121] = {
    0,   1,   4,   7,   9,   11,  12,  14,  16,  19,  20,  23,  24,  27,
    28,  31,  32,  35,  40,  39,  40,  45,  48,  51,  52,  55,  56,  59,
    59,  64,  65,  70,  73,  75,  79,  80,  84,  85,  88,  89,  91,  93,
    96,  97,  101, 103, 106, 108, 112, 115, 119, 122, 128, 127, 131, 133,
    137, 139, 140, 141, 144, 145, 150, 152, 157, 159, 162, 165, 167, 169,
    173, 175, 178, 181, 184, 186, 190, 192, 195, 197, 201, 204, 207, 209,
    209, 210, 222, 223, 226, 227, 232, 231, 238, 237, 244, 243, 247, 247,
    251, 252, 257, 258, 259, 262, 267, 270, 269, 270, 270, 278, 281, 281,
    285, 286, 289, 289, 293, 293, 294, 315, 320};

const std::string atom_name_z[121] = {
    "0",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",    "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",   "Sc",
    "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",   "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo",   "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",    "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",   "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",   "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",   "Fr",
    "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk",   "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",   "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "E119", "E120"};

// Short function that returns orbital term given l
inline std::string l_symbol(int l) {
  std::string tmp_l_array[6] = {"s", "p", "d", "f", "g", "h"};
  if (l < 6)
    return tmp_l_array[l];
  else
    return std::to_string(l);
}

// Short function that returns orbital l (int), given kappa
inline int l_k(int ka) { return (abs(2 * ka + 1) - 1) / 2; }
// Short function that returns 2*j (int), given kappa
inline int twoj_k(int ka) { return 2 * abs(ka) - 1; }
inline double j_k(int ka) { return 0.5 * twoj_k(ka); }

inline int indexFromKappa(int ka) {
  if (ka < 0)
    return -2 * ka - 2;
  else
    return 2 * ka - 1;
}
inline int kappaFromIndex(int i) {
  if (i % 2 == 0)
    return -(i + 2) / 2;
  else
    return (i + 1) / 2;
}

// Given an atomic symbol (H, He, etc.), will return Z
// Note: Symbol must be exact, including capitalisation
inline int get_z(std::string at) {
  for (int z = 0; z < 121; z++) {
    if (at == atom_name_z[z])
      return z;
  }
  int t_z = std::stoi(at);
  if (t_z > 0 && t_z < 137)
    return t_z;
  std::cout << "\n FAILURE 47 in atomInfo: " << at << " not found.\n";
  return 0;
}

// Shell configurations for Noble gasses (Group 8)
const std::vector<int> core_He = {2};
const std::vector<int> core_Ne = {2, 2, 6};
const std::vector<int> core_Ar = {2, 2, 6, 2, 6};
const std::vector<int> core_Kr = {2, 2, 6, 2, 6, 10, 2, 6};
const std::vector<int> core_Xe = {2, 2, 6, 2, 6, 10, 2, 6, 10, 0, 2, 6};
const std::vector<int> core_Rn = {2,  2, 6, 2,  6, 10, 2, 6, 10,
                                  14, 2, 6, 10, 0, 0,  2, 6};
const std::vector<int> core_Og = {2,  2,  6, 2, 6, 10, 2, 6, 10, 14, 2, 6,
                                  10, 14, 0, 2, 6, 10, 0, 0, 0,  2,  6};

// Some other useful 'semi' full shells (transition)
const std::vector<int> core_Zn = {2, 2, 6, 2, 6, 10, 2};
const std::vector<int> core_Cd = {2, 2, 6, 2, 6, 10, 2, 6, 10, 0, 2};
const std::vector<int> core_Hg = {2,  2,  6, 2, 6,  10, 2, 6,
                                  10, 14, 2, 6, 10, 0,  0, 2};

const std::vector<int> core_n = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,
                                 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8,
                                 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9};
const std::vector<int> core_l = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4,
                                 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1,
                                 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8};

} // namespace ATI
