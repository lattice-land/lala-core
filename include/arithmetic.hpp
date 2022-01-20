// Copyright 2021 Pierre Talbot

#ifndef ARITHMETIC_HPP
#define ARITHMETIC_HPP

#include <type_traits>

/** \file arithmetic.hpp
This file provides abstract interpretations of arithmetic functions such as addition, subtraction and so on.
For instance, the function \f$ +: \mathbb{R} \times \mathbb{R} \to \mathbb{R} \f$ can be under- or over-approximated by several abstract functions in different abstract domain.
In the abstract domain of increasing floating point number, the under-approximation of addition is given by \f$ \mathit{add_up}: fi \times fi \to fi \f$ which rounds the result towards positive infinity.
Similarly, we have the over-approximation version \f$ \mathit{add_down}: fi \times fi \to fi \f$ rounding the result towards negative infinity.
For integers, the under- and over-approximation versions are the same.

Precondition: For efficiency purposes, all operators suppose their arguments to be different from infinity or any other special value (generally bottom or top).
*/

namespace lala {

template<typename K>
using IsInteger = std::enable_if_t<std::numeric_limits<K>::is_integer, bool>;

template<Approx appx> using IsUnder = std::enable_if_t<appx == UNDER, bool>;
template<Approx appx> using IsOver = std::enable_if_t<appx == OVER, bool>;

// I. Integer-like functions
// =========================

// I.a Ground functions
// --------------------

template<Approx appx = EXACT, typename K, IsInteger<K> = true> CUDA K neg(K a) { return -a; }
template<Approx appx = EXACT, typename K, IsInteger<K> = true> CUDA K add(K a, K b) { return a + b; }
template<Approx appx = EXACT, typename K, IsInteger<K> = true> CUDA K sub(K a, K b) { return a - b; }
template<Approx appx = EXACT, typename K, IsInteger<K> = true> CUDA K mul(K a, K b) { return a * b; }

/** Rounding up the result a / b (towards infinity). */
template<Approx appx, typename K, IsInteger<K> = true, IsUnder<appx> = true>
CUDA K div(K a, K b) {
  assert(b != 0);
  K r = a / b;
  // division is rounded towards zero.
  // We add one only if `r` was truncated and `a, b` are of equal sign (so the division operated in the positive numbers).
  // Inspired by https://stackoverflow.com/questions/921180/how-can-i-ensure-that-a-division-of-integers-is-always-rounded-up/926806#926806
  return (a % b != 0 && a > 0 == b > 0) ? r + 1 : r;
}

/** Rounding down the result a / b (towards negative infinity). */
template<Approx appx, typename K, IsInteger<K> = true, IsOver<appx> = true>
CUDA K div(K a, K b) {
  assert(b != 0);
  K r = a / b;
  return (a % b != 0 && a > 0 != b > 0) ? r - 1 : r;
}
}

#endif
