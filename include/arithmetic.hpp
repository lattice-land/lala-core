// Copyright 2021 Pierre Talbot

#ifndef ARITHMETIC_HPP
#define ARITHMETIC_HPP

#include <type_traits>
#include <utility>
#include <cmath>
#include "z.hpp"

/** \file arithmetic.hpp
This file provides abstract interpretations of arithmetic functions such as addition, subtraction and so on.
For instance, the function \f$ +: \mathbb{R} \times \mathbb{R} \to \mathbb{R} \f$ can be under- or over-approximated by several abstract functions in different abstract domain.
In the abstract domain of increasing floating point number, the under-approximation of addition is given by \f$ \mathit{add_up}: fi \times fi \to fi \f$ which rounds the result towards positive infinity.
Similarly, we have the over-approximation version \f$ \mathit{add_down}: fi \times fi \to fi \f$ rounding the result towards negative infinity.
For integers, the under- and over-approximation versions are the same.

Bottom and top elements are handled correctly in all interpretations.
However, there is nothing done against underflow and overflow, ideally we would need "saturation arithmetic" which is not easy to implement efficiently on integers (it is however built-in for floating-point numbers).
*/

namespace lala {

template<class L>
struct is_lattice {
  static constexpr bool value = false;
};

template<class U>
struct is_lattice<ZTotalOrder<U>> {
  static constexpr bool value = true;
};

template<class T>
inline constexpr bool is_lattice_v = is_lattice<T>::value;

template<class L>
struct has_inf_top {
  static constexpr bool value = true;
};

template<class V>
struct has_inf_top<ZPDec<V>> {
  static constexpr bool value = false;
};

template<class V>
struct has_inf_top<ZNInc<V>> {
  static constexpr bool value = false;
};

template<template<class, Sign> class U, class V>
struct has_inf_top<ZTotalOrder<U<V, BOUNDED>>> {
  static constexpr bool value = false;
};

template<class T>
inline constexpr bool has_inf_top_v = has_inf_top<T>::value;

template<class L>
struct has_inf_bot {
  static constexpr bool value = true;
};

template<class V>
struct has_inf_bot<ZPInc<V>> {
  static constexpr bool value = false;
};

template<class V>
struct has_inf_bot<ZNDec<V>> {
  static constexpr bool value = false;
};

template<template<class, Sign> class U, class V>
struct has_inf_bot<ZTotalOrder<U<V, BOUNDED>>> {
  static constexpr bool value = false;
};

template<class T>
inline constexpr bool has_inf_bot_v = has_inf_bot<T>::value;

#define TOP_UNARY(x, L, R) \
  if constexpr(is_lattice_v<L> && has_inf_top_v<L>) { \
    if(x.is_top().guard()) return R::top();           \
  }

// `value()` is allowed in a fully functional context.
#define BOT_UNARY(x, L, R) \
 if constexpr(is_lattice_v<L> && has_inf_bot_v<L>) { \
    if(x.is_bot().value()) return R::bot();          \
  }

#define BOT_TOP_UNARY(L, R) \
  TOP_UNARY(a, L, R)        \
  BOT_UNARY(a, L, R)

#define BOT_TOP_BINARY(L, K, R) \
  TOP_UNARY(a, L, R)            \
  TOP_UNARY(b, K, R)            \
  BOT_UNARY(a, L, R)            \
  BOT_UNARY(b, K, R)

template<Approx appx = EXACT, class L>
CUDA typename neg_z<L>::type neg(L a) {
  using R = typename neg_z<L>::type;
  BOT_TOP_UNARY(L, R)
  return R(-unwrap(a));
}

template<Approx appx = EXACT, class L>
CUDA typename abs_z<L>::type abs(L a) {
  using R = typename abs_z<L>::type;
  TOP_UNARY(a, L, R)
  auto x = unwrap(a);
  if constexpr(R::increasing) {
    if(x >= 0) return R(x);
    else return R(0);
  }
  else {
    static_assert(R::decreasing);
    if(x < 0) return R::top();
    else return R(x);
  }
}

template<Approx appx = EXACT, class L, class K>
CUDA typename add_z<L, K>::type add(L a, K b) {
  using R = typename add_z<L, K>::type;
  BOT_TOP_BINARY(L, K, R)
  return R(unwrap(a) + unwrap(b));
}

template<Approx appx = EXACT, class L, class K>
CUDA typename sub_z<L, K>::type sub(L a, K b) {
  using R = typename sub_z<L, K>::type;
  BOT_TOP_BINARY(L, K, R)
  return R(unwrap(a) - unwrap(b));
}

template<Approx appx = EXACT, class L, class K>
CUDA typename mul_z<L, K>::type mul(L a, K b) {
  using R = typename mul_z<L, K>::type;
  BOT_TOP_BINARY(L, K, R)
  return R(unwrap(a) * unwrap(b));
}

template<class A, class B>
struct select_non_void { using type = A; };
template <class B>
struct select_non_void<void, B> { using type = B; };

template<Approx appx = EXACT, class R = void, class L, class K,
  class R2 = typename select_non_void<R, typename div_z<L, K>::type>::type, std::enable_if_t<R2::increasing, bool> = true>
CUDA typename div_z<L, K>::type div(L a, K b) {
  using R3 = typename div_z<L, K>::type;
  BOT_TOP_BINARY(L, K, R3)
  auto x = unwrap(a);
  auto y = unwrap(b);
  assert(y != 0);
  auto r = x / y;
  // division is rounded towards zero.
  // We add one only if `r` was truncated and `x, y` are of equal sign (so the division operated in the positive numbers).
  // Inspired by https://stackoverflow.com/questions/921180/how-can-i-ensure-that-a-division-of-integers-is-always-rounded-up/926806#926806
  return R3((x % y != 0 && x > 0 == y > 0) ? r + 1 : r);
}

/** Rounding down the result a / b (towards negative infinity). */
template<Approx appx, class R = void, class L, class K,
  class R2 = typename select_non_void<R, typename div_z<L, K>::type>::type, std::enable_if_t<R2::decreasing, bool> = true>
CUDA typename div_z<L, K>::type div(L a, K b) {
  using R3 = typename div_z<L, K>::type;
  BOT_TOP_BINARY(L, K, R3)
  auto x = unwrap(a);
  auto y = unwrap(b);
  assert(y != 0);
  auto r = x / y;
  return R3((x % y != 0 && x > 0 != y > 0) ? r - 1 : r);
}

template<Approx appx = EXACT, class L>
CUDA typename sqr_z<L>::type sqr(L a) {
  using R = typename sqr_z<L>::type;
  BOT_TOP_UNARY(L, R)
  return R(unwrap(a)*unwrap(a));
}

template<Approx appx = EXACT, class L, class K>
CUDA typename pow_z<L, K>::type pow(L a, K b) {
  using R = typename pow_z<L, K>::type;
  BOT_TOP_BINARY(L, K, R)
  return R(std::pow(unwrap(a), unwrap(b)));
}

template<class O, Approx appx = EXACT, class L, class K>
CUDA typename geq_t<O, L, K>::type geq(L a, K b) { return leq<O>(b, a); }

template<class O, Approx appx = EXACT, class L, class K>
CUDA typename gt_t<O, L, K>::type gt(L a, K b) { return lt<O>(b, a); }

template<Approx appx = EXACT, class L>
CUDA typename not_t<L>::type lnot(L a) {
  return typename not_t<L>::type(!unwrap(a));
}

template <template<class,class> class, class...>
struct fold_t;

template <template<class,class> class OP, class B, class C>
struct fold_t<OP, B, C>
{
  using type = typename OP<B, C>::type;
};

template <template<class,class> class OP, class B, class C, class Head, class... Tail>
struct fold_t<OP, B, C, Head, Tail...>
{
  using type = typename OP<
    typename OP<B, C>::type,
    typename fold_t<OP, Head, Tail...>::type
  >::type;
};

template <class... Ls> using fold_land = fold_t<and_t, Ls...>;
template <class... Ls> using fold_lor = fold_t<or_t, Ls...>;

template<Approx appx = EXACT, class... Ls>
CUDA typename fold_land<Ls...>::type land(Ls... vals) {
  return typename fold_land<Ls...>::type((... && unwrap(vals)));  // C++17 fold expression.
}

template<Approx appx = EXACT, class... Ls>
CUDA typename fold_lor<Ls...>::type lor(Ls... vals) {
  return typename fold_lor<Ls...>::type((... || unwrap(vals)));  // C++17 fold expression.
}

template<Approx appx = EXACT, class L, class K>
CUDA typename imply_t<L, K>::type imply(L a, K b) {
  return typename imply_t<L, K>::type(!unwrap(a) || unwrap(b));
}

template<Approx appx = EXACT, class L, class K>
CUDA typename equiv_t<L, K>::type equiv(L a, K b) {
  return typename equiv_t<L, K>::type(unwrap(a) == unwrap(b));
}

template<Approx appx = EXACT, class L, class K>
CUDA typename xor_t<L, K>::type lxor(L a, K b) {
  return typename xor_t<L, K>::type(!unwrap(a) != !unwrap(b));
}

}

#endif
