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

Precondition: For efficiency purposes, all operators suppose their arguments to be different from infinity or any other special value (generally bottom or top).
*/

namespace lala {

template <typename T, typename = void>
struct value_type {
  static constexpr bool value = false;
  using type = T;
};

template <typename T>
struct value_type<T, std::void_t<typename T::ValueType>> {
  static constexpr bool value = true;
  using type = T::ValueType;
};

template <typename L>
typename value_type<L>::type unwrap(L x) {
  if constexpr(value_type<L>::value) {
    return x.value();
  }
  else {
    return x;
  }
}

template<Approx appx = EXACT, typename L>
CUDA typename neg_z<L>::type neg(L a) { return typename neg_z<L>::type(-unwrap(a)); }

template<Approx appx = EXACT, typename L>
CUDA typename abs_z<L>::type abs(L a) { return typename abs_z<L>::type(std::abs(unwrap(a))); }

template<Approx appx = EXACT, typename L, typename K>
CUDA typename add_z<L, K>::type add(L a, K b) { return typename add_z<L, K>::type(unwrap(a) + unwrap(b)); }

template<Approx appx = EXACT, typename L, typename K>
CUDA typename sub_z<L, K>::type sub(L a, K b) { return typename sub_z<L, K>::type(unwrap(a) - unwrap(b)); }

template<Approx appx = EXACT, typename L, typename K>
CUDA typename mul_z<L, K>::type mul(L a, K b) { return typename mul_z<L, K>::type(unwrap(a) * unwrap(b)); }

template<class R, Approx appx>
using UpRounding =
  std::enable_if_t<
    (R::increasing && appx == UNDER) ||
    (R::decreasing && appx == OVER)
  , bool>;

template<class R, Approx appx>
using DownRounding =
  std::enable_if_t<
    (R::increasing && appx == OVER) ||
    (R::decreasing && appx == UNDER)
  , bool>;

template<class A, class B>
struct select_non_void { using type = A; };
template <class B>
struct select_non_void<void, B> { using type = B; };

template<Approx appx, typename R = void, typename L, typename K, typename R2 = select_non_void<R, typename div_z<L, K>::type>::type, UpRounding<R, appx> = true>
CUDA R2 div(L x, K y) {
  auto a = unwrap(x);
  auto b = unwrap(y);
  assert(b != 0);
  auto r = a / b;
  // division is rounded towards zero.
  // We add one only if `r` was truncated and `a, b` are of equal sign (so the division operated in the positive numbers).
  // Inspired by https://stackoverflow.com/questions/921180/how-can-i-ensure-that-a-division-of-integers-is-always-rounded-up/926806#926806
  return R2((a % b != 0 && a > 0 == b > 0) ? r + 1 : r);
}

/** Rounding down the result a / b (towards negative infinity). */
template<Approx appx, typename R = void, typename L, typename K, typename R2 = select_non_void<R, typename div_z<L, K>::type>::type, DownRounding<R2, appx> = true>
CUDA R2 div(L x, K y) {
  auto a = unwrap(x);
  auto b = unwrap(y);
  assert(b != 0);
  auto r = a / b;
  return R2((a % b != 0 && a > 0 != b > 0) ? r - 1 : r);
}

template<Approx appx = EXACT, typename L>
CUDA typename sqr_z<L>::type sqr(L a) { return typename sqr_z<L>::type(unwrap(a)*unwrap(a)); }

template<Approx appx = EXACT, typename L, typename K>
CUDA typename pow_z<L, K>::type pow(L a, K b) { return typename pow_z<L, K>::type(std::pow(unwrap(a), unwrap(b))); }

template<class O, Approx appx = EXACT, typename L, typename K>
CUDA typename geq_t<O, L, K>::type geq(L a, K b) { return leq<O>(b, a); }

template<class O, Approx appx = EXACT, typename L, typename K>
CUDA typename gt_t<O, L, K>::type gt(L a, K b) { return lt<O>(b, a); }

template<Approx appx = EXACT, typename L>
CUDA typename not_t<L>::type lnot(L a) { return typename not_t<L>::type(!unwrap(a)); }

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

template<Approx appx = EXACT, typename L, typename K>
CUDA typename imply_t<L, K>::type imply(L a, K b) { return typename imply_t<L, K>::type(!unwrap(a) || unwrap(b)); }

template<Approx appx = EXACT, typename L, typename K>
CUDA typename equiv_t<L, K>::type equiv(L a, K b) { return typename equiv_t<L, K>::type(unwrap(a) == unwrap(b)); }

template<Approx appx = EXACT, typename L, typename K>
CUDA typename xor_t<L, K>::type lxor(L a, K b) { return typename xor_t<L, K>::type(!unwrap(a) != !unwrap(b)); }

}

#endif
