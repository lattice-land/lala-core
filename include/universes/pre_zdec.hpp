// Copyright 2023 Pierre Talbot

#ifndef PRE_ZDEC_HPP
#define PRE_ZDEC_HPP

#include "../logic/logic.hpp"
#include "pre_zinc.hpp"

namespace lala {

template <class VT>
struct PreZInc;

/** `PreZDec` is a pre-abstract universe \f$ \langle \{\infty, \ldots, 2, 1, 0, -1, -2, \ldots, -\infty\}, \leq \rangle \f$ totally ordered by the reversed natural arithmetic comparison operator.
    It is used to represent constraints of the form \f$ x \leq k \f$ where \f$ k \f$ is an integer.
*/
template <class VT>
struct PreZDec {
  using this_type = PreZDec<VT>;
  using dual_type = PreZInc<VT>;
  using value_type = VT;
  using increasing_type = dual_type;

  static_assert(std::is_integral_v<value_type>, "PreZDec only works over integer types.");

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = true;
  constexpr static const bool complemented = false;
  constexpr static const bool increasing = false;
  constexpr static const char *name = "ZDec";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return 0; }
  CUDA constexpr static value_type one() { return 1; }

  template <class F>
  using iresult = IResult<value_type, F>;

  template <class F>
  CUDA static iresult<F> interpret_tell(const F &f) {
    return dual_type::interpret_ask(f);
  }

  template <class F>
  CUDA static iresult<F> interpret_ask(const F &f) {
    return dual_type::interpret_tell(f);
  }

  template <class F>
  CUDA static iresult<F> interpret_type(const F &f) {
    auto r = dual_type::interpret_type(f);
    if (r.has_value() && r.value() == dual_type::bot()) {
      return std::move(r).map(bot());
    }
    return r;
  }

  CUDA static constexpr Sig sig_order() { return GEQ; }
  CUDA static constexpr Sig sig_strict_order() { return GT; }
  CUDA static constexpr value_type bot() { return dual_type::top(); }
  CUDA static constexpr value_type top() { return dual_type::bot(); }
  CUDA static constexpr value_type join(value_type x, value_type y) { return dual_type::meet(x, y); }
  CUDA static constexpr value_type meet(value_type x, value_type y) { return dual_type::join(x, y); }
  CUDA static constexpr bool order(value_type x, value_type y) { return dual_type::order(y, x); }
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return dual_type::strict_order(y, x); }
  CUDA static constexpr value_type next(value_type x) { return dual_type::prev(x); }
  CUDA static constexpr value_type prev(value_type x) { return dual_type::next(x); }
  CUDA static constexpr bool is_supported_fun(Sig sig) { return sig != ABS && dual_type::is_supported_fun(sig); }
  template <Sig sig> CUDA static constexpr value_type fun(value_type x) {
    static_assert(is_supported_fun(sig), "Unsupported unary function.");
    return dual_type::template fun<sig>(x);
  }
  template <Sig sig> CUDA static constexpr value_type fun(value_type x, value_type y) { return dual_type::template fun<sig>(x, y); }
};

} // namespace lala

#endif