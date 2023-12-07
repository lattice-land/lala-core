// Copyright 2023 Pierre Talbot

#ifndef LALA_CORE_PRE_ZDEC_HPP
#define LALA_CORE_PRE_ZDEC_HPP

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

  template <bool diagnose, class F>
  CUDA static bool interpret_tell(const F &f, value_type& tell, IDiagnostics<F>& diagnostics) {
    return dual_type::template interpret_ask<diagnose>(f, tell, diagnostics);
  }

  template <bool diagnose, class F>
  CUDA static bool interpret_ask(const F &f, value_type& ask, IDiagnostics<F>& diagnostics) {
    return dual_type::template interpret_tell<diagnose>(f, ask, diagnostics);
  }

  template <bool diagnose, class F>
  CUDA static bool interpret_type(const F &f, value_type& k, IDiagnostics<F>& diagnostics) {
    bool res = dual_type::template interpret_type<diagnose>(f, k, diagnostics);
    // We reverse top and bottom due to the dual interpretation.
    if (res && k == dual_type::bot()) {
      k = bot();
    }
    return res;
  }

  template<class F>
  CUDA static F deinterpret(const value_type& v) {
    return dual_type::template deinterpret<F>(v);
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