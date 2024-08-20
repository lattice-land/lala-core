// Copyright 2023 Pierre Talbot

#ifndef LALA_CORE_PRE_ZLB_HPP
#define LALA_CORE_PRE_ZLB_HPP

#include "../logic/logic.hpp"
#include "pre_zub.hpp"

namespace lala {

template <class VT>
struct PreZUB;

/** `PreZLB` is a pre-abstract universe \f$ \langle \{\infty, \ldots, 2, 1, 0, -1, -2, \ldots, -\infty\}, \geq \rangle \f$ totally ordered by the reversed natural arithmetic comparison operator.
    It is used to represent constraints of the form \f$ x \geq k \f$ where \f$ k \f$ is an integer.
*/
template <class VT>
struct PreZLB {
  using this_type = PreZLB<VT>;
  using dual_type = PreZUB<VT>;
  using value_type = VT;
  using lower_bound_type = this_type;
  using upper_bound_type = dual_type;

  static_assert(std::is_integral_v<value_type>, "PreZLB only works over integer types.");

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool preserve_join = true;
  constexpr static const bool preserve_meet = true;
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = true;
  constexpr static const bool is_lower_bound = true;
  constexpr static const bool is_upper_bound = false;
  constexpr static const char *name = "ZLB";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return 0; }
  CUDA constexpr static value_type one() { return 1; }

  template <bool diagnose, class F>
  CUDA static bool interpret_tell(const F &f, value_type& tell, IDiagnostics& diagnostics) {
    return dual_type::template interpret_ask<diagnose, F, true>(f, tell, diagnostics);
  }

  template <bool diagnose, class F>
  CUDA static bool interpret_ask(const F &f, value_type& ask, IDiagnostics& diagnostics) {
    return dual_type::template interpret_tell<diagnose, F, true>(f, ask, diagnostics);
  }

  template <bool diagnose, class F>
  CUDA static bool interpret_type(const F &f, value_type& k, IDiagnostics& diagnostics) {
    return dual_type::template interpret_type<diagnose, F, true>(f, k, diagnostics);
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
  CUDA static constexpr value_type project(Sig fun, value_type x) {
    if(fun == ABS) { return x >= 0 ? x : 0; }
    else {
      return dual_type::project(fun, x);
    }
  }
  CUDA static constexpr value_type project(Sig fun, value_type x, value_type y) {
    return dual_type::project(fun, x, y);
  }
};

} // namespace lala

#endif