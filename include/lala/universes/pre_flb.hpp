// Copyright 2023 Pierre Talbot

#ifndef LALA_CORE_PRE_FLB_HPP
#define LALA_CORE_PRE_FLB_HPP

#include "../logic/logic.hpp"
#include "pre_finc.hpp"

namespace lala {

template<class VT>
struct PreFUB;

/** `PreFLB` is a pre-abstract universe \f$ \langle \mathbb{F}\setminus\{NaN\}, \geq \rangle \f$ totally ordered by the reversed floating-point arithmetic comparison operator.
    We work on a subset of floating-point numbers without NaN.
    It is used to represent (and possibly approximate) constraints of the form \f$ x \geq k \f$ where \f$ k \f$ is a real number.
*/
template<class VT>
struct PreFLB {
  using this_type = PreFLB<VT>;
  using dual_type = PreFUB<VT>;
  using value_type = VT;
  using increasing_type = dual_type;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool preserve_join = true;
  constexpr static const bool preserve_meet = true;
  /** Note that -0 and +0 are treated as the same element. */
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = false;
  constexpr static const bool increasing = false;
  constexpr static const char* name = "FLB";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return 0.0; }
  CUDA constexpr static value_type one() { return 1.0; }

  template <bool diagnose, class F>
  CUDA static bool interpret_tell(const F &f, value_type& tell, IDiagnostics& diagnostics) {
    return dual_type::template interpret_ask<diagnose>(f, tell, diagnostics);
  }

  template <bool diagnose, class F>
  CUDA static bool interpret_ask(const F &f, value_type& ask, IDiagnostics& diagnostics) {
    return dual_type::template interpret_tell<diagnose>(f, ask, diagnostics);
  }

  template<bool diagnose, class F>
  CUDA static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
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
    switch(fun) {
      case ADD: return battery::add_down(x, y);
      case SUB: return battery::sub_down(x, y);
      case MUL: return battery::mul_down(x, y);
      case DIV: return battery::div_down(x, y);
      default: return dual_type::project(fun, x, y);
    }
  }
};

} // namespace lala

#endif