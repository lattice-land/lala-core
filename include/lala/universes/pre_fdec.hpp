// Copyright 2023 Pierre Talbot

#ifndef LALA_CORE_PRE_FDEC_HPP
#define LALA_CORE_PRE_FDEC_HPP

#include "../logic/logic.hpp"
#include "pre_finc.hpp"

namespace lala {

template<class VT>
struct PreFInc;

/** `PreFDec` is a pre-abstract universe \f$ \langle \mathbb{F}\setminus\{NaN\}, \leq \rangle \f$ totally ordered by the reversed floating-point arithmetic comparison operator.
    We work on a subset of floating-point numbers without NaN.
    It is used to represent (and possibly approximate) constraints of the form \f$ x \leq k \f$ where \f$ k \f$ is a real number.
*/
template<class VT>
struct PreFDec {
  using this_type = PreFDec<VT>;
  using dual_type = PreFInc<VT>;
  using value_type = VT;
  using increasing_type = dual_type;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  /** Note that -0 and +0 are treated as the same element. */
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = false;
  constexpr static const bool complemented = false;
  constexpr static const bool increasing = false;
  constexpr static const char* name = "FDec";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return 0.0; }
  CUDA constexpr static value_type one() { return 1.0; }

  template<class F>
  using iresult = IResult<value_type, F>;

  template <class F>
  CUDA NI static iresult<F> interpret_tell(const F &f) {
    return dual_type::interpret_ask(f);
  }

  template <class F>
  CUDA NI static iresult<F> interpret_ask(const F &f) {
    return dual_type::interpret_tell(f);
  }

  template<class F>
  CUDA NI static iresult<F> interpret_type(const F& f) {
    auto r = dual_type::interpret_type(f);
    if (r.has_value() && r.value() == dual_type::bot())
    {
      return std::move(r).map(bot());
    }
    return r;
  }

  template<class F>
  CUDA NI static F deinterpret(const value_type& v) {
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

  template<Sig sig>
  CUDA static constexpr value_type fun(value_type x) {
    static_assert(is_supported_fun(sig), "Unsupported unary function.");
    return dual_type::template fun<sig>(x);
  }

  template<Sig sig>
  CUDA static constexpr value_type fun(value_type x, value_type y) {
    static_assert(is_supported_fun(sig), "Unsupported binary function.");
    switch(sig) {
      case ADD: return battery::add_up(x, y);
      case SUB: return battery::sub_up(x, y);
      case MUL: return battery::mul_up(x, y);
      case DIV: return battery::div_up(x, y);
      default: return dual_type::template fun<sig>(x, y);
    }
  }
};

} // namespace lala

#endif