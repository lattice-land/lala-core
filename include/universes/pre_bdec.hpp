// Copyright 2023 Pierre Talbot

#ifndef PRE_BDEC_HPP
#define PRE_BDEC_HPP

#include "../logic/logic.hpp"
#include "pre_binc.hpp"

namespace lala {

/** `PreBDec` is a pre-abstract universe \f$ \langle \{\mathit{true}, \mathit{false}\}, \leq \rangle \f$ such that \f$ \mathit{false} \geq \mathit{true} \f$.
    It is used to represent Boolean variables which truth's value progresses from \f$ \mathit{true} \f$ to \f$ \mathit{false} \f$.
    Note that this type is unable to represent Boolean domain which requires four states: unknown (bot), true, false and failed (top).
    To obtain such a domain, you should use `Interval<BInc>`.
*/
struct PreBDec {
  using this_type = PreBDec;
  using dual_type = PreBInc;
  using value_type = dual_type::value_type;
  using increasing_type = PreBInc;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = false;
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = true;
  constexpr static const bool complemented = false;
  constexpr static const bool increasing = false;
  constexpr static const char* name = "BDec";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return false; }
  CUDA constexpr static value_type one() { return true; }

  template<class F>
  using iresult = IResult<value_type, F>;

  template<class F, class Sort>
  CUDA static iresult<F> interpret(const F& f, const Sort& sort, Approx appx) { return dual_type::interpret(f, sort, dapprox(appx)); }

  template<class F>
  CUDA static iresult<F> interpret_type(const F& f) {
    auto r = dual_type::interpret_type(f);
    if (r.has_value() && r.value() == dual_type::bot())
    {
      return std::move(r).map(bot());
    }
    return r;
  }

  CUDA static constexpr Sig sig_order() { return GEQ; }
  CUDA static constexpr Sig sig_strict_order() { return GT; }
  CUDA static constexpr value_type bot() { return dual_type::top(); }
  CUDA static constexpr value_type top() { return dual_type::bot(); }
  CUDA static constexpr value_type join(value_type x, value_type y) { return dual_type::meet(x, y);}
  CUDA static constexpr value_type meet(value_type x, value_type y) { return dual_type::join(x, y);}
  CUDA static constexpr bool order(value_type x, value_type y) { return dual_type::order(y, x);}
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return dual_type::strict_order(y, x);}
  CUDA static constexpr value_type next(value_type x) { return dual_type::prev(x); }
  CUDA static constexpr value_type prev(value_type x) { return dual_type::next(x); }
  CUDA static constexpr bool is_supported_fun(Sig sig) { return dual_type::is_supported_fun(sig); }
  template<Sig sig> CUDA static constexpr value_type fun(value_type x) { return dual_type::fun<sig>(x); }
  template<Sig sig> CUDA static constexpr value_type fun(value_type x, value_type y) { return dual_type::fun<sig>(x, y); }
};

} // namespace lala

#endif