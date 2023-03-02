// Copyright 2022 Pierre Talbot

#ifndef PRE_BINC_HPP
#define PRE_BINC_HPP

#include "../logic/logic.hpp"
#include "chain_pre_dual.hpp"

namespace lala {

/** `PreBInc` is a pre-abstract universe \f$ \langle \{\mathit{true}, \mathit{false}\}, \leq \rangle \f$ such that \f$ \mathit{false} \leq \mathit{true} \f$.
    It is used to represent Boolean variables which truth's value progresses from \f$ \mathit{false} \f$ to \f$ \mathit{true} \f$.
    Note that this type is unable to represent Boolean domain which requires four states: unknown (bot), true, false and failed (top).
    To obtain such a domain, you should use `Interval<BInc>`.
*/
struct PreBInc {
  using this_type = PreBInc;
  using reverse_type = ChainPreDual<this_type>;
  using value_type = bool;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = false;
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_inner_covers = true;
  constexpr static const bool complemented = false;
  constexpr static const bool increasing = true;
  constexpr static const char* name = "BInc";
  constexpr static const char* dual_name = "BDec";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return false; }
  CUDA constexpr static value_type one() { return true; }

  template<class F>
  using iresult = IResult<value_type, F>;

  /** Besides the type `CType::Bool`, Booleans can be over-approximated by arithmetic types in the logical formula, integer or floating-point numbers.
   * `false` is given by the constant 0 and `true` by any other number. */
  template<class F, class Sort, bool dualize = false>
  CUDA static iresult<F> interpret(const F& f, const Sort& sort, Approx appx) {
    if(f.is(F::B) && sort.is_bool()) {
      return iresult<F>(f.b());
    }
    else if(f.is(F::Z)) {
      auto z = f.z();
      auto warning = IError<F>(false, name, "Integer constant interpreted as a Boolean (0 is false, everything else is true).", f);
      if(z == 0) {
        return iresult<F>(false, std::move(warning));
      }
      else {
        return iresult<F>(true, std::move(warning));
      }
    }
    else if(f.is(F::R) && sort.is_bool()) {
      auto lb = battery::rd_cast<value_type>(battery::get<0>(f.r()));
      auto ub = battery::ru_cast<value_type>(battery::get<1>(f.r()));
      auto warning = IError<F>(false, name, "Real constant interpreted as a Boolean (0 is false, everything else is true).", f);
      if(lb == ub && lb == 0) {
        return iresult<F>(false, std::move(warning));
      }
      else {
        return iresult<F>(true, std::move(warning));
      }
    }
    return iresult<F>(IError<F>(true, name, "Only constant of types `CType::Bool`, `CType::Int` and `CType::Real` can be interpreted in a Boolean domain.", f));
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
   * Interpretations:
        * Variables of type `CType::Bool` are interpreted exactly (\f$ \mathbb{B} = \gamma(\bot) \f$).
        * Variables of type `CType::Int` are under-approximated  (\f$ \mathbb{Z} \supseteq \gamma(\bot) \f$).
        * Variables of type `CType::Real` are under-approximated (\f$ \mathbb{R} \supseteq \gamma(\bot) \f$). */
  template<class F>
  CUDA static iresult<F> interpret_type(const F& f) {
    assert(f.is(F::E));
    const auto& vname = battery::get<0>(f.exists());
    const auto& cty = battery::get<1>(f.exists());
    if(cty.is_bool()) {
      return iresult<F>(bot());
    }
    else if((cty.is_int() || cty.is_real()) && f.is_under()) {
      return iresult<F>(bot(), IError<F>(false, name, "Constant of type `CType::Real` or `CType::Int` under-approximated by a Boolean.", f));
    }
    else {
      return iresult<F>(IError<F>(true, name, "The type of `" + vname + "` can only be `CType::Bool` or under-approximated `CType::Int` or `CType::Real`.", f));
    }
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{BInc} b \Leftrightarrow a \leq b \f$.
      \return The logical symbol `LEQ`. */
  CUDA static constexpr Sig sig_order() { return LEQ; }
  CUDA static constexpr Sig dual_sig_order() { return GEQ; }

  /** The logical predicate symbol corresponding to the strict order of this pre-universe.
      We have \f$ a <_\mathit{BInc} b \Leftrightarrow a < b \f$.
      \return The logical symbol `LT`. */
  CUDA static constexpr Sig sig_strict_order() { return LT; }
  CUDA static constexpr Sig dual_sig_strict_order() { return GT; }

  /** \f$ \bot \f$ is represented by `false`. */
  CUDA static constexpr value_type bot() {
    return false;
  }

  /** \f$ \top \f$ is represented by `true`. */
  CUDA static constexpr value_type top() {
    return true;
  }

  /** \return \f$ x \sqcup y \f$ defined as \f$ x \lor y \f$. */
  CUDA static constexpr value_type join(value_type x, value_type y) { return x || y; }

  /** \return \f$ x \sqcap y \f$ defined as \f$ x \land y \f$. */
  CUDA static constexpr value_type meet(value_type x, value_type y) { return x && y; }

  /** \return \f$ \mathit{true} \f$ if \f$ x \leq_\mathit{BInc} y \f$ where the order \f$ \mathit{false} \leq_\mathit{BInc} \mathit{true} \f$, otherwise returns \f$ \mathit{false} \f$.
      Note that the order is the Boolean implication, \f$ x \leq y \Rightleftarrow x \Rightarrow y \f$. */
  CUDA static constexpr bool order(value_type x, value_type y) { return !x || y; }

/** \return \f$ \mathit{true} \f$ if \f$ x \leq_\mathit{BInc} y \f$ where the order \f$ \mathit{false} \leq_\mathit{BInc} \mathit{true} \f$, otherwise returns \f$ \mathit{false} \f$.
      Note that the strict order is the Boolean converse nonimplication, \f$ x \leq y \Rightleftarrow x \not\Leftarrow y \f$. */
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return !x && y; }

  /** `true` if the element \f$ x \f$ has a unique cover in the abstract universe. */
  CUDA static constexpr bool has_unique_next(value_type x) { return true; }

  /** `true` if the element \f$ x \f$ covers a unique element in the abstract universe. */
  CUDA static constexpr bool has_unique_prev(value_type x) { return true; }

  /**  From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ y \f$ is a cover of \f$ x \f$.

    \return `true`. */
  CUDA static constexpr value_type next(value_type x) {
    return true;
  }

  /** From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ x \f$ is a cover of \f$ y \f$.

   \return `false`. */
  CUDA static constexpr value_type prev(value_type x) {
    return false;
  }

  CUDA static constexpr bool is_supported_fun(Approx, Sig sig) {
    switch(sig) {
      case AND:
      case OR:
      case IMPLY:
      case EQUIV:
      case XOR:
      case NOT:
      case EQ:
      case NEQ:
        return true;
      default:
        return false;
    }
  }

  template<Approx appx, Sig sig>
  CUDA static constexpr value_type fun(value_type x) {
    static_assert(sig == NOT, "Unsupported unary function.");
    switch(sig) {
      case NOT: return !x;
      default: assert(0); return x;
    }
  }

  template<Approx appx, Sig sig>
  CUDA static constexpr value_type fun(value_type x, value_type y) {
    static_assert(sig == AND || sig == OR || sig == IMPLY || sig == EQUIV || sig == XOR,
      "Unsupported binary function.");
    switch(sig) {
      case AND: return x && y;
      case OR: return x || y;
      case IMPLY: return !x || y;
      case EQUIV:
      case EQ: return x == y;
      case XOR:
      case NEQ: return x != y;
      default: assert(0); return x;
    }
  }
};

} // namespace lala

#endif