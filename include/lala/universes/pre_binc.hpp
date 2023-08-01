// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_PRE_BINC_HPP
#define LALA_CORE_PRE_BINC_HPP

#include "../logic/logic.hpp"

namespace lala {

struct PreBDec;

/** `PreBInc` is a pre-abstract universe \f$ \langle \{\mathit{true}, \mathit{false}\}, \leq \rangle \f$ such that \f$ \mathit{false} \leq \mathit{true} \f$.
    It is used to represent Boolean variables which truth's value progresses from \f$ \mathit{false} \f$ to \f$ \mathit{true} \f$.
    Note that this type is unable to represent Boolean domain which requires four states: unknown (bot), true, false and failed (top).
    To obtain such a domain, you should use `Interval<BInc>`.
*/
struct PreBInc {
  using this_type = PreBInc;
  using dual_type = PreBDec;
  using value_type = bool;
  using increasing_type = PreBInc;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = false;
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = true;
  constexpr static const bool complemented = false;
  constexpr static const bool increasing = true;
  constexpr static const char* name = "BInc";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return false; }
  CUDA constexpr static value_type one() { return true; }

  template<class F>
  using iresult = IResult<value_type, F>;

  /** Interpret a formula into an upset Boolean lattice.
   * \return The result of the interpretation when the formula `f` is a constant of type `Bool`. Otherwise it returns an explanation of the error. */
  template<class F>
  CUDA NI static iresult<F> interpret_tell(const F& f) {
    if(f.is(F::B)) {
      return iresult<F>(f.b());
    }
    return iresult<F>(IError<F>(true, name, "Only constant of types `Bool` can be interpreted in a Boolean domain.", f));
  }

  /** In this domain, the ask version of any constraint is the same as the tell version.
   * This is because this domain can exactly represent Boolean values.
  */
  template<class F>
  CUDA NI static iresult<F> interpret_ask(const F& f) {
    return interpret_tell(f);
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
   * \return `bot()` if the type of the existentially quantified variable is `Bool`. Otherwise it returns an explanation of the error.
  */
  template<class F>
  CUDA NI static iresult<F> interpret_type(const F& f) {
    assert(f.is(F::E));
    const auto& cty = battery::get<1>(f.exists());
    if(cty.is_bool()) {
      return iresult<F>(bot());
    }
    else {
      const auto& vname = battery::get<0>(f.exists());
      return iresult<F>(IError<F>(true, name, "The type of `" + vname + "` can only be `Bool`.", f));
    }
  }

  /** Given a Boolean value, create a logical constant representing that value.
   * Note that the lattice order has no influence here.
  */
  template<class F>
  CUDA NI static F deinterpret(const value_type& v) {
    return F::make_bool(v);
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{BInc} b \Leftrightarrow a \leq b \f$.
      \return The logical symbol `LEQ`. */
  CUDA static constexpr Sig sig_order() { return LEQ; }
  CUDA static constexpr Sig sig_strict_order() { return LT; }

  /** \f$ \bot \f$ is represented by `false`. */
  CUDA static constexpr value_type bot() { return false; }

  /** \f$ \top \f$ is represented by `true`. */
  CUDA static constexpr value_type top() { return true; }

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

  /**  From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ y \f$ is a cover of \f$ x \f$.

    \return `true`. */
  CUDA static constexpr value_type next(value_type x) { return true; }

  /** From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ x \f$ is a cover of \f$ y \f$.

   \return `false`. */
  CUDA static constexpr value_type prev(value_type x) { return false; }

  CUDA static constexpr bool is_supported_fun(Sig sig) {
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

  template<Sig sig>
  CUDA static constexpr value_type fun(value_type x) {
    static_assert(sig == NOT, "Unsupported unary function.");
    switch(sig) {
      case NOT: return !x;
      default: assert(0); return x;
    }
  }

  template<Sig sig>
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