// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_PRE_B_HPP
#define LALA_CORE_PRE_B_HPP

#include "../logic/logic.hpp"

namespace lala {

struct PreBD;

/** `PreB` is a domain abstracting the Boolean universe of discourse \f$ \mathbb{B}=\{true,false\} \f$.
    We overload the symbols \f$ \mathit{true} \f$ and \f$ \mathit{false} \f$ to be used in both `PreB` and the concrete domain (\f$ \bot, \top \f$ can be used to specifically refer to PreB elements).
    We have \f$ PreB \triangleq \langle \{\mathit{false}, \mathit{true}\}, \implies, \land, \lor, \mathit{false}, \mathit{true} \rangle \f$ with the usual logical connectors (in particular, \f$ \mathit{false} \implies \mathit{true} \f$, \f$ \bot = \mathit{false} \f$ and \f$ \top = \mathit{true} \f$).

    We have a Galois connection between the concrete domain of values \f$ \mathcal{P}(\mathbb{B}) \f$ and PreB:
    * Concretization: \f$ \gamma(b) \triangleq b = \top ? \{\mathit{true}, \mathit{false}\} : \{\mathit{false}\} \f$.
    * Abstraction: \f$ \alpha(S) \triangleq \mathit{true} \in S ? \top : \bot \f$.

    Beware that, as suggested by the concretization function, \f$ \top \f$ represents both the true and false values in the concrete domain, and should be interpreted as "I don't know yet the value".

    Further, note that this lattice is unable to represent exactly Dunn/Belnap logic which requires four states (which is necessary when working in the abstract): unknown, true, false and failed.
    To obtain Dunn/Belnap logic with a knowledge ordering, you can use `Interval<Bound<PreB>>`.
*/
struct PreB {
  using this_type = PreB;
  using dual_type = PreBD;
  using value_type = bool;

  constexpr static const bool is_natural = true; /** We consider \f$ \top = \mathit{true} \f$ is the natural order on Boolean. */
  using natural_order = PreB;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = false; /** \f$ \gamma(\mathit{false}) = \{\mathit{false}\} \f$, therefore the empty set cannot be represented in this domain. */
  constexpr static const bool preserve_top = true; /** \f$ \gamma(\mathit{unknown}) = \{false, true\} \f$ */
  constexpr static const bool preserve_join = true; /** \f$ \gamma(x \sqcup y) = \gamma(x) \cup \gamma(y) \f$ */
  constexpr static const bool preserve_meet = true; /** \f$ \gamma(x \sqcap y) = \gamma(x) \cap \gamma(y) \f$ */
  constexpr static const bool injective_concretization = true; /** Each element of PreB maps to a different concrete value. */
  constexpr static const bool preserve_concrete_covers = true; /** \f$ x \lessdot y \Leftrightarrow \gamma(x) \lessdot \gamma(y) \f$ */
  constexpr static const char* name = "B";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return false; }
  CUDA constexpr static value_type one() { return true; }

  /** @sequential
   * Interpret a formula into the PreB lattice.
   * \return `true` if an overapproximation of a Boolean constant `b` could be placed in `tell`. Otherwise it returns `false` with a diagnostic. */
  template<bool diagnose, class F, bool dualize=false>
  CUDA static bool interpret_tell(const F& f, value_type& tell, IDiagnostics& diagnostics) {
    if(f.is(F::B)) {
      if(f.b() == dualize) {
        if constexpr(dualize) {
          INTERPRETATION_WARNING("Overapproximating the constant `false` by the top element (which concretization gives {true, false}) in the `PreBD` domain.");
        }
        else {
          INTERPRETATION_WARNING("Overapproximating the constant `true` by the top element (which concretization gives {true, false}) in the `PreB` domain.");
        }
      }
      tell = f.b();
      return true;
    }
    RETURN_INTERPRETATION_ERROR("Only constant of types `Bool` can be interpreted in a Boolean domain.");
  }

  /** @sequential
   * We can only ask if an element of this lattice is `false`, because it cannot exactly represent `true`.
   * This operation can be dualized.
  */
  template<bool diagnose, class F, bool dualize=false>
  CUDA static bool interpret_ask(const F& f, value_type& ask, IDiagnostics& diagnostics) {
    if(f.is(F::B)) {
      if(f.b() == dualize) { /** In the dual, we can only ask for `true` elements. */
        tell = f.b();
        return true;
      }
      else {
        if constexpr(dualize) {
          RETURN_INTERPRETATION_ERROR("This Boolean domain can only represent two values: `true` or 'false'.");
        }
        else {
          RETURN_INTERPRETATION_ERROR("This Boolean domain can only represent two values: `false` or 'I don't know'.");
        }
      }
    }
    RETURN_INTERPRETATION_ERROR("Only constant of types `Bool` can be interpreted in Boolean domains.");
  }

  /** @sequential
   * Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
   * \return `bot()` if the type of the existentially quantified variable is `Bool`. Otherwise it returns an explanation of the error.
   * This operation can be dualized.
  */
  template<bool diagnose, class F, bool dualize = false>
  CUDA NI static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
    assert(f.is(F::E));
    const auto& cty = battery::get<1>(f.exists());
    if(cty.is_bool()) {
      k = dualize ? top() : bot();
      return true;
    }
    else {
      const auto& vname = battery::get<0>(f.exists());
      RETURN_INTERPRETATION_ERROR("The type of `" + vname + "` can only be `Bool` when interpreted in Boolean domains.");
    }
  }

  /** @parallel
   * Given a Boolean value, create a logical constant representing that value.
   * Note that the lattice order has no influence here.
  */
  // template<class F>
  // CUDA static F deinterpret(const value_type& v) {
  //   return F::make_bool(v);
  // }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{BInc} b \Leftrightarrow a \leq b \f$.
      \return The logical symbol `LEQ`. */
  CUDA static constexpr Sig sig_order() { return LEQ; }

  /** Converse nonimplication: we have a < b only when `a` is `false` and `b` is `true`. */
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

  CUDA NI static constexpr bool is_supported_fun(Sig sig) {
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
  CUDA NI static constexpr value_type fun(value_type x, value_type y) {
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