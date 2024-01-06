// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_PRE_ZINC_HPP
#define LALA_CORE_PRE_ZINC_HPP

#include "../logic/logic.hpp"

namespace lala {

template<class VT>
struct PreZDec;

/** `PreZInc` is a pre-abstract universe \f$ \langle \{-\infty, \ldots, -2, -1, 0, 1, 2, \ldots, \infty\}, \leq \rangle \f$ totally ordered by the natural arithmetic comparison operator.
    It is used to represent constraints of the form \f$ x \geq k \f$ where \f$ k \f$ is an integer.
*/
template<class VT>
struct PreZInc {
  using this_type = PreZInc<VT>;
  using dual_type = PreZDec<VT>;
  using value_type = VT;
  using increasing_type = this_type;

  static_assert(std::is_integral_v<value_type>, "PreZInc only works over integer types.");

  constexpr static const bool is_totally_ordered = true;

  /** `true` if \f$ \gamma(\bot) = \bot^\flat \f$. */
  constexpr static const bool preserve_bot = true;

  /** `true` if \f$ \gamma(\top) = \top^\flat \f$. */
  constexpr static const bool preserve_top = true;

  /** `true` if \f$ \gamma(a \sqcup b) = \gamma(a) \cap \gamma(b) \f$ .*/
  constexpr static const bool preserve_join = true;

    /** `true` if \f$ \gamma(a \sqcap b) = \gamma(a) \cup \gamma(b) \f$ .*/
  constexpr static const bool preserve_meet = true;

  /** The concretization is injective when each abstract element maps to a distinct concrete element.
      This is important for the correctness of `prev` and `next` because we suppose \f$ \gamma(x) != \gamma(\mathit{next}(x)) \f$ when \f$ x \neq \bot \land x \neq \top \f$. */
  constexpr static const bool injective_concretization = true;

  /** `true` if concrete covers are preserved by the concretization function, i.e., \f$ \gamma(\mathit{next}(x)) \f$ is a cover of \f$ \gamma(x) \f$, and dually for \f$ \mathit{prev}(x) \f$.
   * \remark `preserve_concrete_covers` implies `injective_concretization`.
   */
  constexpr static const bool preserve_concrete_covers = true;

  /** `true` if for all element \f$ x \in A \f$, there exists a unique element \f$ \lnot x \in A \f$ such that \f$ x \sqcup \lnot x = \top \f$ and \f$ x \sqcap \lnot x = \bot \f$. */
  constexpr static const bool complemented = false;

  /** `true` if the natural order of the universe of discourse coincides with the lattice order of this pre-universe, `false` if it is reversed. */
  constexpr static const bool increasing = true;

  constexpr static const char* name = "ZInc";

  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return 0; }
  CUDA constexpr static value_type one() { return 1; }

private:
  template<bool diagnose, bool is_tell, class F>
  CUDA NI static bool interpret(const F& f, value_type& k, IDiagnostics& diagnostics) {
    if(f.is(F::Z)) {
      auto z = f.z();
      if(z == bot() || z == top()) {
        RETURN_INTERPRETATION_ERROR("Constant of sort `Int` with the minimal or maximal representable value of the underlying integer type. We use those values to model negative and positive infinities. Example: Suppose we use a byte type, `x >= 256` is interpreted as `x >= INF` which is always false and thus is different from the intended constraint.");
      }
      k = z; // Truncation bug? When using ZInc, k is int, but z is logic_int (int64_t) -- See line 81 of primitive_upset.hpp
      return true;
    }
    else if(f.is(F::R)) {
      if constexpr(is_tell) {
        k = battery::ru_cast<value_type>(battery::get<0>(f.r()));
      }
      else {
        k = battery::ru_cast<value_type>(battery::get<1>(f.r()));
      }
      return true;
    }
    else if(f.is(F::B)) {
      k = f.b() ? one() : zero();
      return true;
    }
    RETURN_INTERPRETATION_ERROR("Only constant of sorts `Int` and `Real` can be interpreted by an integer abstract universe.");
  }

public:
  /** Interpret a constant in the lattice of increasing integers according to the upset semantics (see universe.hpp for explanation).
      Overflows are not verified (issue #1).
      Interpretations:
        * Formulas of kind `F::Z` are interpreted exactly: \f$ [\![ x:\mathbb{Z} \geq k:\mathbb{Z} ]\!] = k \f$.
        * Formulas of kind `F::R` are over-approximated: \f$ [\![ x:\mathbb{Z} \geq [l..u]:\mathbb{R} ]\!] = \lceil l \rceil \f$.
          Note that all elements in \f$ [l..\lceil l \rceil[\f$ do not belong to \f$ \mathbb{Z} \f$, so they can be safely ignored.
      Examples:
        * \f$ [\![x >= [2.5..2.5]:R ]\!] = 3 \f$.
        * \f$ [\![x >= [2.9..3.1]:R ]\!] = 3 \f$.
  */
  template<bool diagnose, class F>
  CUDA static bool interpret_tell(const F& f, value_type& tell, IDiagnostics& diagnostics) {
    return interpret<diagnose, true>(f, tell, diagnostics);
  }

  /** Similar to `interpret_tell` but the formula is under-approximated, in particular: \f$ [\![ x:\mathbb{Z} \geq [l..u]:\mathbb{R} ]\!] = \lceil u \rceil \f$. */
  template<bool diagnose, class F>
  CUDA static bool interpret_ask(const F& f, value_type& ask, IDiagnostics& diagnostics) {
    return interpret<diagnose, false>(f, ask, diagnostics);
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
      Variables of type `Int` are interpreted exactly (\f$ \mathbb{Z} = \gamma(\bot) \f$).
      Note that we assume there is no overflow, that might be taken into account the future. */
  template<bool diagnose, class F>
  CUDA NI static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
    assert(f.is(F::E));
    const auto& sort = battery::get<1>(f.exists());
    if(sort.is_int()) {
      k = bot();
      return true;
    }
    else {
      const auto& vname = battery::get<0>(f.exists());
      RETURN_INTERPRETATION_ERROR("The type of `" + vname + "` can only be `Int`.")
    }
  }

  /** Given an Integer value, create a logical constant representing that value.
   * Note that the lattice order has no influence here.
   * `\precondition` `v != bot()` and `v != top()`.
  */
  template<class F>
  CUDA static F deinterpret(const value_type& v) {
    return F::make_z(v);
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{ZInc} b \Leftrightarrow a \leq b \f$.
      \return The logical symbol `LEQ`. */
  CUDA static constexpr Sig sig_order() { return LEQ; }

  /** The logical predicate symbol corresponding to the strict order of this pre-universe.
      We have \f$ a <_\mathit{ZInc} b \Leftrightarrow a < b \f$.
      \return The logical symbol `LT`. */
  CUDA static constexpr Sig sig_strict_order() { return LT; }

  /** \f$ \bot \f$ is represented by the minimal representable value of the underlying value type. */
  CUDA static constexpr value_type bot() {
    return battery::limits<value_type>::bot();
  }

  /** \f$ \top \f$ is represented by the maximal representable value of the underlying value type. */
  CUDA static constexpr value_type top() {
    return battery::limits<value_type>::top();
  }

  /** \return \f$ x \sqcup y \f$ defined as \f$ \mathit{max}(x, y) \f$. */
  CUDA static constexpr value_type join(value_type x, value_type y) { return battery::max(x, y); }

  /** \return \f$ x \sqcap y \f$ defined as \f$ \mathit{min}(x, y) \f$. */
  CUDA static constexpr value_type meet(value_type x, value_type y) { return battery::min(x, y); }

  /** \return \f$ \mathit{true} \f$ if \f$ x \leq_\mathit{ZInc} y \f$ where the order \f$ \leq_\mathit{ZInc} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool order(value_type x, value_type y) { return x <= y; }

  /** \return \f$ \mathit{true} \f$ if \f$ x <_\mathit{ZInc} y \f$ where the order \f$ <_\mathit{ZInc} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return x < y; }

  /** From a lattice perspective, `next: ZInc -> ZInc` returns an element `next(x)` such that `x` is covered by `next(x)`.
      \param x The element covering the returned element.
      \return The next value of `x` in the discrete increasing chain `bot, ..., -2, -1, 0, 1, ..., top`.
      \remark The next value of `bot` is `bot` and the next value of `top` is `top`.
      \remark There is no element \f$ x \neq \top \f$ such that \f$ \mathit{next}(x) = \top \f$, but it can occur in case of overflow which is not checked. */
  CUDA static constexpr value_type next(value_type x) {
    return x + (x != top() && x != bot());
  }

  /** From a lattice perspective, `prev: ZInc -> ZInc` returns an element `prev(x)` such that `x` covers `prev(x)`.
      \param x The element covered by the returned element.
      \return The previous value of `x` in the discrete increasing chain `bot, ..., -2, -1, 0, 1, ..., top`.
      \remark The previous value of `bot` is `bot` and the previous value of `top` is `top`.
      \remark There is no element \f$ x \neq \bot \f$ such that \f$ \mathit{prev}(x) = \bot \f$, but it can occur in case of overflow which is not checked. */
  CUDA static constexpr value_type prev(value_type x)
  {
    return x - (x != top() && x != bot());
  }

  CUDA NI static constexpr bool is_supported_fun(Sig sig) {
    switch(sig) {
      case NEG:
      case ABS:
      case ADD:
      case SUB:
      case MUL:
      case TDIV:
      case TMOD:
      case FDIV:
      case FMOD:
      case CDIV:
      case CMOD:
      case EDIV:
      case EMOD:
      case POW:
      case MIN:
      case MAX:
      case EQ:
      case NEQ:
      case LEQ:
      case GEQ:
      case LT:
      case GT: return true;
      default: return false;
    }
  }

  /** `fun: value_type -> ZInc` is an abstract function on `ZInc` over-approximating the function denoted by `sig` on the concrete domain.
   * \tparam sig The signature of the function to over-approximate, can be either `NEG` or `ABS`.
   * \param x The argument of the function, which is a constant value in the underlying universe of discourse.
   * \note Since `x` is a constant, we do not check for equality with `bot()` or `top()`.
   */
  template<Sig sig>
  CUDA static constexpr value_type fun(value_type x) {
    static_assert(sig == NEG || sig == ABS, "Unsupported unary function.");
    switch(sig) {
      case NEG: return -x;
      case ABS: return abs(x);
      default: assert(0); return x;
    }
  }

  /** `fun: value_type X value_type -> ZInc` is similar to its unary version but with an arity of 2. */
  template<Sig sig>
  CUDA NI static constexpr value_type fun(value_type x, value_type y) {
    static_assert(
      sig == ADD || sig == SUB || sig == MUL || sig == TDIV || sig == TMOD || sig == FDIV || sig == FMOD || sig == CDIV || sig == CMOD || sig == EDIV || sig == EMOD || sig == POW || sig == MIN || sig == MAX || sig == EQ || sig == NEQ || sig == LEQ || sig == GEQ || sig == LT || sig == GT,
      "Unsupported binary function.");
    switch(sig) {
      case ADD: return x + y;
      case SUB: return x - y;
      case MUL: return x * y;
      // Truncated division and modulus, by default in C++.
      case TDIV: return x / y;
      case TMOD: return x % y;
      // Floor division and modulus, see (Leijen D. (2003). Division and Modulus for Computer Scientists).
      case FDIV: return battery::fdiv(x, y);
      case FMOD: return battery::fmod(x, y);
      // Ceil division and modulus.
      case CDIV: return battery::cdiv(x, y);
      case CMOD: return battery::cmod(x, y);
      // Euclidean division and modulus, see (Leijen D. (2003). Division and Modulus for Computer Scientists).
      case EDIV: return battery::ediv(x, y);
      case EMOD: return battery::emod(x, y);
      case POW: return battery::ipow(x, y);
      case MIN: return battery::min(x, y);
      case MAX: return battery::max(x, y);
      case EQ: return x == y;
      case NEQ: return x != y;
      case LEQ: return x <= y;
      case GEQ: return x >= y;
      case LT: return x < y;
      case GT: return x >= y;
      default: assert(0); return x;
    }
  }
};

} // namespace lala

#endif
