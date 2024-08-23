// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_PRE_ZUB_HPP
#define LALA_CORE_PRE_ZUB_HPP

#include "../logic/logic.hpp"

namespace lala {

template<class VT>
struct PreZLB;

/** `PreZUB` is a pre-abstract universe \f$ \langle \{-\infty, \ldots, -2, -1, 0, 1, 2, \ldots, \infty\}, \leq \rangle \f$ totally ordered by the natural arithmetic comparison operator.
    It is used to represent constraints of the form \f$ x \leq k \f$ where \f$ k \f$ is an integer.
*/
template<class VT>
struct PreZUB {
  using this_type = PreZUB<VT>;
  using dual_type = PreZLB<VT>;
  using value_type = VT;
  using lower_bound_type = dual_type;
  using upper_bound_type = this_type;

  static_assert(std::is_integral_v<value_type> && std::is_signed_v<value_type>, "PreZUB only works over signed integer types.");

  constexpr static const bool is_totally_ordered = true;

  /** `true` if \f$ \gamma(\bot) = \{\} \f$. */
  constexpr static const bool preserve_bot = true;

  /** `true` if \f$ \gamma(\top) = U \f$. */
  constexpr static const bool preserve_top = true;

  /** `true` if \f$ \gamma(a \sqcup b) = \gamma(a) \cup \gamma(b) \f$ .*/
  constexpr static const bool preserve_join = true;

    /** `true` if \f$ \gamma(a \sqcap b) = \gamma(a) \cap \gamma(b) \f$ .*/
  constexpr static const bool preserve_meet = true;

  /** The concretization is injective when each abstract element maps to a distinct concrete element.
      This is important for the correctness of `prev` and `next` because we suppose \f$ \gamma(x) != \gamma(\mathit{next}(x)) \f$ when \f$ x \neq \bot \land x \neq \top \f$. */
  constexpr static const bool injective_concretization = true;

  /** `true` if concrete covers are preserved by the concretization function, i.e., \f$ \gamma(\mathit{next}(x)) \f$ is a cover of \f$ \gamma(x) \f$, and dually for \f$ \mathit{prev}(x) \f$.
   * \remark `preserve_concrete_covers` implies `injective_concretization`.
   */
  constexpr static const bool preserve_concrete_covers = true;

  /** `true` if this lattice aims to represents lower bounds of concrete sets. */
  constexpr static const bool is_lower_bound = false;
  constexpr static const bool is_upper_bound = true;

  constexpr static const char* name = "ZUB";

  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return 0; }
  CUDA constexpr static value_type one() { return 1; }

private:
  template<bool diagnose, bool is_tell, bool dualize, class F>
  CUDA NI static bool interpret(const F& f, value_type& k, IDiagnostics& diagnostics) {
    if(f.is(F::Z)) {
      auto z = f.z();
      if(z == bot() || z == top()) {
        RETURN_INTERPRETATION_ERROR("Constant of sort `Int` with the minimal or maximal representable value of the underlying integer type. We use those values to model negative and positive infinities. Example: Suppose we use a byte type, `x >= 256` is interpreted as `x >= INF` which is always false and thus is different from the intended constraint.");
      }
      k = z;
      return true;
    }
    else if(f.is(F::R)) {
      if constexpr(dualize) {
        if constexpr(is_tell) {
          k = battery::ru_cast<value_type>(battery::get<0>(f.r()));
        }
        else {
          k = battery::ru_cast<value_type>(battery::get<1>(f.r()));
        }
      }
      else {
        if constexpr(is_tell) {
          k = battery::rd_cast<value_type>(battery::get<1>(f.r()));
        }
        else {
          k = battery::rd_cast<value_type>(battery::get<0>(f.r()));
        }
      }
      return true;
    }
    else if(f.is(F::B)) {
      k = f.b() ? one() : zero();
      return true;
    }
    RETURN_INTERPRETATION_ERROR("Only constants of sorts `Int`, `Bool` and `Real` can be interpreted by an integer abstract universe.");
  }

public:
  /** Interpret a constant in the lattice of increasing integers according to the downset semantics.
      Overflows are not verified.
      Interpretations:
        * Formulas of kind `F::Z` are interpreted exactly: \f$ [\![ x:\mathbb{Z} \leq k:\mathbb{Z} ]\!] = k \f$.
        * Formulas of kind `F::R` are over-approximated: \f$ [\![ x:\mathbb{Z} \leq [l..u]:\mathbb{R} ]\!] = \lfloor u \rfloor \f$.
      Examples:
        * \f$ [\![x <= [3.5..3.5]:R ]\!] = 3 \f$: there is no integer greater than 3 satisfying this constraint.
        * \f$ [\![x <= [2.9..3.1]:R ]\!] = 3 \f$.
  */
  template<bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_tell(const F& f, value_type& tell, IDiagnostics& diagnostics) {
    return interpret<diagnose, true, dualize>(f, tell, diagnostics);
  }

  /** Similar to `interpret_tell` but the formula is under-approximated, in particular: \f$ [\![ x:\mathbb{Z} \leq [l..u]:\mathbb{R} ]\!] = \lfloor u \rfloor \f$.
      Examples:
        * \f$ [\![x <= [3.5..3.5]:R ]\!] = 3 \f$.
        * \f$ [\![x <= [2.9..3.1]:R ]\!] = 2 \f$: the constraint is entailed only when x is less or equal to 2.9. */
  template<bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_ask(const F& f, value_type& ask, IDiagnostics& diagnostics) {
    return interpret<diagnose, false, dualize>(f, ask, diagnostics);
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
      Variables of type `Int` are interpreted exactly (\f$ \mathbb{Z} = \gamma(\top) \f$).
      Note that we assume there is no overflow, that might be taken into account the future. */
  template<bool diagnose, class F, bool dualize = false>
  CUDA NI static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
    assert(f.is(F::E));
    const auto& sort = battery::get<1>(f.exists());
    if(sort.is_int()) {
      k = dualize ? bot() : top();
      return true;
    }
    else {
      const auto& vname = battery::get<0>(f.exists());
      RETURN_INTERPRETATION_ERROR("The type of `" + vname + "` can only be `Int`.")
    }
  }

  /** Given an Integer value, create a logical constant representing that value.
   * Note that the lattice order has no influence here.
   * \pre `v != bot()` and `v != top()`.
  */
  template<class F>
  CUDA static F deinterpret(const value_type& v) {
    return F::make_z(v);
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      We have \f$ a \leq_\mathit{ZUB} b \Leftrightarrow a \leq b \f$.
      \return The logical symbol `LEQ`. */
  CUDA static constexpr Sig sig_order() { return LEQ; }

  /** The logical predicate symbol corresponding to the strict order of this pre-universe.
      We have \f$ a <_\mathit{ZUB} b \Leftrightarrow a < b \f$.
      \return The logical symbol `LT`. */
  CUDA static constexpr Sig sig_strict_order() { return LT; }

private:
  CUDA INLINE static constexpr value_type inf() {
    return battery::limits<value_type>::inf();
  }

  CUDA INLINE static constexpr value_type neg_inf() {
    return battery::limits<value_type>::neg_inf();
  }

public:

  /** \f$ \bot \f$ is represented by the minimal representable value of the underlying value type. */
  CUDA INLINE static constexpr value_type bot() {
    return neg_inf();
  }

  /** \f$ \top \f$ is represented by the maximal representable value of the underlying value type. */
  CUDA INLINE static constexpr value_type top() {
    return inf();
  }

  /** \return \f$ x \sqcup y \f$ defined as \f$ \mathit{max}(x, y) \f$. */
  CUDA INLINE static constexpr value_type join(value_type x, value_type y) { return battery::max(x, y); }

  /** \return \f$ x \sqcap y \f$ defined as \f$ \mathit{min}(x, y) \f$. */
  CUDA INLINE static constexpr value_type meet(value_type x, value_type y) { return battery::min(x, y); }

  /** \return \f$ \mathit{true} \f$ if \f$ x \leq_\mathit{ZUB} y \f$ where the order \f$ \leq_\mathit{ZUB} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA INLINE static constexpr bool order(value_type x, value_type y) { return x <= y; }

  /** \return \f$ \mathit{true} \f$ if \f$ x <_\mathit{ZUB} y \f$ where the order \f$ <_\mathit{ZUB} \f$ is the natural arithmetic ordering, otherwise returns \f$ \mathit{false} \f$. */
  CUDA INLINE static constexpr bool strict_order(value_type x, value_type y) { return x < y; }

  /** From a lattice perspective, `next: ZUB -> ZUB` returns an element `next(x)` such that `x` is covered by `next(x)`.
      \param x The element covering the returned element.
      \return The next value of `x` in the discrete increasing chain `bot, ..., -2, -1, 0, 1, ..., top`.
      \remark The next value of `bot` is `bot` and the next value of `top` is `top`.
      \remark There is no element \f$ x \neq \top \f$ such that \f$ \mathit{next}(x) = \top \f$, but it can occur in case of overflow which is not checked. */
  CUDA static constexpr value_type next(value_type x) {
    return x + has_not_inf(x);
  }

  /** From a lattice perspective, `prev: ZUB -> ZUB` returns an element `prev(x)` such that `x` covers `prev(x)`.
      \param x The element covered by the returned element.
      \return The previous value of `x` in the discrete increasing chain `bot, ..., -2, -1, 0, 1, ..., top`.
      \remark The previous value of `bot` is `bot` and the previous value of `top` is `top`.
      \remark There is no element \f$ x \neq \bot \f$ such that \f$ \mathit{prev}(x) = \bot \f$, but it can occur in case of overflow which is not checked. */
  CUDA static constexpr value_type prev(value_type x)
  {
    return x - has_not_inf(x);
  }

private:
  CUDA INLINE static constexpr bool has_not_inf(value_type x) {
    return x != inf() && x != neg_inf();
  }

  CUDA INLINE static constexpr bool has_inf(value_type x) {
    return x == inf() || x == neg_inf();
  }

  CUDA INLINE static constexpr int sign(value_type x) {
    return battery::signum(x) + 1;
  }

  // Little bitwise trick to know if one of x or y is -inf or inf.
  CUDA INLINE static constexpr bool has_inf(value_type x, value_type y) {
    return has_inf(x) || has_inf(y);
  }

  template<bool dualize>
  CUDA static constexpr value_type dtop() {
    return dualize ? bot() : top();
  }

public:
  template <bool dualize>
  CUDA static constexpr value_type dproject(Sig fun, value_type x) {
    switch(fun) {
      case NEG: return has_inf(x) ? x : -x;
      default: return dtop<dualize>();
    }
  }

  /** `project: value_type -> value_type` is an abstract function on `ZUB` over-approximating the function denoted by `fun` on the concrete domain.
   * \tparam fun The signature of the function to over-approximate (only `NEG`).
   * \param x The argument of the function, which is a constant value in the underlying universe of discourse, or top().
   */
  CUDA static constexpr value_type project(Sig fun, value_type x) {
    return dproject<false>(fun, x);
  }

private:
  /** Look-up table for multiplication (division) between x * y (x / y) in presence of infinities.
   */
  static constexpr value_type infs[3][3] = {
  //          y < 0, y == 0, y > 0
  /* x < 0 */ {inf(), 0, neg_inf()},
  /* x = 0 */ {0, 0, 0},
  /* x > 0 */ {neg_inf(), 0, inf()}
  };

public:
  /** The projection of a function `fun` on two arguments `x` and `y` is simply the standard application of the operation, with specific cases for infinities.
   * When the function `fun` is not supported, we return `top()`.
   *
   * We deal with infinities in the "standard" way. The rules below are tried in order (so we apply inf - X even if X is inf or -inf).
   * For addition: inf + X = inf, -inf + X = -inf, X + inf = inf, X + -inf = -inf.
   * For subtraction: inf - X = inf, -inf - X = -inf, X - inf = -inf, X - -inf = inf.
   * For multiplication and division, we look at the sign (see the table `infs`).
   * For division, we expect y != 0 (but it returns 0 if it happens).
   * For division, when dividing by infinity, it actually depends on whether we are computing a lower or upper bound and the sign of infinity; not sure of the exact rules so we overapproximate to top.
   * For min and max, it is trivial.
   * For modulus and ipow, we expect x and y to be non-infinities.
   */
  template <bool dualize>
  CUDA static constexpr value_type dproject(Sig fun, value_type x, value_type y) {
    switch(fun) {
      case ADD: return has_inf(x, y) ? (has_inf(x) ? x : y) : x + y;
      case SUB: return has_inf(x, y) ? (has_inf(x) ? x : -y) : x - y;
      case MUL: return has_inf(x, y) ? infs[sign(x)][sign(y)] : x * y;
      // Truncated division and modulus, by default in C++.
      case TDIV: return has_inf(x, y) ? (has_inf(x) ? infs[sign(x)][sign(y)] : dtop<dualize>()) : x / y;
      case TMOD: return x % y;
      // Floor division and modulus, see (Leijen D. (2003). Division and Modulus for Computer Scientists).
      case FDIV: return has_inf(x, y) ? (has_inf(x) ? infs[sign(x)][sign(y)] : dtop<dualize>()) : battery::fdiv(x, y);
      case FMOD: return battery::fmod(x, y);
      // Ceil division and modulus.
      case CDIV: return has_inf(x, y) ? (has_inf(x) ? infs[sign(x)][sign(y)] : dtop<dualize>()) : battery::cdiv(x, y);
      case CMOD: return battery::cmod(x, y);
      // Euclidean division and modulus, see (Leijen D. (2003). Division and Modulus for Computer Scientists).
      case EDIV: return has_inf(x, y) ? (has_inf(x) ? infs[sign(x)][sign(y)] : dtop<dualize>()) : battery::ediv(x, y);
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
      default: return dtop<dualize>();
    }
  }

  CUDA static constexpr value_type project(Sig fun, value_type x, value_type y) {
    return dproject<false>(fun, x, y);
  }
};

} // namespace lala

#endif
