// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_INTERVAL_HPP
#define LALA_CORE_INTERVAL_HPP

#include "cartesian_product.hpp"
#include "universes/flat_universe.hpp"

namespace lala {

template <class U>
class Interval;

/** An interval is a Cartesian product of a lower and upper bounds, themselves represented as lattices.
    One difference, is that the \f$ \top \f$ can be represented by multiple interval elements, whenever \f$ l > u \f$, therefore some operations are different than on the Cartesian product, e.g., \f$ [3..2] \equiv [4..1] \f$ in the interval lattice. */
template <class U>
class Interval {
public:
  using LB = U;
  using UB = typename LB::dual_type;
  using this_type = Interval<LB>;
  using local_type = Interval<typename LB::local_type>;
  using value_type = battery::tuple<typename LB::value_type, typename UB::value_type>;
  using memory_type = typename LB::memory_type;

  template <class A>
  friend class Interval;

  constexpr static const bool is_abstract_universe = true;
  constexpr static const bool sequential = LB::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_top = LB::preserve_top && UB::preserve_top;
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_meet = true;
  constexpr static const bool preserve_join = false;
  constexpr static const bool injective_concretization = LB::injective_concretization && UB::injective_concretization;
  constexpr static const bool preserve_concrete_covers = LB::preserve_concrete_covers && UB::preserve_concrete_covers;
  constexpr static const bool complemented = false;
  constexpr static const bool is_arithmetic = LB::is_arithmetic && UB::is_arithmetic;
  constexpr static const char* name = "Interval";

private:
  using LB2 = typename LB::local_type;
  using UB2 = typename UB::local_type;
  LB l;
  UB u;

  CUDA constexpr local_type lb2() const { return local_type(lb(), dual_bound<UB2>(lb())); }
  CUDA constexpr local_type ub2() const { return local_type(dual_bound<LB2>(ub()), ub()); }

public:
  /** Initialize the interval to top using the default constructor of the bounds. */
  constexpr Interval() = default;
  constexpr Interval(const this_type&) = default;
  constexpr Interval(this_type&&) = default;

  /** Given a value \f$ x \in U \f$ where \f$ U \f$ is the universe of discourse, we initialize a singleton interval \f$ [x..x] \f$. */
  CUDA constexpr Interval(const typename U::value_type& x): l(x), u(x) {}
  CUDA constexpr Interval(const LB& lb, const UB& ub): l(lb), u(ub) {}

  template<class A>
  CUDA constexpr Interval(const Interval<A>& other): l(other.l), u(other.u) {}

  template<class A>
  CUDA constexpr Interval(Interval<A>&& other): l(std::move(other.l)), u(std::move(other.u)) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive. */
  template <class A>
  CUDA constexpr this_type& operator=(const Interval<A>& other) {
    l = other.l;
    u = other.u;
    return *this;
  }

  constexpr this_type& operator=(const this_type&) = default;
  constexpr this_type& operator=(this_type&&) = default;

  /** Pre-interpreted formula `x == 0`. */
  CUDA constexpr static local_type eq_zero() { return local_type(LB::geq_k(LB::pre_universe::zero()), UB::leq_k(UB::pre_universe::zero())); }
  /** Pre-interpreted formula `x == 1`. */
  CUDA constexpr static local_type eq_one() { return local_type(LB::geq_k(LB::pre_universe::one()), UB::leq_k(UB::pre_universe::one())); }

  CUDA constexpr static local_type bot() { return Interval(LB::bot(), UB::bot()); }
  CUDA constexpr static local_type top() { return Interval(LB::top(), UB::top()); }
  CUDA constexpr local::B is_bot() const {
    // The conversion to UB2 is possible because we have verified that lb() is different from bot and top.
    return l.is_bot() || u.is_bot() || (!is_top() && UB2(lb().value()) > ub());
  }
  CUDA constexpr local::B is_top() const { return l.is_top() && u.is_top(); }
  CUDA constexpr value_type value() const { return battery::make_tuple(l.value(), u.value()); }

private:
  template<class A, class B>
  CUDA constexpr void flat_fun(Sig fun, const Interval<A>& a, const Interval<B>& b) {
    lb().project(fun,
      typename A::template flat_type<battery::local_memory>(a.lb()),
      typename B::template flat_type<battery::local_memory>(b.lb()));
    ub().project(fun,
      typename A::template flat_type<battery::local_memory>(a.ub()),
      typename B::template flat_type<battery::local_memory>(b.ub()));
  }

  template<class A>
  CUDA constexpr void flat_fun(Sig fun, const Interval<A>& a) {
    lb().project(fun, typename A::template flat_type<battery::local_memory>(a.lb()));
    ub().project(fun, typename A::template flat_type<battery::local_memory>(a.ub()));
  }

public:
  /** Support the same language than the Cartesian product, and more:
   *    * `var x:B` when the underlying universe is arithmetic and preserve concrete covers.
   * Therefore, the element `k` is always in \f$ \gamma(lb) \cap \gamma(ub) \f$. */
  template<bool diagnose = false, class F, class Env, class U2>
  CUDA NI static bool interpret_tell(const F& f, const Env& env, Interval<U2>& k, IDiagnostics& diagnostics) {
    if constexpr(LB::preserve_concrete_covers && LB::is_arithmetic) {
      if(f.is(F::E)) {
        auto sort = f.sort();
        if(sort.has_value() && sort->is_bool()) {
          k.meet(local_type(LB::geq_k(LB::pre_universe::zero()), UB::leq_k(UB::pre_universe::one())));
          return true;
        }
      }
    }
    bool r;
    CALL_WITH_ERROR_CONTEXT(
      "No component of this interval can interpret this formula.",
        (r = LB::template interpret_tell<diagnose>(f, env, k.lb(), diagnostics),
         r |= UB::template interpret_tell<diagnose>(f, env, k.ub(), diagnostics),
         r));
  }

  /** Support the same language than the Cartesian product, and more:
   *    * `x != k` is under-approximated by interpreting `x != k` in the lower bound.
   *    * `x == k` is interpreted by over-approximating `x == k` in both bounds and then verifying both bounds are the same.
   *    * `x in {[l..u]} is interpreted by under-approximating `x >= l` and `x <= u`. */
  template<bool diagnose = false, class F, class Env, class U2>
  CUDA NI static bool interpret_ask(const F& f, const Env& env, Interval<U2>& k, IDiagnostics& diagnostics) {
    local_type itv = local_type::top();
    if(f.is_binary() && f.sig() == NEQ) {
      return LB::template interpret_ask<diagnose>(f, env, k.lb(), diagnostics);
    }
    else if(f.is_binary() && f.sig() == EQ) {
      CALL_WITH_ERROR_CONTEXT_WITH_MERGE(
        "When interpreting equality, the underlying bounds LB and UB failed to agree on the same value.",
        (LB::template interpret_tell<diagnose>(f, env, itv.lb(), diagnostics) &&
         UB::template interpret_tell<diagnose>(f, env, itv.ub(), diagnostics) &&
         itv.lb() == itv.ub()),
        (k.meet(itv)));
    }
    else if(f.is_binary() && f.sig() == IN && f.seq(0).is_variable()
     && f.seq(1).is(F::S) && f.seq(1).s().size() == 1)
    {
      const auto& lb = battery::get<0>(f.seq(1).s()[0]);
      const auto& ub = battery::get<1>(f.seq(1).s()[0]);
      if(lb == ub) {
        CALL_WITH_ERROR_CONTEXT(
          "Failed to interpret the decomposition of set membership `x in {[v..v]}` into equality `x == v`.",
          (interpret_ask<diagnose>(F::make_binary(f.seq(0), EQ, lb), env, k, diagnostics)));
      }
      CALL_WITH_ERROR_CONTEXT_WITH_MERGE(
        "Failed to interpret the decomposition of set membership `x in {[l..u]}` into `x >= l /\\ x <= u`.",
        (LB::template interpret_ask<diagnose>(F::make_binary(f.seq(0), geq_of_constant(lb), lb), env, itv.lb(), diagnostics) &&
         UB::template interpret_ask<diagnose>(F::make_binary(f.seq(0), leq_of_constant(ub), ub), env, itv.ub(), diagnostics)),
        (k.meet(itv))
      );
    }
    bool r;
    CALL_WITH_ERROR_CONTEXT(
      "No component of this interval can interpret this formula.",
        (r = LB::template interpret_ask<diagnose>(f, env, k.lb(), diagnostics),
         r |= UB::template interpret_ask<diagnose>(f, env, k.ub(), diagnostics),
         r));
  }

  template<IKind kind, bool diagnose = false, class F, class Env, class U2>
  CUDA NI static bool interpret(const F& f, const Env& env, Interval<U2>& k, IDiagnostics& diagnostics) {
    if constexpr(kind == IKind::ASK) {
      return interpret_ask<diagnose>(f, env, k, diagnostics);
    }
    else {
      return interpret_tell<diagnose>(f, env, k, diagnostics);
    }
  }

  /** You must use the lattice interface (join/meet methods) to modify the lower and upper bounds, if you use assignment you violate the PCCP model. */
  CUDA INLINE constexpr LB& lb() { return l; }
  CUDA INLINE constexpr UB& ub() { return u; }

  CUDA INLINE constexpr const LB& lb() const { return l; }
  CUDA INLINE constexpr const UB& ub() const { return u; }

  CUDA constexpr void join_top() {
    l.join_top();
    u.join_top();
  }

  template<class A>
  CUDA constexpr bool join_lb(const A& lb) {
    return l.join(lb);
  }

  template<class A>
  CUDA constexpr bool join_ub(const A& ub) {
    return u.join(ub);
  }

  template<class A>
  CUDA constexpr bool join(const Interval<A>& other) {
    bool r = l.join(other.l);
    r |= u.join(other.u);
    return r;
  }

  CUDA constexpr void meet_bot() {
    l.meet_bot();
    u.meet_bot();
  }

  template<class A>
  CUDA constexpr bool meet_lb(const A& lb) {
    return l.meet(lb);
  }

  template<class A>
  CUDA constexpr bool meet_ub(const A& ub) {
    return u.meet(ub);
  }

  template<class A>
  CUDA constexpr bool meet(const Interval<A>& other) {
    bool r = l.meet(other.l);
    r |= u.meet(other.u);
    return r;
  }

  template <class A>
  CUDA constexpr bool extract(Interval<A>& ua) const {
    return l.extract(ua.l) && u.extract(ua.u);
  }

  template<class Env, class Allocator = typename Env::allocator_type>
  CUDA TFormula<Allocator> deinterpret(AVar x, const Env& env, const Allocator& allocator = Allocator()) const {
    using F = TFormula<Allocator>;
    if(is_bot()) {
      return F::make_false();
    }
    if(is_top()) {
      return F::make_true();
    }
    if(lb().is_top()) {
      return ub().deinterpret(x, env, allocator);
    }
    else if(ub().is_top()) {
      return lb().deinterpret(x, env, allocator);
    }
    F logical_lb = lb().template deinterpret<F>();
    F logical_ub = ub().template deinterpret<F>();
    logic_set<F> logical_set(1, allocator);
    logical_set[0] = battery::make_tuple(std::move(logical_lb), std::move(logical_ub));
    F set = F::make_set(std::move(logical_set));
    F var = F::make_avar(x);
    return F::make_binary(var, IN, std::move(set), UNTYPED, allocator);
  }

  /** Deinterpret the current value to a logical constant.
   * The lower bound is deinterpreted, and it is up to the user to check that interval is a singleton.
   * A special case is made for real numbers where the both bounds are used, since the logical interpretation uses interval.
  */
  template<class F>
  CUDA NI F deinterpret() const {
    F logical_lb = lb().template deinterpret<F>();
    if(logical_lb.is(F::R)) {
      F logical_ub = ub().template deinterpret<F>();
      battery::get<1>(logical_lb.r()) = battery::get<0>(logical_ub.r());
    }
    else {
      assert(lb().value() == ub().value());
    }
    return logical_lb;
  }

  CUDA NI void print() const {
    printf("[");
    lb().print();
    printf("..");
    ub().print();
    printf("]");
  }

  /** The additive inverse is obtained by pairwise negation of the components.
   * Equivalent to `neg(reverse(x))`.
   * Note that the inverse of `bot` is `bot`, simply because `bot` has no mathematical inverse. */
  CUDA constexpr void additive_inverse(const this_type& x) {
    flat_fun(NEG, x);
  }

  template<class L>
  CUDA constexpr static local_type reverse(const Interval<L>& x) {
    return local_type(dual_bound<LB2>(x.ub()), dual_bound<UB2>(x.lb()));
  }

  CUDA constexpr void neg(const local_type& x) {
    flat_fun(NEG, reverse(x));
  }

  // This operation preserves top, i.e., \f$ abs(x) \in [\top] \f$ if \f$ x \in [\top] \f$, \f$ [\top] \f$ being the equivalence class of top elements.
  CUDA constexpr void abs(const local_type& x) {
    local_type nx{};
    nx.neg(x);
    nx.meet_lb(LB2::geq_k(LB2::pre_universe::zero()));
    meet_lb(fmeet(x.lb(), nx.lb()));
    meet_ub(fjoin(x.ub(), nx.ub()));
  }

  CUDA constexpr void project(Sig fun, const local_type& x) {
    switch(fun) {
      case NEG: neg(x); break;
      case ABS: abs(x); break;
    }
  }

  CUDA constexpr void add(const local_type& x, const local_type& y) {
    flat_fun(ADD, x, y);
  }

  CUDA constexpr void sub(const local_type& x, const local_type& y) {
    local_type ny{};
    ny.neg(y);
    add(x, ny);
  }

private:
  /** Characterization of the sign of the bounds (e.g., NP = lower bound is negative, upper bound is positive).
   * It can viewed as a lattice with the order NP < PP < PN and NP < NN < PN. */
  enum bounds_sign {
    PP, NN, NP, PN
  };

  /** The sign function is monotone w.r.t. the order of `Interval<A>` and `bounds_sign`. */
  template<class A>
  CUDA constexpr static bounds_sign sign(const Interval<A>& a) {
    if(a.lb() >= LB2::geq_k(LB2::pre_universe::zero())) {
      if(a.ub() > UB2::leq_k(UB2::pre_universe::zero())) {
        return PN;
      }
      else {
        return PP;
      }
    }
    else {
      if(a.ub() >= UB2::leq_k(UB2::pre_universe::zero())) {
        return NN;
      }
      else {
        return NP;
      }
    }
  }

  /** This is a generic implementation of interval piecewise monotone function, where the function is monotone for each pair of bounds.
   * Let `fun` be a monotone binary operation such as multiplication: [rl..ru] = [al..au] * [bl..bu]
     where:
       (1) rl = min(al*bl, al*bu, au*bl, au*bu)
       (2) ru = max(al*bl, al*bu, au*bl, au*bu)
     In the discrete case, we can precompute the products in `l` and `u`.
     Otherwise, due to rounding, al*bl can be rounded downwards or upwards, and therefore must be computed differently for (1) and (2).

    We do not check if a and b are bot. This is the responsibility of the caller depending if it could pose issues with the function `fun`.
  */
  CUDA constexpr void piecewise_monotone_fun(Sig fun, const local_type& a, const local_type& b) {
    using PLB = typename LB::pre_universe;
    using PUB = typename UB::pre_universe;
    using value_t = typename PUB::value_type;
    if(preserve_concrete_covers && !is_division(fun)) {
      value_t x1 = PUB::project(fun, a.lb().value(), b.lb().value());
      value_t x2 = PUB::project(fun, a.lb().value(), b.ub().value());
      value_t x3 = PUB::project(fun, a.ub().value(), b.lb().value());
      value_t x4 = PUB::project(fun, a.ub().value(), b.ub().value());
      meet_lb(LB(PLB::join(PLB::join(x1, x2), PLB::join(x3, x4))));
      meet_ub(UB(PUB::join(PUB::join(x1, x2), PUB::join(x3, x4))));
    }
    else {
      value_t x1 = PLB::project(fun, a.lb().value(), b.lb().value());
      value_t x2 = PLB::project(fun, a.lb().value(), b.ub().value());
      value_t x3 = PLB::project(fun, a.ub().value(), b.lb().value());
      value_t x4 = PLB::project(fun, a.ub().value(), b.ub().value());
      meet_lb(LB(PLB::join(PLB::join(x1, x2), PLB::join(x3, x4))));

      x1 = PUB::project(fun, a.lb().value(), b.lb().value());
      x2 = PUB::project(fun, a.lb().value(), b.ub().value());
      x3 = PUB::project(fun, a.ub().value(), b.lb().value());
      x4 = PUB::project(fun, a.ub().value(), b.ub().value());
      meet_ub(UB(PUB::join(PUB::join(x1, x2), PUB::join(x3, x4))));
    }
  }

public:
  CUDA constexpr void mul(const local_type& a, const local_type& b) {
    if(a.is_bot() || b.is_bot()) { meet_bot(); return; } // Perhaps not necessary?
    piecewise_monotone_fun(MUL, a, b);
  }

  // Interval division, [al..au] / [bl..bu]
  CUDA constexpr void div(Sig divfun, const local_type& a, const local_type& b) {
    constexpr auto zero = LB2::pre_universe::zero();
    using flat_type = LB2::local_flat_type;
    constexpr auto fzero = flat_type::eq_k(zero);
    if(a.is_bot() || b.is_bot() || (b.lb().value() == zero && b.ub().value() == zero)) { meet_bot(); return; }
    // Interval division, [rl..ru] = [al..au] / [bl..bu]
    if constexpr(preserve_concrete_covers) {
      if(b.lb().value() < 0 && b.ub().value() > 0) {
        meet_lb(LB(battery::min(a.lb().value(), -a.ub().value())));
        meet_ub(UB(battery::max(-a.lb().value(), a.ub().value())));
      }
      else {
        // Remove 0 from the bounds of b if any is equal to it.
        piecewise_monotone_fun(divfun, a,
          local_type((b.lb().value() == zero) ? LB2(1) : b.lb(),
                      (b.ub().value() == zero) ? UB2(-1) : b.ub()));
      }
    }
    else {
      flat_type al(a.lb());
      flat_type au(a.ub());
      flat_type bl(b.lb());
      flat_type bu(b.ub());

      // The case where 0 in [bl, bu].
      if(bl <= fzero && bu >= fzero) {
        if(bl == fzero && bu == fzero) { meet_bot(); return; }  // b is a singleton equal to zero.
        if(al == fzero && au == fzero) { meet_lb(LB2::geq_k(zero)); meet_ub(UB2::leq_k(zero)); return; } // 0 / b = 0 (b != 0)
        if(bl == fzero) { lb().project(divfun, al, bu); return; } // If bl is 0, then the upper bound is infinite.
        ub().project(divfun, al, bl);  // if bu is 0, then the lower bound is infinite.
        return;
      }
      else {
        piecewise_monotone_fun(divfun, a, b);
      }
    }
  }

  CUDA constexpr void mod(Sig modfun, const local_type& a, const local_type& b) {
    if(a.is_bot() || b.is_bot()) { meet_bot(); return; }
    if(a.lb().value() == a.ub().value() && b.lb().value() == b.ub().value()) {
      flat_fun(modfun, a, b);
    }
  }

  CUDA constexpr void pow(const local_type& a, const local_type& b) {
    if(a.is_bot() || b.is_bot()) { meet_bot(); return; }
    if(a.lb().value() == a.ub().value() && b.lb().value() == b.ub().value()) {
      flat_fun(POW, a, b);
    }
  }

  CUDA static constexpr bool is_trivial_fun(Sig fun) {
    return LB2::is_trivial_fun(fun) && UB2::is_trivial_fun(fun)
      && fun != MUL && !is_division(fun) && fun != ABS && fun != SUB && fun != POW && !is_modulo(fun);
  }

  CUDA constexpr void project(Sig fun, const local_type& x, const local_type& y) {
    if(LB2::is_order_preserving_fun(fun) && UB2::is_order_preserving_fun(fun)) {
      l.project(fun, x.lb(), y.lb());
      u.project(fun, x.ub(), y.ub());
    }
    else if constexpr(LB::is_arithmetic) {
      if (fun == SUB) { sub(x, y); }
      else if (fun == MUL) { mul(x, y); }
      else if (is_division(fun)) { div(fun, x, y); }
      else if (is_modulo(fun)) { mod(fun, x, y); }
      else if (fun == POW) { pow(x, y); }
    }
  }

  CUDA constexpr local_type width() const {
    static_assert(LB::is_totally_ordered && LB::is_arithmetic,
      "Width is only defined for totally ordered arithmetic intervals.");
    local_type width{};
    width.sub(ub2(), lb2());
    return width;
  }

  /** \return The median value of the interval, which is computed by `lb() + ((ub() - lb()) / 2)`. */
  CUDA constexpr local_type median() const {
    static_assert(LB::is_totally_ordered && LB::is_arithmetic,
      "Median function is only defined for totally ordered arithmetic intervals.");
    if(is_bot()) { return local_type::bot(); }
    if(lb().is_top() || ub().is_top()) { return local_type::top(); }
    auto l = lb().value();
    auto u = ub().value();
    typename LB::value_type two{2};
    return local_type(l + battery::fdiv((u - l), two), l + battery::cdiv((u - l), two));
  }
};

// Lattice operations

template<class L, class K>
CUDA constexpr Interval<typename L::local_type> fjoin(const Interval<L>& a, const Interval<K>& b)
{
  if(a.is_bot()) { return b; }
  if(b.is_bot()) { return a; }
  return Interval<typename L::local_type>(fjoin(a.lb(), b.lb()), fjoin(a.ub(), b.ub()));
}

template<class L, class K>
CUDA constexpr Interval<typename L::local_type> fmeet(const Interval<L>& a, const Interval<K>& b)
{
  return Interval<typename L::local_type>(fmeet(a.lb(), b.lb()), fmeet(a.ub(), b.ub()));
}

template<class L, class K>
CUDA constexpr bool operator<=(const Interval<L>& a, const Interval<K>& b)
{
  return a.is_bot() || (a.lb() <= b.lb() && a.ub() <= b.ub());
}

template<class L, class K>
CUDA constexpr bool operator<(const Interval<L>& a, const Interval<K>& b)
{
  return a <= b && a != b;
}

template<class L, class K>
CUDA constexpr bool operator>=(const Interval<L>& a, const Interval<K>& b)
{
  return b <= a;
}

template<class L, class K>
CUDA constexpr bool operator>(const Interval<L>& a, const Interval<K>& b)
{
  return b < a;
}

template<class L, class K>
CUDA constexpr bool operator==(const Interval<L>& a, const Interval<K>& b)
{
  return (a.is_bot() && b.is_bot()) || (a.lb() == b.lb() && a.ub() == b.ub());
}

template<class L>
CUDA constexpr bool operator==(const Interval<L>& a, typename L::value_type k)
{
  return a.lb().value() == k && a.ub().value() == k;
}

template<class L, class K>
CUDA constexpr bool operator!=(const Interval<L>& a, const Interval<K>& b)
{
  return !(a == b);
}

template<class L>
std::ostream& operator<<(std::ostream &s, const Interval<L> &itv) {
  return s << "[" << itv.lb() << ".." << itv.ub() << "]";
}

} // namespace lala

#endif
