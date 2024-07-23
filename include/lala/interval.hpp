// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_INTERVAL_HPP
#define LALA_CORE_INTERVAL_HPP

#include "cartesian_product.hpp"
#include "universes/flat_universe.hpp"

namespace lala {

template <class U>
class Interval;

namespace impl {
  template <class LB, class UB>
  constexpr typename Interval<LB>::local_type make_itv(CartesianProduct<LB, UB> cp) {
    return typename Interval<LB>::local_type(project<0>(cp), project<1>(cp));
  }

  template <class U> constexpr const typename Interval<U>::LB& lb(const Interval<U>& itv) { return itv.lb(); }
  template <class L> constexpr const L& lb(const L& other) { return other; }
  template <class U> constexpr const typename Interval<U>::UB& ub(const Interval<U>& itv) { return itv.ub(); }
  template <class L> constexpr const L& ub(const L& other) { return other; }
}

/** An interval is a Cartesian product of a lower and upper bounds, themselves represented as lattices.
    One difference, is that the \f$ \top \f$ can be represented by multiple interval elements, whenever \f$ l > u \f$, therefore some operations are different than on the Cartesian product, e.g., \f$ [3..2] \equiv [4..1] \f$ in the interval lattice. */
template <class U>
class Interval {
public:
  using LB = U;
  using UB = typename LB::dual_type;
  using this_type = Interval<LB>;
  using local_type = Interval<typename LB::local_type>;
  using CP = CartesianProduct<LB, UB>;
  using value_type = typename CP::value_type;
  using memory_type = typename CP::memory_type;

  template <class A>
  friend class Interval;

  constexpr static const bool is_abstract_universe = true;
  constexpr static const bool sequential = CP::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = CP::preserve_bot;
  constexpr static const bool preserve_top = true;
  constexpr static const bool preserve_join = true;
  constexpr static const bool preserve_meet = false;
  constexpr static const bool injective_concretization = CP::injective_concretization;
  constexpr static const bool preserve_concrete_covers = CP::preserve_concrete_covers;
  constexpr static const bool complemented = false;
  constexpr static const char* name = "Interval";

private:
  using LB2 = typename LB::local_type;
  using UB2 = typename UB::local_type;
  CP cp;
  CUDA constexpr Interval(const CP& cp): cp(cp) {}
  CUDA constexpr local_type lb2() const { return local_type(lb(), dual<UB2>(lb())); }
  CUDA constexpr local_type ub2() const { return local_type(dual<LB2>(ub()), ub()); }
public:
  /** Initialize the interval to bottom using the default constructor of the bounds. */
  CUDA constexpr Interval() {}
  /** Given a value \f$ x \in U \f$ where \f$ U \f$ is the universe of discourse, we initialize a singleton interval \f$ [x..x] \f$. */
  CUDA constexpr Interval(const typename U::value_type& x): cp(x, x) {}
  CUDA constexpr Interval(const LB& lb, const UB& ub): cp(lb, ub) {}

  template<class A>
  CUDA constexpr Interval(const Interval<A>& other): cp(other.cp) {}

  template<class A>
  CUDA constexpr Interval(Interval<A>&& other): cp(std::move(other.cp)) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive. */
  template <class A>
  CUDA constexpr this_type& operator=(const Interval<A>& other) {
    cp = other.cp;
    return *this;
  }

  CUDA constexpr this_type& operator=(const this_type& other) {
    cp = other.cp;
    return *this;
  }

  /** Pre-interpreted formula `x == 0`. */
  CUDA constexpr static local_type eq_zero() { return local_type(LB::geq_k(LB::pre_universe::zero()), UB::leq_k(UB::pre_universe::zero())); }
  /** Pre-interpreted formula `x == 1`. */
  CUDA constexpr static local_type eq_one() { return local_type(LB::geq_k(LB::pre_universe::one()), UB::leq_k(UB::pre_universe::one())); }

  CUDA constexpr static local_type bot() { return Interval(CP::bot()); }
  CUDA constexpr static local_type top() { return Interval(CP::top()); }
  CUDA constexpr local::BInc is_top() const { return cp.is_top() || (!ub().is_bot() && lb() > dual<LB2>(ub())); }
  CUDA constexpr local::BDec is_bot() const { return cp.is_bot(); }
  CUDA constexpr const CP& as_product() const { return cp; }
  CUDA constexpr value_type value() const { return cp.value(); }

private:
  template<Sig sig, class A, class B>
  CUDA constexpr static local_type flat_fun(const Interval<A>& a, const Interval<B>& b) {
    return local_type(
      LB2::template fun<sig>(
        typename A::template flat_type<battery::local_memory>(a.lb()),
        typename B::template flat_type<battery::local_memory>(b.lb())),
      UB2::template fun<sig>(
        typename A::template flat_type<battery::local_memory>(a.ub()),
        typename B::template flat_type<battery::local_memory>(b.ub())));
  }

  template<Sig sig, class A>
  CUDA constexpr static local_type flat_fun(const Interval<A>& a) {
    return local_type(
      LB2::template fun<sig>(typename A::template flat_type<battery::local_memory>(a.lb())),
      UB2::template fun<sig>(typename A::template flat_type<battery::local_memory>(a.ub())));
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
          k.tell(local_type(LB::geq_k(LB::pre_universe::zero()), UB::leq_k(UB::pre_universe::one())));
          return true;
        }
      }
    }
    return CP::template interpret_tell<diagnose>(f, env, k.cp, diagnostics);
  }

  /** Support the same language than the Cartesian product, and more:
   *    * `x != k` is under-approximated by interpreting `x != k` in the lower bound.
   *    * `x == k` is interpreted by over-approximating `x == k` in both bounds and then verifying both bounds are the same.
   *    * `x in {[l..u]} is interpreted by under-approximating `x >= l` and `x <= u`. */
  template<bool diagnose = false, class F, class Env, class U2>
  CUDA NI static bool interpret_ask(const F& f, const Env& env, Interval<U2>& k, IDiagnostics& diagnostics) {
    local_type itv = local_type::bot();
    if(f.is_binary() && f.sig() == NEQ) {
      return LB::template interpret_ask<diagnose>(f, env, k.lb(), diagnostics);
    }
    else if(f.is_binary() && f.sig() == EQ) {
      CALL_WITH_ERROR_CONTEXT_WITH_MERGE(
        "When interpreting equality, the underlying bounds LB and UB failed to agree on the same value.",
        (LB::template interpret_tell<diagnose>(f, env, itv.lb(), diagnostics) &&
         UB::template interpret_tell<diagnose>(f, env, itv.ub(), diagnostics) &&
         itv.lb() == itv.ub()),
        (k.tell(itv)));
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
        (k.tell(itv))
      );
    }
    return CP::template interpret_ask<diagnose>(f, env, k.cp, diagnostics);
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

  /** You must use the lattice interface (tell methods) to modify the lower and upper bounds, if you use assignment you violate the PCCP model. */
  CUDA constexpr LB& lb() { return project<0>(cp); }
  CUDA constexpr UB& ub() { return project<1>(cp); }

  CUDA constexpr const LB& lb() const { return project<0>(cp); }
  CUDA constexpr const UB& ub() const { return project<1>(cp); }

  CUDA constexpr this_type& tell_top() {
    cp.tell_top();
    return *this;
  }

  template<class A, class M>
  CUDA constexpr this_type& tell_lb(const A& lb, BInc<M>& has_changed) {
    cp.template tell<0>(lb, has_changed);
    return *this;
  }

  template<class A, class M>
  CUDA constexpr this_type& tell_ub(const A& ub, BInc<M>& has_changed) {
    cp.template tell<1>(ub, has_changed);
    return *this;
  }

  template<class A>
  CUDA constexpr this_type& tell_lb(const A& lb) {
    cp.template tell<0>(lb);
    return *this;
  }

  template<class A>
  CUDA constexpr this_type& tell_ub(const A& ub) {
    cp.template tell<1>(ub);
    return *this;
  }

  template<class A, class M>
  CUDA constexpr this_type& tell(const Interval<A>& other, BInc<M>& has_changed) {
    cp.tell(other.cp, has_changed);
    return *this;
  }

  template<class A>
  CUDA constexpr this_type& tell(const Interval<A>& other) {
    cp.tell(other.cp);
    return *this;
  }

  CUDA constexpr this_type& dtell_bot() {
    cp.dtell_bot();
    return *this;
  }

  template<class A, class M>
  CUDA constexpr this_type& dtell_lb(const A& lb, BInc<M>& has_changed) {
    cp.template dtell<0>(lb, has_changed);
    return *this;
  }

  template<class A, class M>
  CUDA constexpr this_type& dtell_ub(const A& ub, BInc<M>& has_changed) {
    cp.template dtell<1>(ub, has_changed);
    return *this;
  }

  template<class A>
  CUDA constexpr this_type& dtell_lb(const A& lb) {
    cp.template dtell<0>(lb);
    return *this;
  }

  template<class A>
  CUDA constexpr this_type& dtell_ub(const A& ub) {
    cp.template dtell<1>(ub);
    return *this;
  }

  template<class A, class M>
  CUDA constexpr this_type& dtell(const Interval<A>& other, BInc<M>& has_changed) {
    cp.dtell(other.cp, has_changed);
    return *this;
  }

  template<class A>
  CUDA constexpr this_type& dtell(const Interval<A>& other) {
    cp.dtell(other.cp);
    return *this;
  }

  template <class A>
  CUDA constexpr bool extract(Interval<A>& ua) const {
    return cp.extract(ua.cp);
  }

  template<class Env>
  CUDA TFormula<typename Env::allocator_type> deinterpret(AVar x, const Env& env) const {
    using F = TFormula<typename Env::allocator_type>;
    if(lb().is_top() || ub().is_top() || lb().is_bot() || ub().is_bot()) {
      return cp.deinterpret(x, env);
    }
    F logical_lb = lb().template deinterpret<F>();
    F logical_ub = ub().template deinterpret<F>();
    logic_set<F> logical_set(1, env.get_allocator());
    logical_set[0] = battery::make_tuple(std::move(logical_lb), std::move(logical_ub));
    F set = F::make_set(std::move(logical_set));
    F var = F::make_avar(x);
    return F::make_binary(var, IN, std::move(set), UNTYPED, env.get_allocator());
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
    return logical_lb;
  }

  CUDA NI void print() const {
    printf("[");
    lb().print();
    printf("..");
    ub().print();
    printf("]");
  }

  CUDA NI constexpr static bool is_supported_fun(Sig sig) {
    switch(sig) {
      case ABS: return CP::is_supported_fun(NEG);
      case SUB: return CP::is_supported_fun(ADD) && CP::is_supported_fun(NEG);
      case NEG:
      case ADD:
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
      case MAX: return CP::is_supported_fun(sig);
      default: return false;
    }
  }

  /** The additive inverse is obtained by pairwise negation of the components.
   * Equivalent to `neg(reverse(x))`.
   * Note that the inverse of `bot` is `bot`, simply because `bot` has no mathematical inverse. */
  CUDA constexpr static local_type additive_inverse(const this_type& x) {
    static_assert(LB::is_supported_fun(NEG) && UB::is_supported_fun(NEG),
      "Negation of interval bounds are required to compute the additive inverse.");
    return flat_fun<NEG>(x);
  }

public:
  template<class L>
  CUDA constexpr static local_type reverse(const Interval<L>& x) {
    return local_type(dual<LB2>(x.ub()), dual<UB2>(x.lb()));
  }

  template<class L>
  CUDA constexpr static local_type neg(const Interval<L>& x) {
    return reverse(flat_fun<NEG>(x));
  }

  // This operation preserves top, i.e., \f$ abs(x) \in [\top] \f$ if \f$ x \in [\top] \f$, \f$ [\top] \f$ being the equivalence class of top elements.
  template<class L>
  CUDA constexpr static local_type abs(const Interval<L>& x) {
    switch(sign(x)) {
      case PP: return x;
      case NP: return local_type(   // [0..max(-lb, ub)]
        LB2::geq_k(LB2::pre_universe::zero()),
        meet(dual<UB2>(LB2::template fun<ABS>(typename L::template flat_type<battery::local_memory>(x.lb()))), x.ub()));
      case NN: return neg(x);
      case PN: return local_type(x.lb(), meet(UB2::leq_k(UB2::pre_universe::zero()), x.ub()));
    }
    assert(0); // all cases should be covered:
    return top();
  }

  template<Sig sig, class L>
  CUDA constexpr static local_type fun(const Interval<L>& x) {
    static_assert(sig == NEG || sig == ABS, "Unsupported unary function.");
    switch(sig) {
      case NEG: return neg(x);
      case ABS: return abs(x);
      default:
        assert(0); return x;
    }
  }

public:
  template<class L, class K>
  CUDA constexpr static local_type add(const Interval<L>& x, const Interval<K>& y) {
    return flat_fun<ADD>(x, y);
  }

  template<class L, class K>
  CUDA constexpr static local_type sub(const Interval<L>& x, const Interval<K>& y) {
    return add(x, neg(y));
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

  template<class A, class B>
  CUDA constexpr static local_type mul2(const Interval<A>& a, const Interval<B>& b) {
    return flat_fun<MUL>(a, b);
  }

  template<Sig sig, class R, class A, class B>
  CUDA constexpr static R flat_fun2(const A& a, const B& b) {
    return R::template fun<sig>(
      typename A::template flat_type<battery::local_memory>(a),
      typename B::template flat_type<battery::local_memory>(b));
  }

public:
  template<class L, class K>
  CUDA constexpr static local_type mul(const Interval<L>& l, const Interval<K>& k) {
    auto a = typename Interval<L>::local_type(l);
    auto b = typename Interval<K>::local_type(k);
    // Interval multiplication case, [al..au] * [bl..bu]
    switch(sign(a)) {
      case PP:
        switch(sign(b)) {
          case PP: return mul2(a, b);
          case NP: return mul2(a.ub2(), b);
          case NN: return mul2(reverse(a), b);
          case PN:
            if(b.as_product().is_top()) { return top(); }
            else { return mul2(a.lb2(), b); }
        }
      case NP:
        switch(sign(b)) {
          case PP: return mul2(a, b.ub2());
          // Note: we use meet for both bounds because UB is the dual of LB (e.g., if meet in LB is min, then meet in UB is max).
          case NP: return local_type(
              meet(flat_fun2<MUL, LB2>(a.lb(), b.ub()), flat_fun2<MUL, LB2>(a.ub(), b.lb())),
              meet(flat_fun2<MUL, UB2>(a.lb(), b.lb()), flat_fun2<MUL, UB2>(a.ub(), b.ub())));
          case NN: return mul2(reverse(a), b.lb2());
          case PN:
            if(b.as_product().is_top()) { return top(); }
            else { return eq_zero(); }
        }
      case NN:
        switch(sign(b)) {
          case PP: return mul2(a, reverse(b));
          case NP: return mul2(a.lb2(), reverse(b));
          case NN: return mul2(reverse(a), reverse(b));
          case PN:
            if(b.as_product().is_top()) { return top(); }
            else { return mul2(a.ub2(), reverse(b)); }
        }
      case PN:
        if(a.as_product().is_top()) { return top(); }
        else {
          switch(sign(b)) {
            case PP: return mul2(a, b.lb2());
            case NP: return eq_zero();
            case NN: return mul2(reverse(a), b.ub2());
            case PN:
              if(b.as_product().is_top()) { return top(); }
              else {
                return local_type(
                  join(flat_fun2<MUL, LB2>(a.lb(), b.lb()), flat_fun2<MUL, LB2>(a.ub(), b.ub())),
                  join(flat_fun2<MUL, UB2>(a.lb(), b.ub()), flat_fun2<MUL, UB2>(a.ub(), b.lb())));
              }
          }
        }
    }
    assert(0); // All cases should be covered.
    return top();
  }

private:
  /** For division, we cannot change the type of the bounds due to its importance in the underlying domain (usually PrimitiveUpset) when computing with zeroes. */
  template<Sig divsig, class AL, class AU, class BL, class BU>
  CUDA constexpr static local_type div2(const AL& al, const AU& au, const BL& bl, const BU& bu) {
    return local_type(LB2::template guarded_div<divsig>(al, bl),
                     UB2::template guarded_div<divsig>(au, bu));
  }

public:
  template<Sig divsig, class L, class K>
  CUDA constexpr static local_type div(const Interval<L>& l, const Interval<K>& k) {
    auto a = typename Interval<L>::local_type(l);
    auto b = typename Interval<K>::local_type(k);
    // Below, you can find cases where a or b are top, similar to multiplication above, but this has bugs and need to be studied in more depth.
    if(a.is_top() || b.is_top()) { return top(); }
    using UB_K = typename Interval<K>::UB::local_type;
    constexpr auto leq_zero = UB_K::leq_k(UB_K::pre_universe::zero());
    // Interval division, [al..au] / [bl..bu]
    switch(sign(b)) {
      case PP:
        if(b.ub() >= leq_zero) { return top(); }  // b is a singleton equal to zero.
        switch(sign(a)) {
          case PP: return div2<divsig>(a.lb(), a.ub(), b.ub(), b.lb());
          case NP: return div2<divsig>(a.lb(), a.ub(), b.lb(), b.lb());
          case NN: return div2<divsig>(a.lb(), a.ub(), b.lb(), b.ub());
          case PN:
            if(a.as_product().is_top()) { return top(); }
            else { return div2<divsig>(a.lb(), a.ub(), b.ub(), b.ub()); }
        }
      case NP:
        if constexpr(L::preserve_concrete_covers && K::preserve_concrete_covers) { // In the discrete case, division can be more precise.
          switch(sign(a)) {
            case PP: return div2<divsig>(a.ub(), a.ub(), b.lb(), b.ub());
            case NP: return local_type(
              meet(LB2::template guarded_div<divsig>(a.lb(), b.ub()), LB2::template guarded_div<divsig>(a.ub(), b.lb())),
              meet(UB2::template guarded_div<divsig>(a.lb(), b.lb()), UB2::template guarded_div<divsig>(a.ub(), b.ub())));
            case NN: return div2<divsig>(a.lb(), a.lb(), b.ub(), b.lb());
            case PN: return (a.as_product().is_top()) ? top() : eq_zero();
          }
        }
        else {
          return bot();
        }
      case NN:
        switch(sign(a)) {
          case PP: return div2<divsig>(a.ub(), a.lb(), b.ub(), b.lb());
          case NP: return div2<divsig>(a.ub(), a.lb(), b.ub(), b.ub());
          case NN: return div2<divsig>(a.ub(), a.lb(), b.lb(), b.ub());
          case PN:
            if(a.as_product().is_top()) { return top(); }
            else { return div2<divsig>(a.ub(), a.lb(), b.lb(), b.lb()); }
        }
      case PN:
        if(b.as_product().is_top()) { return top(); }
        if constexpr(L::preserve_concrete_covers && K::preserve_concrete_covers) {
          switch(sign(a)) {
            case PP: return div2<divsig>(a.lb(), a.lb(), b.ub(), b.lb());
            case NP: return eq_zero();
            case NN: return div2<divsig>(a.ub(), a.ub(), b.ub(), b.lb());
            case PN:
              if(a.as_product().is_top()) { return top(); }
              else {
                return local_type(
                  join(LB2::template guarded_div<divsig>(a.lb(), b.ub()), LB2::template guarded_div<divsig>(a.ub(), b.lb())),
                  join(UB2::template guarded_div<divsig>(a.lb(), b.lb()), UB2::template guarded_div<divsig>(a.ub(), b.ub())));
              }
          }
        }
        else {
          return top();  /* This is by duality with the case above, but I am unsure if it makes sense. */
        }
    }
    assert(0); // All cases should be covered.
    return top();
  }

  template<Sig modsig, class L, class K>
  CUDA constexpr static local_type mod(const Interval<L>& l, const Interval<K>& k) {
    auto a = typename Interval<L>::local_type(l);
    auto b = typename Interval<K>::local_type(k);
    if(a.is_top() || b.is_top()) { return top(); }
    if(a.lb() == dual<LB2>(a.ub()) && b.lb() == dual<LB2>(b.ub())) {
      return flat_fun<modsig>(a, b);
    }
    else {
      return bot();
    }
  }

  template<class L, class K>
  CUDA constexpr static local_type pow(const Interval<L>& l, const Interval<K>& k) {
    auto a = typename Interval<L>::local_type(l);
    auto b = typename Interval<K>::local_type(k);
    if(a.is_top() || b.is_top()) { return top(); }
    if(a.lb() == dual<LB2>(a.ub()) && b.lb() == dual<LB2>(b.ub())) {
      return flat_fun<POW>(a, b);
    }
    else {
      return bot();
    }
  }

  template<Sig sig, class L, class K>
  CUDA constexpr static local_type fun(const Interval<L>& x, const Interval<K>& y) {
    if constexpr(sig == ADD) { return add(x, y); }
    else if constexpr(sig == SUB) { return sub(x, y); }
    else if constexpr(sig == MUL) { return mul(x, y); }
    else if constexpr(is_division(sig)) { return div<sig>(x, y); }
    else if constexpr(is_modulo(sig)) { return mod<sig>(x, y); }
    else if constexpr(sig == POW) { return pow(x, y); }
    else if constexpr(sig == MIN || sig == MAX) { return CP::template fun<sig>(x.as_product(), y.as_product()); }
    else { static_assert(
      sig == ADD || sig == SUB || sig == MUL || sig == TDIV || sig == TMOD || sig == FDIV || sig == FMOD || sig == CDIV || sig == CMOD || sig == EDIV || sig == EMOD || sig == POW || sig == MIN || sig == MAX,
      "Unsupported binary function.");
    }
  }

  CUDA constexpr local_type width() const {
    static_assert(LB::is_totally_ordered && LB::is_arithmetic,
      "Width is only defined for totally ordered arithmetic intervals.");
    return sub(ub2(), lb2());
  }

  /** \return The median value of the interval, which is computed by `lb() + ((ub() - lb()) / 2)`. */
  CUDA constexpr local_type median() const {
    static_assert(LB::is_totally_ordered && LB::is_arithmetic,
      "Median function is only defined for totally ordered arithmetic intervals.");
    auto x = sub(ub2(), lb2());
    return
      add(lb2(), meet(div<FDIV>(x, local_type(2,2)), div<CDIV>(x, local_type(2,2))));
  }
};

// Lattice operations

template<class L, class K>
CUDA constexpr auto join(const Interval<L>& a, const Interval<K>& b)
{
  return impl::make_itv(join(a.as_product(), b.as_product()));
}

template<class L, class K>
CUDA constexpr auto meet(const Interval<L>& a, const Interval<K>& b)
{
  if(a.is_top()) { return b; }
  if(b.is_top()) { return a; }
  return impl::make_itv(meet(a.as_product(), b.as_product()));
}

template<class L, class K>
CUDA constexpr bool operator<=(const Interval<L>& a, const Interval<K>& b)
{
  return b.is_top() || a.as_product() <= b.as_product();
}

template<class L, class K>
CUDA constexpr bool operator<(const Interval<L>& a, const Interval<K>& b)
{
  return (b.is_top() && !a.is_top()) || a.as_product() < b.as_product();
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
  return a.as_product() == b.as_product() || (a.is_top() && b.is_top());
}

template<class L, class K>
CUDA constexpr bool operator!=(const Interval<L>& a, const Interval<K>& b)
{
  return a.as_product() != b.as_product() && !(a.is_top() && b.is_top());
}

template<class L>
std::ostream& operator<<(std::ostream &s, const Interval<L> &itv) {
  return s << "[" << itv.lb() << ".." << itv.ub() << "]";
}

} // namespace lala

#endif
