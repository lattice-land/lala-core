// Copyright 2021 Pierre Talbot

#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#include "cartesian_product.hpp"

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
  using CP = CartesianProduct<LB, UB>;
  using value_type = typename CP::value_type;
  using memory_type = typename CP::memory_type;

  template <class A>
  friend class Interval;

  template<class F>
  using iresult = IResult<this_type, F>;

  constexpr static const bool sequential = CP::sequential;
  constexpr static const char* name = "Interval";

  using local_type = Interval<typename LB::local_type>;

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

  /** Same as the Cartesian product interpretation but for equality:
   *    * Exact interpretation of equality is attempted by over-approximating both bounds and checking they are equal. */
  template<class F, class Env>
  CUDA static iresult<F> interpret(const F& f, const Env& env) {
    // In interval, we can handle the equality predicate exactly or by over-approximation.
    // Under-approximation does not make sense since it would either be exact or give an empty interval.
    // The equality is interpreted in both bounds by over-approximation, therefore the equal element must be in \f$ \gamma(lb) \cap \gamma(ub) \f$.
    // If an exact equality is asked, we verify the interpretations in LB and UB are equal.
    if(f.is_binary() && f.sig() == EQ) {
      if(f.is_under()) {
        return iresult<F>(IError<F>(true, name, "Equality cannot be interpreted by under-approximation (it would always give an empty interval).", f));
      }
      auto cp_res = CP::interpret(f.map_approx(OVER), env);
      if(cp_res.has_value()) {
        local_type itv(cp_res.value());
        if(f.is_exact() && itv.lb() != itv.ub()) {
          return iresult<F>(IError<F>(true, name, "Equality cannot be interpreted exactly because LB over-approximates the equality to a different value than UB.", f));
        }
        return std::move(iresult<F>(std::move(itv)).join_warnings(std::move(cp_res)));
      }
    }
    else if(f.is_binary() && f.sig() == NEQ) {
      if(f.is_over()) {
        return iresult<F>(IError<F>(true, name, "Disequality cannot be interpreted by over-approximation (it would always give the bottom interval [-oo..oo]).", f));
      }
      else if(f.is_under()) {
        auto lb = LB::interpret(f, env);
        if(lb.has_value()) {
          local_type itv(lb.value(), UB::bot());
          return std::move(iresult<F>(std::move(itv)).join_warnings(std::move(lb)));
        }
      }
    }
    else if(f.is_binary() && f.sig() == IN && f.seq(1).is(F::S)) {
      auto cp_res = CP::interpret(f.map_approx(OVER), env);
      if(cp_res.has_value()) {
        local_type itv(cp_res.value());
        return std::move(iresult<F>(std::move(itv)).join_warnings(std::move(cp_res)));
      }
    }
    // Forward to CP in case the formula `f` did not fit the cases above.
    auto cp_interpret = CP::interpret(f, env);
    if(cp_interpret.has_value()) {
      return std::move(iresult<F>(Interval(cp_interpret.value())).join_warnings(std::move(cp_interpret)));
    }
    return std::move(cp_interpret).template map_error<this_type>();
  }

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
    using allocator_t = typename Env::allocator_type;
    if(is_top()) {
      return TFormula<allocator_t>::make_false();
    }
    else if(is_bot()) {
      return TFormula<allocator_t>::make_true();
    }
    else {
      return cp.deinterpret(x, env);
    }
  }

  CUDA void print() const {
    printf("[");
    lb().print();
    printf("..");
    ub().print();
    printf("]");
  }

  CUDA constexpr static bool is_supported_fun(Sig sig) {
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
    return local_type(CP::template fun<NEG>(x.as_product()));
  }

private:
  // A faster version of reverse when we know x != bot.
  template<class L>
  CUDA constexpr static local_type reverse2(const Interval<L>& x) {
    return local_type(x.ub().value(), x.lb().value());
  }

public:
  template<class L>
  CUDA constexpr static local_type reverse(const Interval<L>& x) {
    return x.is_bot() ? bot() : reverse2(x);
  }

  template<Approx appx, class L>
  CUDA constexpr static local_type neg(const Interval<L>& x) {
    return local_type(
      dual<LB2>(UB2::template fun<NEG>(x.ub())),
      dual<UB2>(LB2::template fun<NEG>(x.lb())));
  }

  // This operation preserves top, i.e., \f$ abs(x) \in [\top] \f$ if \f$ x \in [\top] \f$, \f$ [\top] \f$ being the equivalence class of top elements.
  template<Approx appx, class L>
  CUDA constexpr static local_type abs(const Interval<L>& x) {
    switch(sign(x)) {
      case PP: return x;
      case NP: return local_type(LB2::template fun<ABS>(x.lb()), x.ub());
      case NN: return local_type(dual<LB2>(UB2::template fun<NEG>(x.ub())),
                                dual<UB2>(LB2::template fun<NEG>(x.lb())));
      case PN: return local_type(x.lb(), UB2::template fun<ABS>(x.ub()));
    }
    assert(0); // all cases should be covered:
    return top();
  }

  template<Approx appx, Sig sig, class L>
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
    return impl::make_itv(CP::template fun<ADD>(x.as_product(), y.as_product()));
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
    return local_type(CP::template fun<MUL>(a.as_product(), b.as_product()));
  }

  template<Sig divsig, class A, class B>
  CUDA constexpr static local_type div2(const A& a, const B& b) {
    return local_type(LB2::template guarded_div<divsig>(a.lb(), b.lb()),
                     UB2::template guarded_div<divsig>(a.ub(), b.ub()));
  }

public:

  /** By default, multiplication is over-approximating as it is not possible to exactly represent multiplication in general.
    Note that we do not rely on the commutativity property of multiplication. */
  template<class L, class K>
  CUDA constexpr static local_type mul(const Interval<L>& a, const Interval<K>& b) {
    // Interval multiplication case, [al..au] * [bl..bu]
    switch(sign(a)) {
      case PP:
        switch(sign(b)) {
          case PP: return mul2(a, b);
          case NP: return mul2(a.ub2(), b);
          case NN: return mul2(reverse2(a), b);
          case PN:
            if(b.as_product().is_top()) { return top(); }
            else { return mul2(a.lb2(), b); }
        }
      case NP:
        switch(sign(b)) {
          case PP: return mul2(a, b.ub2());
          // Note: we use meet for both bounds because UB is the dual of LB (e.g., if meet in LB is min, then meet in UB is max).
          case NP: return local_type(
              meet(LB2::template fun<MUL>(a.lb(), b.ub()), LB2::template fun<MUL>(a.ub(), b.lb())),
              meet(UB2::template fun<MUL>(a.lb(), b.lb()), UB2::template fun<MUL>(a.ub(), b.ub())));
          case NN: return mul2(reverse(a), b.lb2());
          case PN:
            if(b.as_product().is_top()) { return top(); }
            else { return eq_zero(); }
        }
      case NN:
        switch(sign(b)) {
          case PP: return mul2(a, reverse2(b));
          case NP: return mul2(a.lb2(), reverse(b));
          case NN: return mul2(reverse2(a), reverse2(b));
          case PN:
            if(b.as_product().is_top()) { return top(); }
            else { return mul2(a.ub2(), reverse2(b)); }
        }
      case PN:
        if(a.as_product().is_top()) { return top(); }
        else {
          switch(sign(b)) {
            case PP: return mul2(a, b.lb2());
            case NP: return eq_zero();
            case NN: return mul2(reverse2(a), b.ub2());
            case PN:
              if(b.as_product().is_top()) { return top(); }
              else {
                return local_type(
                  join(LB2::template fun<MUL>(a.lb(), b.lb()), LB2::template fun<MUL>(a.ub(), b.ub())),
                  join(UB2::template fun<MUL>(a.lb(), b.ub()), UB2::template fun<MUL>(a.ub(), b.lb())));
              }
          }
        }
    }
    assert(0); // All cases should be covered.
    return top();
  }

  template<Sig divsig, class L, class K>
  CUDA constexpr static local_type div(const Interval<L>& a, const Interval<K>& b) {
    constexpr auto leq_zero = Interval<K>::UB::leq_k(Interval<K>::UB::pre_universe::zero());
    // Interval division, [al..au] / [bl..bu]
    switch(sign(b)) {
      case PP:
        if(b.ub() >= leq_zero) { return top(); }  // b is a singleton equal to zero.
        switch(sign(a)) {
          case PP: return div2<divsig>(a, reverse2(b));
          case NP: return div2<divsig>(a, b.lb2());
          case NN: return div2<divsig>(a, b);
          case PN:
            if(a.as_product().is_top()) { return top(); }
            else { return div2<divsig>(a, b.ub2()); }
        }
      case NP:
        if(a.is_top()) { return top(); }
        else {
          if constexpr(L::preserve_concrete_covers && K::preserve_concrete_covers) { // In the discrete case, division can be more precise.
            switch(sign(a)) {
              case PP: return div2<divsig>(a.ub2(), reverse(b));
              case NP: return local_type(
                meet(LB2::template guarded_div<divsig>(a.lb(), b.lb()), LB2::template guarded_div<divsig>(a.ub(), b.ub())),
                meet(UB2::template guarded_div<divsig>(a.lb(), b.ub()), UB2::template guarded_div<divsig>(a.ub(), b.lb())));
              case NN: return div2<divsig>(a.lb2(), b);
              case PN: return (a.as_product().is_top()) ? top() : eq_zero();
            }
          }
          else {
            return bot();
          }
        }
      case NN:
        switch(sign(a)) {
          case PP: return div2<divsig>(reverse2(a), reverse2(b));
          case NP: return div2<divsig>(reverse(a), b.ub2());
          case NN: return div2<divsig>(reverse2(a), b);
          case PN:
            if(a.as_product().is_top()) { return top(); }
            else { return div2<divsig>(reverse2(a), b.lb2()); }
        }
      case PN:
        if(b.as_product().is_top()) { return top(); }
        if constexpr(L::preserve_concrete_covers && K::preserve_concrete_covers) {
          switch(sign(a)) {
            case PP: return div2<divsig>(a.lb2(), reverse2(b));
            case NP: return eq_zero();
            case NN: return div2<divsig>(a.ub2(), reverse2(b));
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
  CUDA constexpr static this_type mod(const Interval<L>& a, const Interval<K>& b) {
    if(a.is_top() || b.is_top()) { return top(); }
    if(a.lb() == dual<LB2>(a.ub()) && b.lb() == dual<LB2>(b.ub())) {
      auto l = LB2::template fun<modsig>(a.lb(), b.lb());
      auto u = UB2::template fun<modsig>(a.ub(), b.ub());
      return this_type(l, u);
    }
    else {
      return bot();
    }
  }

  template<class L, class K>
  CUDA constexpr static this_type pow(const Interval<L>& a, const Interval<K>& b) {
    if(a.is_top() || b.is_top()) { return top(); }
    if(a.lb() == dual<LB2>(a.ub()) && b.lb() == dual<LB2>(b.ub())) {
      auto l = LB2::template fun<POW>(a.lb(), b.lb());
      auto u = UB2::template fun<POW>(a.ub(), b.ub());
      return this_type(l, u);
    }
    else {
      return bot();
    }
  }

  template<Sig sig, class L, class K>
  CUDA constexpr static this_type fun(const Interval<L>& x, const Interval<K>& y) {
    if constexpr(sig == ADD) { return this_type::add(x, y); }
    else if constexpr(sig == SUB) { return this_type::sub(x, y); }
    else if constexpr(sig == MUL) { return this_type::mul(x, y); }
    else if constexpr(is_division(sig)) { return this_type::div<appx, sig>(x, y); }
    else if constexpr(is_modulo(sig)) { return this_type::mod<appx, sig>(x, y); }
    else if constexpr(sig == POW) { return this_type::pow(x, y); }
    else if constexpr(sig == MIN || sig == MAX) { return this_type(CP::template fun<sig>(x.as_product(), y.as_product())); }
    else { static_assert(
      sig == ADD || sig == SUB || sig == MUL || sig == TDIV || sig == TMOD || sig == FDIV || sig == FMOD || sig == CDIV || sig == CMOD || sig == EDIV || sig == EMOD || sig == POW || sig == MIN || sig == MAX,
      "Unsupported binary function.");
    }
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
