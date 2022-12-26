// Copyright 2021 Pierre Talbot

#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#include "cartesian_product.hpp"

namespace lala {

template <class U>
class Interval;

namespace impl {
  template <class LB, class UB>
  Interval<LB> make_itv(CartesianProduct<LB, UB> cp) {
    return Interval<LB>(project<0>(cp), project<1>(cp));
  }

  template <class U> const typename Interval<U>::LB& lb(const Interval<U>& itv) { return itv.lb(); }
  template <class L> const L& lb(const L& other) { return other; }
  template <class U> const typename Interval<U>::UB& ub(const Interval<U>& itv) { return itv.ub(); }
  template <class L> const L& ub(const L& other) { return other; }
}

/** An interval is a Cartesian product of a lower and upper bounds, themselves represented as lattices.
    One difference, is that the \f$ \top \f$ can be represented by multiple interval elements, whenever \f$ l > u \f$, therefore some operations are different than on the Cartesian product, e.g., \f$ [3..2] \equiv [4..1] \f$ in the interval lattice. */
template <class U>
class Interval {
public:
  using LB = U;
  using UB = typename LB::reverse_type;
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

private:
  CP cp;
  CUDA Interval(const CP& cp): cp(cp) {}
  CUDA this_type lb2() const { return Interval(lb(), dual<UB>(lb())); }
  CUDA this_type ub2() const { return Interval(dual<LB>(ub()), ub()); }
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
  CUDA std::enable_if_t<sequential, this_type&> operator=(const Interval<A>& other) {
    cp = other.cp;
    return *this;
  }

  CUDA std::enable_if_t<sequential, this_type&> operator=(const this_type& other) {
    cp = other.cp;
    return *this;
  }

  inline static const this_type zero = this_type(LB::zero, UB::zero);
  inline static const this_type one = this_type(LB::one, UB::one);

  CUDA static this_type bot() { return Interval(CP::bot()); }
  CUDA static this_type top() { return Interval(CP::top()); }
  CUDA local::BInc is_top() const { return cp.is_top() || (!ub().is_bot() && lb() > dual<LB>(ub())); }
  CUDA local::BDec is_bot() const { return cp.is_bot(); }
  CUDA const CP& as_product() const { return cp; }
  CUDA value_type value() const { return cp.value(); }

  /** Same as the Cartesian product interpretation but for equality:
   *    * Exact interpretation of equality is attempted by over-approximating both bounds and checking they are equal. */
  template<class F, class Env>
  CUDA static iresult<F> interpret(const F& f, const Env& env) {
    // In interval, we can handle the equality predicate exactly or by over-approximation.
    // Under-approximation does not make sense since it would give an empty interval.
    // The equality is interpreted in both bounds by over-approximation, therefore the equal element must be in \f$ \gamma(lb) \cap \gamma(ub) \f$.
    // If an exact equality is asked, we verify the interpretations in LB and UB are equal.
    if(f.is_binary() && f.sig() == EQ) {
      if(f.is_under()) {
        return iresult<F>(IError<F>(true, name, "Equality cannot be interpreted by under-approximation (it would always give an empty interval).", f));
      }
      auto cp_res = CP::interpret(f.map_approx(OVER), env);
      if(cp_res.has_value()) {
        this_type itv(cp_res.value());
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
          this_type itv(lb.value(), UB::bot());
          return std::move(iresult<F>(std::move(itv)).join_warnings(std::move(lb)));
        }
      }
    }
    // Forward to CP in case the formula `f` did not fit the cases above.
    auto cp_interpret = CP::interpret(f, env);
    if(cp_interpret.has_value()) {
      return std::move(iresult<F>(Interval(cp_interpret.value())).join_warnings(std::move(cp_interpret)));
    }
    return std::move(cp_interpret).template map_error<this_type>();
  }

  CUDA const LB& lb() const { return project<0>(cp); }
  CUDA const UB& ub() const { return project<1>(cp); }

  CUDA this_type& tell_top() {
    cp.tell_top();
    return *this;
  }

  template<class A, class M>
  CUDA this_type& tell_lb(const A& lb, BInc<M>& has_changed) {
    cp.template tell<0>(lb, has_changed);
    return *this;
  }

  template<class A, class M>
  CUDA this_type& tell_ub(const A& ub, BInc<M>& has_changed) {
    cp.template tell<1>(ub, has_changed);
    return *this;
  }

  template<class A>
  CUDA this_type& tell_lb(const A& lb) {
    cp.template tell<0>(lb);
    return *this;
  }

  template<class A>
  CUDA this_type& tell_ub(const A& ub) {
    cp.template tell<1>(ub);
    return *this;
  }

  template<class A, class M>
  CUDA this_type& tell(const Interval<A>& other, BInc<M>& has_changed) {
    cp.tell(other.cp, has_changed);
    return *this;
  }

  template<class A>
  CUDA this_type& tell(const Interval<A>& other) {
    cp.tell(other.cp);
    return *this;
  }

  CUDA this_type& dtell_bot() {
    cp.dtell_bot();
    return *this;
  }

  template<class A, class M>
  CUDA this_type& dtell_lb(const A& lb, BInc<M>& has_changed) {
    cp.template dtell<0>(lb, has_changed);
    return *this;
  }

  template<class A, class M>
  CUDA this_type& dtell_ub(const A& ub, BInc<M>& has_changed) {
    cp.template dtell<1>(ub, has_changed);
    return *this;
  }

  template<class A>
  CUDA this_type& dtell_lb(const A& lb) {
    cp.template dtell<0>(lb);
    return *this;
  }

  template<class A>
  CUDA this_type& dtell_ub(const A& ub) {
    cp.template dtell<1>(ub);
    return *this;
  }

  template<class A, class M>
  CUDA this_type& dtell(const Interval<A>& other, BInc<M>& has_changed) {
    cp.dtell(other.cp, has_changed);
    return *this;
  }

  template<class A>
  CUDA this_type& dtell(const Interval<A>& other) {
    cp.dtell(other.cp);
    return *this;
  }

  template <class A>
  CUDA bool extract(Interval<A>& ua) const {
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

  CUDA static constexpr bool is_supported_fun(Approx appx, Sig sig) {
    switch(sig) {
      case ABS: return CP::is_supported_fun(appx, NEG);
      case SUB: return CP::is_supported_fun(appx, ADD) && CP::is_supported_fun(appx, NEG);
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
      case MAX: return CP::is_supported_fun(appx, sig);
      default: return false;
    }
  }

  /** The additive inverse is obtained by pairwise negation of the components.
   * Equivalent to `neg(reverse(x))`.
   * Note that the inverse of `bot` is `bot`, simply because `bot` has no mathematical inverse. */
  CUDA static this_type additive_inverse(const this_type& x) {
    static_assert(LB::is_supported_fun(EXACT, NEG) && UB::is_supported_fun(EXACT, NEG),
      "Exact negation of interval bounds are required to compute the additive inverse.");
    return this_type(CP::template fun<EXACT, NEG>(x.as_product()));
  }

private:
  // A faster version of reverse when we know x != bot.
  template<class L>
  CUDA static constexpr this_type reverse2(const Interval<L>& x) {
    return this_type(x.ub().value(), x.lb().value());
  }

public:
  template<class L>
  CUDA static constexpr this_type reverse(const Interval<L>& x) {
    return x.is_bot() ? bot() : reverse2(x);
  }

  template<Approx appx, class L>
  CUDA static constexpr this_type neg(const Interval<L>& x) {
    using LB2 = typename Interval<L>::LB;
    using UB2 = typename Interval<L>::UB;
    return this_type(
      dual<LB>(UB2::template fun<appx, NEG>(x.ub())),
      dual<UB>(LB2::template fun<appx, NEG>(x.lb())));
  }

  // This operation preserves top, i.e., \f$ abs(x) \in [\top] \f$ if \f$ x \in [\top] \f$, \f$ [\top] \f$ being the equivalence class of top elements.
  template<Approx appx, class L>
  CUDA static constexpr this_type abs(const Interval<L>& x) {
    using LB2 = typename Interval<L>::LB;
    using UB2 = typename Interval<L>::UB;
    switch(sig(x)) {
      case PP: return x;
      case NP: return this_type(LB2::template fun<appx, ABS>(x.lb()), x.ub());
      case NN: return this_type(dual<LB>(UB2::template fun<appx, NEG>(x.ub())),
                                dual<UB>(LB2::template fun<appx, NEG>(x.lb())));
      case PN: return this_type(x.lb(), UB2::template fun<appx, ABS>(x.ub()));
    }
    assert(0); // all cases should be covered:
    return top();
  }

  template<Approx appx, Sig sig, class L>
  CUDA static constexpr this_type fun(const Interval<L>& x) {
    static_assert(sig == NEG || sig == ABS, "Unsupported unary function.");
    switch(sig) {
      case NEG: return this_type::neg<appx>(x);
      case ABS: return this_type::abs<appx>(x);
      default:
        assert(0); return x;
    }
  }

public:
  template<Approx appx, class L, class K>
  CUDA static constexpr this_type add(const Interval<L>& x, const Interval<K>& y) {
    return impl::make_itv(CP::template fun<appx, ADD>(x.as_product(), y.as_product()));
  }

  template<Approx appx, class L, class K>
  CUDA static constexpr this_type sub(const Interval<L>& x, const Interval<K>& y) {
    return this_type::add<appx>(x, neg<appx>(y));
  }

private:
  /** Characterization of the sign of the bounds (e.g., NP = lower bound is negative, upper bound is positive). */
  enum bounds_sign {
    PP, NN, NP, PN
  };

  template<class A>
  CUDA static constexpr bounds_sign sig(const Interval<A>& a) {
    if(a.lb() >= LB::zero) { return (a.ub() >= UB::zero) ? PP : PN; }
    else { return (a.ub() <= UB::zero) ? NN : NP; }
  }

  template<Approx appx, class A, class B>
  CUDA static this_type mul2(const Interval<A>& a, const Interval<B>& b) {
    return this_type(CP::template fun<appx, MUL>(a.as_product(), b.as_product()));
  }

  template<Approx appx, Sig divsig, class A, class B>
  CUDA static constexpr this_type div2(const A& a, const B& b) {
    return this_type(CP::template fun<appx, divsig>(a.as_product(), b.as_product()));
  }

public:
  /** By default, multiplication is over-approximating as it is not possible to exactly represent multiplication in general.
    Note that we do not rely on the commutativity property of multiplication. */
  template<Approx appx = OVER, class L, class K>
  CUDA static constexpr this_type mul(const Interval<L>& a, const Interval<K>& b) {
    static_assert(appx == OVER, "Only over-approximation of multiplication is supported.");
    // Interval multiplication case, [al..au] * [bl..bu]
    switch(sig(a)) {
      case PP:
        switch(sig(b)) {
          case PP: return mul2<appx>(a, b);
          case NP: return mul2<appx>(a.ub2(), b);
          case NN: return mul2<appx>(reverse2(a), b);
          case PN:
            if(b.as_product().is_top()) { return top(); }
            else { return mul2<appx>(a.lb2(), b); }
        }
      case NP:
        switch(sig(b)) {
          case PP: return mul2<appx>(a, b.ub2());
          // Note: we use meet for both bounds because UB is the dual of LB (e.g., if meet in LB is min, then meet in UB is max).
          case NP: return this_type(
              meet(LB::template fun<appx, MUL>(a.lb(), b.ub()), LB::template fun<appx, MUL>(a.ub(), b.lb())),
              meet(UB::template fun<appx, MUL>(a.lb(), b.lb()), UB::template fun<appx, MUL>(a.ub(), b.ub())));
          case NN: return mul2<appx>(reverse(a), b.lb2());
          case PN:
            if(b.as_product().is_top()) { return top(); }
            else { return zero; }
        }
      case NN:
        switch(sig(b)) {
          case PP: return mul2<appx>(a, reverse2(b));
          case NP: return mul2<appx>(a.lb2(), reverse(b));
          case NN: return mul2<appx>(reverse2(a), reverse2(b));
          case PN:
            if(b.as_product().is_top()) { return top(); }
            else { return mul2<appx>(a.ub2(), reverse2(b)); }
        }
      case PN:
        if(a.as_product().is_top()) { return top(); }
        else {
          switch(sig(b)) {
            case PP: return mul2<appx>(a, b.lb2());
            case NP: return zero;
            case NN: return mul2<appx>(reverse2(a), b.ub2());
            case PN:
              if(b.as_product().is_top()) { return top(); }
              else {
                return this_type(
                  join(LB::template fun<appx, MUL>(a.lb(), b.lb()), LB::template fun<appx, MUL>(a.ub(), b.ub())),
                  join(UB::template fun<appx, MUL>(a.lb(), b.ub()), UB::template fun<appx, MUL>(a.ub(), b.lb())));
              }
          }
        }
    }
    assert(0); // All cases should be covered.
    return top();
  }

  template<Approx appx = OVER, Sig divsig, class L, class K>
  CUDA static constexpr this_type div(const Interval<L>& a, const Interval<K>& b) {
    static_assert(appx == OVER, "Only over-approximation of division is supported.");
    // Interval division, [al..au] / [bl..bu]
    switch(sig(b)) {
      case PP:
        if(b.ub() == K::zero) { return top(); }  // b is a singleton equal to zero.
        switch(sig(a)) {
          case PP: return div2<appx, divsig>(a, reverse2(b));
          case NP: return div2<appx, divsig>(a, b.lb2());
          case NN: return div2<appx, divsig>(a, b);
          case PN:
            if(a.as_product().is_top()) { return top(); }
            else { return div2<appx, divsig>(a, b.ub2()); }
        }
      case NP:
        if(a.is_top()) { return top(); }
        else {
          if constexpr(L::preserve_inner_covers && K::preserve_inner_covers) { // In the discrete case, division can be more precise.
            switch(sig(a)) {
              case PP: return div2<appx, divsig>(a.ub2(), reverse(b));
              case NP: return this_type(
                meet(LB::template fun<appx, divsig>(a.lb(), b.lb()), LB::template fun<appx, divsig>(a.ub(), b.ub())),
                meet(UB::template fun<appx, divsig>(a.lb(), b.ub()), UB::template fun<appx, divsig>(a.ub(), b.lb())));
              case NN: return div2<appx, divsig>(a.lb2(), b);
              case PN: return (a.as_product().is_top()) ? top() : zero;
            }
          }
          else {
            return bot();
          }
        }
      case NN:
        switch(sig(a)) {
          case PP: return div2<appx, divsig>(reverse2(a), reverse2(b));
          case NP: return div2<appx, divsig>(reverse(a), b.ub2());
          case NN: return div2<appx, divsig>(reverse2(a), b);
          case PN:
            if(a.as_product().is_top()) { return top(); }
            else { return div2<appx, divsig>(reverse2(a), b.lb2()); }
        }
      case PN:
        if(b.as_product().is_top()) { return top(); }
        if constexpr(L::preserve_inner_covers && K::preserve_inner_covers) {
          switch(sig(a)) {
            case PP: return div2<appx, divsig>(a.lb2(), reverse2(b));
            case NP: return zero;
            case NN: return div2<appx, divsig>(a.ub2(), reverse2(b));
            case PN:
              if(a.as_product().is_top()) { return top(); }
              else {
                return this_type(
                  join(LB::template fun<appx, divsig>(a.lb(), b.ub()), LB::template fun<appx, divsig>(a.ub(), b.lb())),
                  join(UB::template fun<appx, divsig>(a.lb(), b.lb()), UB::template fun<appx, divsig>(a.ub(), b.ub())));
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

  template<Approx appx = OVER, Sig modsig, class L, class K>
  CUDA static constexpr this_type mod(const Interval<L>& a, const Interval<K>& b) {
    static_assert(appx == OVER, "Only over-approximation of modulo is supported.");
    if(a.is_top() || b.is_top()) { return top(); }
    if(a.lb() == dual<LB>(a.ub()) && b.lb() == dual<LB>(b.ub())) {
      auto l = LB::template fun<appx, modsig>(a.lb(), b.lb());
      auto u = UB::template fun<appx, modsig>(a.ub(), b.ub());
      return this_type(l, u);
    }
    else {
      return bot();
    }
  }

  template<Approx appx = OVER, class L, class K>
  CUDA static constexpr this_type pow(const Interval<L>& a, const Interval<K>& b) {
    static_assert(appx == OVER, "Only over-approximation of exponentiation is supported.");
    if(a.is_top() || b.is_top()) { return top(); }
    if(a.lb() == dual<LB>(a.ub()) && b.lb() == dual<LB>(b.ub())) {
      auto l = LB::template fun<appx, POW>(a.lb(), b.lb());
      auto u = UB::template fun<appx, POW>(a.ub(), b.ub());
      return this_type(l, u);
    }
    else {
      return bot();
    }
  }

  template<Approx appx, Sig sig, class L, class K>
  CUDA static constexpr this_type fun(const Interval<L>& x, const Interval<K>& y) {
    if constexpr(sig == ADD) { return this_type::add<appx>(x, y); }
    else if constexpr(sig == SUB) { return this_type::sub<appx>(x, y); }
    else if constexpr(sig == MUL) { return this_type::mul<appx>(x, y); }
    else if constexpr(is_division(sig)) { return this_type::div<appx, sig>(x, y); }
    else if constexpr(is_modulo(sig)) { return this_type::mod<appx, sig>(x, y); }
    else if constexpr(sig == POW) { return this_type::pow<appx>(x, y); }
    else if constexpr(sig == MIN || sig == MAX) { return this_type(CP::template fun<appx, sig>(x.as_product(), y.as_product())); }
    else { static_assert(
      sig == ADD || sig == SUB || sig == MUL || sig == TDIV || sig == TMOD || sig == FDIV || sig == FMOD || sig == CDIV || sig == CMOD || sig == EDIV || sig == EMOD || sig == POW || sig == MIN || sig == MAX,
      "Unsupported binary function.");
    }
  }
};

// Lattice operations

template<class L, class K>
CUDA auto join(const Interval<L>& a, const Interval<K>& b)
{
  return impl::make_itv(join(a.as_product(), b.as_product()));
}

template<class L, class K>
CUDA auto meet(const Interval<L>& a, const Interval<K>& b)
{
  return impl::make_itv(meet(a.as_product(), b.as_product()));
}

template<class L, class K>
CUDA bool operator<=(const Interval<L>& a, const Interval<K>& b)
{
  return b.is_top() || a.as_product() <= b.as_product();
}

template<class L, class K>
CUDA bool operator<(const Interval<L>& a, const Interval<K>& b)
{
  return (b.is_top() && !a.is_top()) || a.as_product() < b.as_product();
}

template<class L, class K>
CUDA bool operator>=(const Interval<L>& a, const Interval<K>& b)
{
  return b <= a;
}

template<class L, class K>
CUDA bool operator>(const Interval<L>& a, const Interval<K>& b)
{
  return b < a;
}

template<class L, class K>
CUDA bool operator==(const Interval<L>& a, const Interval<K>& b)
{
  return a.as_product() == b.as_product() || (a.is_top() && b.is_top());
}

template<class L, class K>
CUDA bool operator!=(const Interval<L>& a, const Interval<K>& b)
{
  return a.as_product() != b.as_product() && !(a.is_top() && b.is_top());
}

template<class L>
std::ostream& operator<<(std::ostream &s, const Interval<L> &itv) {
  return s << "[" << itv.lb() << ".." << itv.ub() << "]";
}

} // namespace lala

#endif
