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
    return Interval<LB>(cp);
  }
}

/** An interval is a Cartesian product of a lower and upper bounds, themselves represented as lattices.
    One difference, is that the \f$ \top \f$ can be represented by multiple interval elements, whenever \f$ l > u \f$, therefore some operations are different then on the Cartesian product, e.g., \f$ [3..2] = [4..1] \f$ in the interval lattice. */
template <class U>
class Interval {
public:
  using LB = U;
  using UB = typename LB::reverse_type;
  using this_type = Interval<LB>;
  using CP = CartesianProduct<LB, UB>;
  using value_type = typename CP::value_type;

  template <class A>
  friend class Interval;

private:
  CP cp;
  CUDA Interval(CP&& cp): cp(cp) {}
  CUDA Interval(const CP& cp): cp(cp) {}

public:
  /** Given a value \f$ x \in U \f$ where \f$ U \f$ is the universe of discourse, we initialize a singleton interval \f$ [x..x] \f$. */
  CUDA Interval(const typename U::value_type& x): cp(x, x) {}
  CUDA Interval(const LB& lb, const UB& ub): cp(lb, ub) {}

  template<class A>
  CUDA Interval(const Interval<A>& other): cp(other.cp) {}

  static constexpr this_type zero = Interval(LB::zero, UB::zero);
  static constexpr this_type one = Interval(LB::one, UB::one);

  CUDA static this_type bot() { return Interval(CP::bot()); }
  CUDA static this_type top() { return Interval(CP::top()); }
  CUDA local::BInc is_top() const { return cp.is_top() || lb() > ub(); }
  CUDA local::BDec is_bot() const { return cp.is_bot(); }
  CUDA const CP& as_product() const { return cp; }
  CUDA value_type value() const { return cp.value(); }

  template<class F>
  CUDA static iresult<this_type> interpret(const F& f) {
    // In interval, we can handle the equality predicate exactly if the bounds can be represented exactly.
    if(f.is_binary() && f.sig() == EQ) {

    }
    if(is_v_op_z(f, EQ)) {
      auto lb = CP::template interpret_one<0>(F::make_binary(f.seq(0), GEQ, f.seq(1), UNTYPED, f.approx()));
      if(lb.has_value()) {
        auto ub = CP::template interpret_one<1>(F::make_binary(f.seq(0), LEQ, f.seq(1), UNTYPED, f.approx()));
        if(ub.has_value()) {
          return Interval(join(*lb,*ub));
        }
      }
    }
    // If NEQ is under-approximated in both bounds, we risk to approximate to top while it would be correct to under-approximate only in one of the bounds.
    else if(is_v_op_z(f, NEQ) && f.approx() == UNDER) {
      auto x = CP::interpret(f);
      if(x.has_value()) {
        auto itv = Interval(*x);
        if(!itv.is_top().value()) {
          return itv;
        }
      }
      auto lb = CP::template interpret_one<0>(f);
      if(lb.has_value()) {
        return Interval(*lb);
      }
    }
    // Forward to CP in case the formula `f` did not fit the cases above.
    auto cp_interpret = CP::interpret(f);
    if(cp_interpret.has_value()) {
      return Interval(*cp_interpret);
    }
    return {};
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

  template<class Allocator>
  CUDA TFormula<Allocator> deinterpret(const LVar<Allocator>& x, const Allocator& allocator = Allocator()) const {
    if(is_top()) {
      return TFormula<Allocator>::make_false();
    }
    else if(is_bot()) {
      return TFormula<Allocator>::make_true();
    }
    else {
      return cp.deinterpret(x, allocator);
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
   *  Equivalent to `neg(reverse(x))`. */
  CUDA this_type additive_inverse(const this_type& x) {
    static_assert(LB::is_supported_fun(EXACT, NEG) && UB::is_supported_fun(EXACT, NEG),
      "Exact negation of interval bounds are required to compute the additive inverse.");
    return this_type(CP::fun<EXACT, NEG>(x.as_product()));
  }

  template<class L>
  CUDA static constexpr this_type reverse(const Interval<L>& x) {
    return this_type(ub(), lb());
  }

  template<Approx appx, class L>
  CUDA static constexpr this_type neg(const Interval<L>& x) {
    return this_type(UB::fun<appx, NEG>(x.ub()), LB::fun<appx, NEG>(x.lb()));
  }

  // This operation preserves top, i.e., \f$ abs(x) \in [\top] \f$ if \f$ x \in [\top] \f$, \f$ [\top] \f$ being the equivalence class of top elements.
  template<Approx appx, class L>
  CUDA static constexpr this_type abs(const Interval<L>& x) {
    using LB2 = typename Interval<L>::LB;
    using UB2 = typename Interval<L>::UB;
    const LB2& lb = x.lb();
    if(lb > LB2::zero) {
      return x;
    }
    else {
      const UB2& ub = x.ub();
      if(ub < UB2::zero) {
        return this_type(UB2::fun<appx, NEG>(ub), LB2::fun<appx, NEG>(lb));
      }
      return this_type(LB2::zero, ub);
    }
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

  template<Approx appx, class A, class B>
  CUDA static constexpr this_type add(const A& x, const B& y) {
    return make_itv(CP::fun<appx, sig>(x.as_product(), y.as_product()));
  }

  template<Approx appx, class A, class B>
  CUDA static constexpr this_type sub(const A& x, const B& y) {
    return this_type::add<appx>(x, neg<appx>(y));
  }

private:
  /** Characterization of the sign of the bounds (e.g., NP = lower bound is negative, upper bound is positive). */
  enum bounds_sign {
    PP, NN, NP, PN
  };

  template<class A>
  CUDA static constexpr bounds_sign sig(const Interval<A>& a) {
    if(a.lb().value() >= LB::zero) { return (a.ub().value() >= UB::zero) ? PP : PN; }
    else { return (a.ub().value() <= UB::zero) ? NN : NP; }
  }

  template<class A>
  struct is_interval {
    static constexpr bool value = false;
  };

  template<class A>
  struct is_interval<Interval<A>> {
    static constexpr bool value = true;
  };

  template<class A>
  inline static constexpr bool is_interval_v = is_interval<A>::value;

  CUDA operator CP() const { return as_product(); }

public:
  /** By default, multiplication is over-approximating as it is not possible to exactly represent multiplication in general.
    Under-approximation of multiplication is not the best possible: it returns a singleton [al*bl..al*bl] where only the lower bounds are multiplied.
    Note that we do not rely on the commutativity property of multiplication. */
  template<Approx appx = OVER, class A, class B>
  CUDA static constexpr this_type mul(const A& a, const B& b) {
    static_assert(is_interval_v<A> || is_interval_v<B>,
      "Multiplication over interval is only defined when one of the operands is an interval.");
    static_assert(appx != EXACT, "Multiplication cannot be exactly represented in intervals.");
    // I. Under-approximation case.
    if constexpr(appx == UNDER) {
      auto l = LB::fun<appx, MUL>(is_interval_v<A> ? lb(a) : a, is_interval_v<B> ? lb(b) : b);
      return Interval<L>(l, l);
    }
    // II. Interval multiplied by a bound, [al..au] * b or a * [bl..bu].
    else if constexpr(!is_interval_v<B>) {
      return (b >= B::zero)
        ? this_type(CP::fun<appx, MUL>(a, b));
        : this_type(CP::fun<appx, MUL>(reverse(a), b));
    }
    else if constexpr(!is_interval_v<A>) {
      return (a >= A::zero)
        ? this_type(CP::fun<appx, MUL>(a, b));
        : this_type(CP::fun<appx, MUL>(a, reverse(b)));
    }
    // III. Interval multiplication case, [al..au] * [bl..bu]
    else {
      switch(sig(a)) {
        case PP:
          switch(sig(b)) {
            case PP: return this_type(CP::fun<appx, MUL>(a, b));
            case NP: return this_type(CP::fun<appx, MUL>(a.ub(), b));
            case NN: return this_type(CP::fun<appx, MUL>(reverse(a), b));
            case PN: return this_type(CP::fun<appx, MUl>(a.lb(), b));
          }
        case NP:
          switch(sig(b)) {
            case PP: return this_type(CP::fun<appx, MUL>(a, b.ub()));
            // Note: we use meet for both bounds because UB is the dual of LB (e.g., if meet in LB is min, then meet in UB is max).
            case NP: return this_type(
                meet(LB::fun<appx, MUL>(a.lb(), b.ub()), LB::fun<appx, MUL>(a.ub(), b.lb())),
                meet(UB::fun<appx, MUL>(a.lb(), b.lb()), UB::fun<appx, MUL>(a.ub(), b.ub())));
            case NN: return this_type(CP::fun<appx, MUL>(reverse(a), b.lb()));
            case PN: return zero;
          }
        case NN:
          switch(sig(b)) {
            case PP: return this_type(CP::fun<appx, MUL>(a, reverse(b)));
            case NP: return this_type(CP::fun<appx, MUL>(a.lb(), reverse(b)));
            case NN: return this_type(CP::fun<appx, MUL>(reverse(a), reverse(b)));
            case PN: return this_type(CP::fun<appx, MUL>(a.ub(), reverse(b)));
          }
        case PN:
          switch(sig(b)) {
            case PP: return this_type(CP::fun<appx, MUL>(a, b.lb()));
            case NP: return zero;
            case NN: return this_type(CP::fun<appx, MUL>(reverse(a), b.ub()));
            case PN: return this_type(
                join(LB::fun<appx, MUL>(a.lb(), b.lb()), LB::fun<appx, MUL>(a.ub(), b.ub())),
                join(UB::fun<appx, MUL>(a.lb(), b.ub()), UB::fun<appx, MUL>(a.ub(), b.lb())));
          }
      }
    }
  }

  template<Approx appx = OVER, Sig divsig, class A, class B>
  CUDA static constexpr this_type div(const A& a, const B& b) {
    static_assert(is_interval_v<A> || is_interval_v<B>,
      "Division over interval is only defined when one of the operands is an interval.");
    static_assert(appx != EXACT, "Division cannot be exactly represented in intervals.");
    // I. Under-approximation case.
    if constexpr(appx == UNDER) {
      auto l = LB::fun<appx, divsig>(is_interval_v<A> ? lb(a) : a, is_interval_v<B> ? lb(b) : b);
      return this_type(l, l);
    }
    // II. Interval divided by a bound, [al..au] / b or a / [bl..bu].
    else if constexpr(!is_interval_v<B>) {
      return (b >= B::zero)
        ? this_type(CP::fun<appx, divsig>(a, b));
        : this_type(CP::fun<appx, divsig>(reverse(a), b));
    }
    else if constexpr(!is_interval_v<A>) {
      if(b.lb().value() < B::zero && b.ub().value() > B::zero) {
        return this_type::bot();
      }
      else {
        return (a >= A::zero)
          ? this_type(CP::fun<appx, divsig>(a, b));
          : this_type(CP::fun<appx, divsig>(a, reverse(b)));
      }
    }
    // III. Interval division, [al..au] / [bl..bu]
    else {
      switch(sig(b)) {
        case NP: return this_type::bot();
        case PN: return this_type::top();  /* This is by duality with the case above, but I am unsure if it makes sense. */
        case PP:
          if(b.lb() == B::LB::zero) { return this_type::top(); }  // b is a singleton equal to zero.
          switch(sig(a)) {
            case PP: return this_type(CP::fun<appx, divsig>(a, reverse(b)));
            case NP: return this_type(CP::fun<appx, divsig>(a, b.lb()));
            case NN: return this_type(CP::fun<appx, divsig>(a, b));
            case PN: return this_type(CP::fun<appx, divsig>(a, b.ub()));
          }
        case NN:
          switch(sig(a)) {
            case PP: return this_type(CP::fun<appx, divsig>(reverse(a), reverse(b)));
            case NP: return this_type(CP::fun<appx, divsig>(reverse(a), b.ub()));
            case NN: return this_type(CP::fun<appx, divsig>(reverse(a), b));
            case PN: return this_type(CP::fun<appx, divsig>(reverse(a), b.lb()));
          }
      }
    }
  }

  template<Approx appx = OVER, Sig modsig, class A, class B>
  CUDA static constexpr this_type mod(const A& a, const B& b) {
    static_assert(is_interval_v<A> && is_interval_v<B>,
      "Modulo over interval is only defined when both of the operands are an interval.");
    static_assert(appx != EXACT, "Modulo cannot be exactly represented in intervals.");
    // I. Under-approximation case.
    if constexpr(appx == UNDER) {
      auto l = LB::fun<appx, modsig>(is_interval_v<A> ? lb(a) : a, is_interval_v<B> ? lb(b) : b);
      return this_type(l, l);
    }
    else {
      if(a.lb() == a.ub() && b.lb() == b.ub()) {
        auto l = LB::fun<appx, modsig>(a.lb(), b.lb());
        auto u = UB::fun<appx, modsig>(a.ub(), b.ub());
        return this_type(l, u);
      }
      else {
        return a;
      }
    }
  }

  template<Approx appx = OVER, class A, class B>
  CUDA static constexpr this_type pow(const A& a, const B& b) {
    static_assert(is_interval_v<A> && is_interval_v<B>,
      "Exponentiation over interval is only defined when both of the operands are an interval.");
    static_assert(appx != EXACT, "Exponentiation cannot be exactly represented in intervals.");
    // I. Under-approximation case.
    if constexpr(appx == UNDER) {
      auto l = LB::fun<appx, POW>(is_interval_v<A> ? lb(a) : a, is_interval_v<B> ? lb(b) : b);
      return this_type(l, l);
    }
    else {
      if(a.lb() == a.ub() && b.lb() == b.ub()) {
        auto l = LB::fun<appx, POW>(a.lb(), b.lb());
        auto u = UB::fun<appx, POW>(a.ub(), b.ub());
        return this_type(l, u);
      }
      else {
        return a;
      }
    }
  }

  template<Approx appx, Sig sig, class A, class B>
  CUDA static constexpr this_type fun(const A& x, const B& y) {
    static_assert(
      sig == ADD || sig == SUB || sig == MUL || sig == TDIV || sig == TMOD || sig == FDIV || sig == FMOD || sig == CDIV || sig == CMOD || sig == EDIV || sig == EMOD || sig == POW || sig == MIN || sig == MAX || sig == EQ || sig == NEQ || sig == LEQ || sig == GEQ || sig == LT || sig == GT,
      "Unsupported binary function.");
    switch(sig) {
      case ADD: return this_type::add<appx>(x, y);
      case SUB: return this_type::sub<appx>(x, y);
      case MUL: return this_type::mul<appx>(x, y);
      case DIV:
      case TDIV:
      case FDIV:
      case CDIV:
      case EDIV: return this_type::div<appx, sig>(x, y);
      case TMOD:
      case FMOD:
      case CMOD:
      case EMOD: return this_type::mod<appx, sig>(x, y);
      case POW: return this_type::pow<appx>(x, y);
      case MIN:
      case MAX: return this_type(CP::fun<appx, sig>(a, b));
      default: assert(0); return x;
    }
  }
};

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

template<class A, class B>
CUDA bool operator<=(const A& a, const B& b)
{
  return b.is_top() || a.as_product() <= b.as_product();
}

template<class A, class B>
CUDA bool operator<(const A& a, const B& b)
{
  return (b.is_top() && !a.is_top()) || a.as_product() < b.as_product();
}

template<class A, class B>
CUDA bool operator>=(const A& a, const B& b)
{
  return b <= a;
}

template<class A, class B>
CUDA bool operator>(const A& a, const B& b)
{
  return b < a;
}

template<class A, class B>
CUDA bool operator==(const A& a, const B& b)
{
  return a.as_product() == b.as_product() || (a.is_top() && b.is_top());
}

template<class A, class B>
CUDA bool operator!=(const A& a, const B& b)
{
  return a.as_product() != b.as_product() && !(a.is_top() && b.is_top());
}

} // namespace lala

#endif
