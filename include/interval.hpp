// Copyright 2021 Pierre Talbot

#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#include "cartesian_product.hpp"

namespace lala {

template <typename U>
class Interval;

template<class L>
struct arithmetic_projection {};

template<class U>
struct arithmetic_projection<Interval<ZTotalOrder<U>>> {
  using sub_type = Interval<ZTotalOrder<U>>;
  CUDA const typename ZTotalOrder<U>::pos_t lbp() const { return static_cast<const sub_type*>(this)->lb().pos(); }
  CUDA const typename ZTotalOrder<U>::neg_t lbn() const { return static_cast<const sub_type*>(this)->lb().neg(); }
  CUDA const typename ZTotalOrder<U>::dual_type::pos_t ubp() const { return static_cast<const sub_type*>(this)->ub().pos(); }
  CUDA const typename ZTotalOrder<U>::dual_type::neg_t ubn() const { return static_cast<const sub_type*>(this)->ub().neg(); }
};

template <typename U>
class Interval: public arithmetic_projection<Interval<U>> {
  public:
    using LB = U;
    using UB = typename LB::dual_type;
    using this_type = Interval<LB>;
    using dual_type = Interval<UB>;
    using CP = CartesianProduct<LB, UB>;
    using ValueType = typename CP::ValueType;

    template <typename A>
    friend class Interval;


    template<class L, class K>
    friend CUDA typename join_t<Interval<L>, Interval<K>>::type
    join(const Interval<L>& a, const Interval<K>& b);

    template<class L, class K>
    friend CUDA typename meet_t<Interval<L>, Interval<K>>::type
    meet(const Interval<L>& a, const Interval<K>& b);

  private:
    CP cp;
    CUDA Interval(CP&& cp): cp(cp) {}
    CUDA Interval(const CP& cp): cp(cp) {}

  public:
    CUDA Interval(LB&& lb, UB&& ub): cp(std::forward<LB>(lb), std::forward<UB>(ub)) {}
    CUDA Interval(const LB& lb, const UB& ub): cp(lb, ub) {}
    CUDA Interval(const this_type& other): cp(other.cp) {}
    CUDA this_type& operator=(this_type&& other) {
      cp = std::move(other.cp);
      return *this;
    }

    CUDA static this_type bot() { return Interval(CP::bot()); }
    CUDA static this_type top() { return Interval(CP::top()); }
    CUDA BInc is_top() const { return lor(cp.is_top(), gt<LB>(lb(), ub())); }
    CUDA BDec is_bot() const { return cp.is_bot(); }
    CUDA dual_type dual() const { return dual_type(cp.dual()); }
    CUDA const CP& as_product() const { return cp; }
    CUDA ValueType value() const { return cp.value(); }

    template<typename F>
    CUDA static thrust::optional<this_type> interpret(const F& f) {
      // In interval, we can handle the equality predicate exactly if the bounds can be represented exactly.
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

    CUDA this_type& tell(const LB& lb, BInc& has_changed) {
      cp.tell(lb, has_changed);
      return *this;
    }
    CUDA this_type& tell(const UB& ub, BInc& has_changed) {
      cp.tell(ub, has_changed);
      return *this;
    }
    CUDA this_type& tell(const this_type& other, BInc& has_changed) {
      cp.tell(other.cp, has_changed);
      return *this;
    }
    CUDA this_type& dtell(const LB& lb, BInc& has_changed) {
      cp.dtell(lb, has_changed);
      return *this;
    }
    CUDA this_type& dtell(const UB& ub, BInc& has_changed) {
      cp.dtell(ub, has_changed);
      return *this;
    }
    CUDA this_type& dtell(const this_type& other, BInc& has_changed) {
      cp.dtell(other.cp, has_changed);
      return *this;
    }
    CUDA this_type clone() const { return this_type(cp.clone()); }

    template<class Allocator>
    CUDA TFormula<Allocator> deinterpret(const LVar<Allocator>& x, const Allocator& allocator = Allocator()) const {
      if(is_top().guard()) {
        return TFormula<Allocator>::make_false();
      }
      else if(is_bot().value()) {
        return TFormula<Allocator>::make_true();
      }
      else {
        return CP::deinterpret(x, allocator);
      }
    }

    CUDA void print() const {
      printf("[");
      ::print(lb());
      printf("..");
      ::print(ub());
      printf("]");
    }
};

template<class L, class K>
CUDA typename join_t<Interval<L>, Interval<K>>::type
join(const Interval<L>& a, const Interval<K>& b)
{
  using R = typename join_t<Interval<L>, Interval<K>>::type;
  return R(join(a.as_product(), b.as_product()));
}

template<class L, class K>
CUDA typename meet_t<Interval<L>, Interval<K>>::type
meet(const Interval<L>& a, const Interval<K>& b)
{
  using R = typename meet_t<Interval<L>, Interval<K>>::type;
  return R(meet(a.as_product(), b.as_product()));
}

template<class O, class L, class K>
CUDA typename leq_t<O, Interval<L>, Interval<K>>::type leq(
  const Interval<L>& a,
  const Interval<K>& b)
{
  return leq<typename O::CP>(a.as_product(), b.as_product());
}

template<class O, class L, class K>
CUDA typename lt_t<O, Interval<L>, Interval<K>>::type lt(
  const Interval<L>& a,
  const Interval<K>& b)
{
  return lt<typename O::CP>(a.as_product(), b.as_product());
}

template<Approx appx = EXACT, class L, class K>
CUDA Interval<L> add(const Interval<L>& a, const K& b) {
  if(a.is_top().guard()) { return a; }
  if constexpr(std::is_same_v<K, Interval<L>>) {
    if(b.is_top().guard()) { return b; }
    return Interval<L>(add<appx>(a.lb(), b.lb()), add<appx>(a.ub(), b.ub()));
  }
  else {
    return Interval<L>(add<appx>(a.lb(), b), add<appx>(a.ub(), b));
  }
}

template<Approx appx = EXACT, class L, class K, std::enable_if_t<!std::is_same<L, Interval<K>>::value, bool> = true>
CUDA Interval<K> add(const L& a, const Interval<K>& b) {
  return add<appx>(b, a);
}

template<Approx appx = EXACT, class L>
CUDA Interval<L> rev_add(const Interval<L>& a, const Interval<L>& b) {
  return Interval<L>(sub<appx>(a.lb(), b.lb().value()), sub<appx>(a.ub(), b.ub().value()));
}

template<Approx appx = EXACT, class L, class K>
CUDA Interval<L> sub(const Interval<L>& a, const K& b) {
  if(a.is_top().guard()) { return a; }
  if constexpr(std::is_same_v<K, Interval<L>>) {
    if(b.is_top().guard()) { return b; }
    return Interval<L>(sub<appx>(a.lb(), b.ub()), sub<appx>(a.ub(), b.lb()));
  }
  else {
    return Interval<L>(sub<appx>(a.lb(), b), sub<appx>(a.ub(), b));
  }
}

template<Approx appx = EXACT, class L, class K, std::enable_if_t<!std::is_same<L, Interval<K>>::value, bool> = true>
CUDA Interval<K> sub(const L& a, const Interval<K>& b) {
  if(b.is_top().guard()) { return b; }
  return Interval<K>(sub<appx>(a, b.ub()), sub<appx>(a, b.lb()));
}

// By default, multiplication is over-approximating as it is not possible to exactly represent multiplication in general.
// Under-approximation of multiplication is not the best possible, it returns a singleton with the lower bound.
template<Approx appx = OVER, class L, class K, std::enable_if_t<appx != EXACT, bool> = true>
CUDA Interval<L> mul(const Interval<L>& a, const K& b) {
  // i) Check if the operands are equal to top.
  if(a.is_top().guard()) {
    return a;
  }
  if constexpr(std::is_same_v<K, Interval<L>>) {
    if(b.is_top().guard()) {
      return b;
    }
  }
  // ii) Under-approximation case.
  if constexpr(appx == UNDER) {
    auto l = mul<appx>(a.lb().value(), b.lb().value());
    return Interval<L>(l, l);
  }
  // iii) Interval multiplication case.
  if constexpr(std::is_same_v<K, Interval<L>>) {
    // When both arguments are positive integers, we have [a..b] * [c..d] = [a*c..b*d].
    // As this is a very common case, we make a special case out of it (for efficiency).
    if constexpr(std::is_same_v<L, ZPInc<typename L::ValueType>>) {
      return Interval<L>(mul<appx>(a.lb(), b.lb()), mul<appx>(a.ub(), b.ub()));
    }
    // General case [al..au] * [bl..bu]
    else {
      // au <= 0
      if(leq<L>(a.ub(), 0).guard()) {
        if(leq<L>(b.ub(), 0).guard()) { return Interval<L>(mul<appx>(a.ubn(), b.ubn()), mul<appx>(a.lbn(), b.lbn())); }
        if(geq<L>(b.lb(), 0).guard()) { return Interval<L>(mul<appx>(a.lbn(), b.ubp()), mul<appx>(a.ubn(), b.lbp())); }
        else { return Interval<L>(mul<appx>(a.lbn(), b.ubp()), mul<appx>(a.lbn(), b.lbn())); }
      }
      // al >= 0
      else if(geq<L>(a.lb(), 0).guard()) {
        if(leq<L>(b.ub(), 0).guard()) {
          return Interval<L>(mul<appx>(a.ubp(), b.lbn()), mul<appx>(a.lbp(), b.ubn()));
        }
        else if(geq<L>(b.lb(), 0).guard()) {
          return Interval<L>(mul<appx>(a.lb(), b.lb()), mul<appx>(a.ub(), b.ub()));
        }
        else {
          return Interval<L>(mul<appx>(a.ubp(), b.lbn()), mul<appx>(a.ubp(), b.ubp()));
        }
      }
      // al < 0 < au
      else {
        if(leq<L>(b.ub(), 0).guard()) {
          return Interval<L>(mul<appx>(a.ubp(), b.lbn()), mul<appx>(a.lbn(), b.lbn()));
        }
        else if(geq<L>(b.lb(), 0).guard()) {
          return Interval<L>(mul<appx>(a.lbn(), b.ubp()), mul<appx>(a.ubp(), b.ubp()));
        }
        else {
          return Interval<L>(
            meet(mul<appx>(a.lbn(), b.ubp()), mul<appx>(a.ubp(), b.lbn())),
            meet(mul<appx>(a.lbn(), b.lbn()), mul<appx>(a.ubp(), b.ubp())));
        }
      }
    }
  }
  // iv) Multiplication of an interval and a constant.
  else {
    if(b >= 0) {
      return Interval<L>(mul<appx>(a.lb(), spos(b)), mul<appx>(a.ub(), spos(b)));
    }
    else {
      return Interval<L>(mul<appx>(a.ub(), sneg(b)), mul<appx>(a.lb(), sneg(b)));
    }
  }
}

template<Approx appx = OVER, class L, class K, std::enable_if_t<!std::is_same<L, Interval<K>>::value, bool> = true>
CUDA Interval<K> mul(const L& a, const Interval<K>& b) {
  return mul<appx>(b, a);
}

template<Approx appx = OVER, class L, class K, std::enable_if_t<appx != EXACT, bool> = true>
CUDA Interval<L> div(const Interval<L>& a, const K& b) {
  if(a.is_top().guard()) {
    return a;
  }
  if constexpr(std::is_same_v<K, Interval<L>>) {
    if(b.is_top().guard()) {
      return b;
    }
  }
  if constexpr(appx == UNDER) {
    auto l = div<appx>(a.lb().value(), b.lb().value());
    return Interval<L>(l, l);
  }
  if constexpr(std::is_same_v<K, Interval<L>>) {
    // When both arguments are positive integers, we have [a..b] / [c..d] = [a/d..b/c].
    // If `d == 0`, the result is top.
    // If `c == 0`, the result is [a/d..b].
    // As this is a very common case, we make a special case out of it (for efficiency).
    if(std::is_same_v<L, ZPInc<typename L::ValueType>> || (land(geq<L>(a.lb(), 0), geq<L>(b.lb(), 0)).guard())) {
      if(leq<L>(b.ub(), 0).guard()) { // d == 0. Monotonic as we already checked a >= 0.
        return Interval<L>::top();
      }
      if(gt<L>(b.lb(), 0).guard()) { // c > 0
        return Interval<L>(div<appx, L>(a.lbp(), b.ubp()), div<appx, typename L::dual_type>(a.ubp(), b.lbp()));
      }
      // c == 0
      return Interval<L>(div<appx>(a.lbp(), b.ubp()), a.ubp());
    }
    // General case [al..au] / [bl..bu]
    else {
      assert(false); // not implemented.
      return Interval<L>::bot();
    }
  }
  else {
    return div<appx>(a, Interval<L>(unwrap(b), unwrap(b)));
  }
}

} // namespace lala

#endif
