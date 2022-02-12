// Copyright 2021 Pierre Talbot

#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#include "cartesian_product.hpp"

namespace lala {

template <typename U>
class Interval {
  public:
    using LB = U;
    using UB = typename LB::dual_type;
    using this_type = Interval<LB>;
    using dual_type = Interval<UB>;
    using CP = CartesianProduct<LB, UB>;
    using ValueType = typename CP::ValueType;

    template <typename A>
    friend class Interval;

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

    template<class Allocator>
    CUDA void print(const LVar<Allocator>& x) const {
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


} // namespace lala

#endif
