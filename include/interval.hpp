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

    template <typename A>
    friend class Interval;

  private:
    using CP = CartesianProduct<LB, UB>;
    CP cp;
    CUDA Interval(CP&& cp): cp(cp) {}
    CUDA Interval(const CP& cp): cp(cp) {}

  public:
    using Allocator = typename CP::Allocator;

    CUDA Interval(LB&& lb, UB&& ub): cp(std::forward<LB>(lb), std::forward<UB>(ub)) {}
    CUDA Interval(const LB& lb, const UB& ub): cp(lb, ub) {}
    CUDA Interval(const this_type& other): cp(other.cp) {}

    CUDA static this_type bot() { return Interval(CP::bot()); }
    CUDA static this_type top() { return Interval(CP::top()); }
    CUDA bool is_top() const { return cp.is_top() || !(lb().order(ub())); }
    CUDA bool is_bot() const { return cp.is_bot(); }
    CUDA dual_type dual() const { return dual_type(cp.dual()); }

    template<typename F>
    CUDA static thrust::optional<this_type> interpret(const F& f) {
      // In interval, we can handle the equality predicate exactly if the bounds can be represented exactly.
      if(is_v_op_z(f, EQ)) {
        auto lb = CP::template interpret_one<0>(F::make_binary(f.seq(0), GEQ, f.seq(1), UNTYPED, f.approx()));
        if(lb.has_value()) {
          auto ub = CP::template interpret_one<1>(F::make_binary(f.seq(0), LEQ, f.seq(1), UNTYPED, f.approx()));
          if(ub.has_value()) {
            return Interval(lb->join(*ub));
          }
        }
      }
      // If NEQ is under-approximated in both bounds, we risk to approximate to top while it would be correct to under-approximate only in one of the bounds.
      else if(is_v_op_z(f, NEQ) && f.approx() == UNDER) {
        auto x = CP::interpret(f);
        if(x.has_value()) {
          auto itv = Interval(*x);
          if(!itv.is_top()) {
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

    CUDA const LB& lb() const { return cp.template project<0>(); }
    CUDA const UB& ub() const { return cp.template project<1>(); }
    CUDA this_type& tell(const LB& lb, bool& has_changed) {
      cp.tell(lb, has_changed);
      return *this;
    }
    CUDA this_type& tell(const UB& ub, bool& has_changed) {
      cp.tell(ub, has_changed);
      return *this;
    }
    CUDA this_type& tell(const this_type& other, bool& has_changed) {
      cp.tell(other.cp, has_changed);
      return *this;
    }
    CUDA this_type& join(const this_type& other) {
      cp.join(other.cp);
      return *this;
    }
    CUDA bool order(const dual_type& other) const { return cp.order(other.cp); }
    CUDA void reset(const this_type& other) { cp.reset(other.cp); }
    CUDA this_type clone() const { return this_type(cp.clone()); }

    template<typename Alloc = Allocator>
    CUDA TFormula<Alloc> deinterpret(const LVar<Allocator>& x, const Alloc& allocator = Alloc()) const {
      if(is_top()) {
        return TFormula<Allocator>::make_false();
      }
      else if(is_bot()) {
        return TFormula<Allocator>::make_true();
      }
      else {
        return CP::deinterpret(x, allocator);
      }
    }

    CUDA void print(const LVar<Allocator>& x) const {
      printf("[");
      ::print(lb());
      printf("..");
      ::print(ub());
      printf("]");
    }
};

template<class U>
CUDA bool operator==(const Interval<U>& lhs, const Interval<U>& rhs) {
  return lhs.lb() == rhs.lb() && lhs.ub() == rhs.ub();
}

template<class U>
CUDA bool operator!=(const Interval<U>& lhs, const Interval<U>& rhs) {
  return !(lhs == rhs);
}

}

#endif
