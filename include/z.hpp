// Copyright 2021 Pierre Talbot

#ifndef Z_HPP
#define Z_HPP

#include "thrust/optional.h"
#include "utility.hpp"
#include "darray.hpp"

namespace lala {

/** The lattice of increasing integers. */
template<typename VT, typename Alloc>
class ZInc {
public:
  typedef VT ValueType;
  typedef Alloc Allocator;
  typedef ZInc<ValueType, Allocator> this_type;
private:
  VT value;

  CUDA ZInc() {}
public:
  typedef ZInc LogicalElement;

  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static ZInc bot() {
    ZInc zi;
    zi.value = Limits<ValueType>::bot();
    return zi;
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static ZInc top() {
    ZInc zi;
    zi.value = Limits<ValueType>::top();
    return zi;
  }

  CUDA explicit operator ValueType() const { return value; }

  /** Similar to \f$[\![x \geq i]\!]\f$ for any name `x`. */
  template<typename VT2>
  CUDA ZInc(VT2 i): value(static_cast<ValueType>(i)) {
    assert(i > bot().value && i < top().value);
  }

  template<typename Allocator2>
  CUDA ZInc(const ZInc<ValueType, Allocator2>& i): value(ValueType(i)) {}
  CUDA ZInc(const this_type& i): value(i.value) {}

  CUDA bool operator==(const this_type& other) const {
    return value == other.value;
  }

  /** Expects a predicate of the form `x <op> i` where `x` is any variable's name, and `i` an integer.
    - If `appx` is EXACT: `op` can be >= or >.
    - If `appx` is UNDER: `op` can be >=, > or !=.
    - If `appx` is OVER: `op` can be >=, > or ==.
  An empty optional is also returned if the element would overflow in `int_type`.
    */
  CUDA thrust::optional<LogicalElement> interpret(Approx appx, const Formula& f) {
    typedef Formula F;
    if(f.tag == F::TRUE) {
      return bot();
    }
    else if(f.tag == F::FALSE) {
      return top();
    }
    if(SHAPE(f, F::GEQ, F::AVAR, F::LONG)) {      // x >= 4
      return ZInc(f.children[1].i);
    }
    else if(SHAPE(f, F::GT, F::AVAR, F::LONG)) {  // x > 4
      return ZInc(f.children[1].i + 1);
    }
    // Under-approximation of `x != 4` as `5`.
    else if(SHAPE(f, F::NEQ, F::AVAR, F::LONG) && appx == UNDER) {
      return ZInc(f.children[1].i + 1);
    }
    // Over-approximation of `x == 4` as `4`.
    else if(SHAPE(f, F::EQ, F::AVAR, F::LONG) && appx == OVER) {
      return ZInc(f.children[1].i);
    }
    return {};
  }

  /** \f$ a \sqcup b = \mathit{max}(a,b) \f$. */
  CUDA this_type& join(const LogicalElement& other) {
    value = max(other.value, value);
    return *this;
  }

  /** \f$ a \sqcap b = \mathit{min}(a,b) \f$. */
  CUDA this_type& meet(const LogicalElement& other) {
    value = min(other.value, value);
    return *this;
  }

  /** Has no effect.
     \return `true`. */
  CUDA bool refine() { return true; }

  /** \f$ a \models \varphi \f$ is defined as \f$ a \geq [\![\varphi]\!] \f$. */
  CUDA bool entailment(const LogicalElement& other) const {
    return value >= ValueType(other);
  }

  template<typename Allocator>
  CUDA DArray<LogicalElement, Allocator> split(Allocator& allocator) const {
    if(*this == top()) {
      return DArray<LogicalElement, Allocator>(0, allocator);
    }
    else {
      return DArray<LogicalElement, Allocator>(1, value, allocator);
    }
  }

  /** An abstract domain can be _eventually under-approximating_ which means that after a sufficient number of split and refine operations, it always reach an under-approximating element.
   \return `true` if \f$\gamma(a) \subseteq [\![\varphi]\!]^\flat\f$. `false` is returned whenever `a` is an over-approximation or if we do not know whether `a` is an under-approximation. */
  CUDA bool is_underappx() const {
    return true;
  }

  /** Reset the internal counter to the one of `other`. */
  CUDA void reset(const this_type& other) {
    value = other.value;
  }

  /** \return A copy of the current abstract element. */
  CUDA this_type clone() const {
    return *this;
  }

  /** \return \f$ _ \geq i \f$ where `_` is an arbitrary variable's name and `i` the integer value. */
  template<typename Allocator>
  CUDA Formula deinterpret(Allocator& allocator) const {
    return make_x_op_i(allocator, Formula::GEQ, 0, value);
  }

  /** Print the current element with the logical name of the variables. */
  CUDA void print() const {
    if(value == bot().value) {
      printf("%c", 0x22A5);
    }
    else if(value == top().value) {
      printf("%c", 0x22A4);
    }
    else if(value >= 0) {
      printf("%llu", (unsigned long long int) value);
    }
    else {
      printf("%lld", (long long int) value);
    }
  }
};

} // namespace lala

#endif