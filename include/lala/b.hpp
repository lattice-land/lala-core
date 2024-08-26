// Copyright 2024 Pierre Talbot

#ifndef LALA_CORE_B_HPP
#define LALA_CORE_B_HPP

#include "battery/memory.hpp"

namespace lala {

template <class Mem> class B;
namespace local {
  using B = ::lala::B<::battery::local_memory>;
}

/** This class represents a Boolean lattice \langle \{\mathit{false},\mathit{true}\}, \implies, \lor, \land, \mathit{false}, \mathit{true} \rangle \f$.
 * The order, join and meet operations are given by the usual Boolean logical connectors.
 * Note that \f$ \bot = \mathit{false} \f$ and \f$ \top = \mathit{true} \f$.
 * It cannot represent a logical formula and does not have a concretization function to a concrete domain.
 * For instance, it is used for extra-logical variables such as the variable `has_changed` indicating whether something changed or not.
*/
template<class Mem>
class B
{
public:
  using value_type = bool;
  using memory_type = Mem;
  using this_type = B<Mem>;

  template<class M>
  using this_type2 = B<M>;

  using local_type = this_type2<battery::local_memory>;

  constexpr static const bool sequential = Mem::sequential;
  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool complemented = true;
  constexpr static const char* name = "B";

private:
  using atomic_type = memory_type::template atomic_type<value_type>;
  atomic_type val;

public:
  CUDA static constexpr local_type bot() { return false; }
  CUDA static constexpr local_type top() { return true; }
  CUDA constexpr B(): val(false) {}
  CUDA constexpr B(value_type x): val(x) {}
  CUDA constexpr B(const this_type& other): B(other.value()) {}
  constexpr B(this_type&& other) = default;

  template <class M>
  CUDA constexpr B(const this_type2<M>& other): B(other.value()) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive.
   * @sequential @order-preserving */
  template <class M>
  CUDA constexpr this_type& operator=(const this_type2<M>& other) {
   if constexpr(sequential) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      static_assert(sequential, "The operator= in `B` can only be used when the underlying memory is `sequential`.");
    }
  }

  /** @sequential @order-preserving */
  CUDA constexpr this_type& operator=(const this_type& other) {
    if constexpr(sequential) {
      memory_type::store(val, other.value());
      return *this;
    }
    else {
      static_assert(sequential, "The operator= in `B` can only be used when the underlying memory is `sequential`.");
    }
  }

  CUDA constexpr this_type& operator=(bool other) {
    return operator=(local::B(other));
  }

  /** @parallel */
  CUDA constexpr value_type value() const { return memory_type::load(val); }

  /** @parallel */
  CUDA constexpr operator value_type() const { return value(); }

  /** `true` whenever \f$ a = \top \f$, `false` otherwise.
   * @parallel @order-preserving @increasing
  */
  CUDA constexpr B is_top() const {
    return value();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise.
   * @parallel @order-preserving @decreasing
   */
  CUDA constexpr B is_bot() const {
    return !value();
  }

  /** @parallel @order-preserving @increasing */
  CUDA constexpr void join_top() {
    memory_type::store(val, true);
  }

  /** @parallel @order-preserving @increasing
   * \return `true` if the value has changed, `false` otherwise.
   */
  template<class M1>
  CUDA constexpr bool join(const this_type2<M1>& other) {
    if(!value() && other.value()) {
      memory_type::store(val, true);
      return true;
    }
    return false;
  }

  /** @order-preserving @increasing */
  CUDA constexpr void meet_bot() {
    memory_type::store(val, false);
  }

  /** @order-preserving @increasing
   * \return `true` if the value has changed, `false` otherwise.
   */
  template<class M1>
  CUDA constexpr bool meet(const this_type2<M1>& other) {
    if(value() && !other.value()) {
      memory_type::store(val, false);
      return true;
    }
    return false;
  }

  CUDA constexpr bool join(value_type other) {
    return join(local::B(other));
  }

  CUDA constexpr bool meet(value_type other) {
    return meet(local::B(other));
  }

  /** Print the current element.
   * @sequential
  */
  CUDA NI void print() const {
    ::battery::print(value());
  }

  CUDA constexpr this_type& operator|= (const this_type& other) {
    join(other);
    return *this;
  }

  CUDA constexpr this_type& operator&= (const this_type& other) {
    meet(other);
    return *this;
  }

  CUDA constexpr this_type& operator|= (value_type other) {
    join(other);
    return *this;
  }

  CUDA constexpr this_type& operator&= (value_type other) {
    meet(other);
    return *this;
  }


  template<class Mem2>
  friend class B;
};

/** @parallel */
template<class M1, class M2>
CUDA constexpr bool operator<=(const B<M1>& a, const B<M2>& b) {
  return !a.value() || b.value();
}

template<class M1, class M2>
CUDA constexpr bool operator<(const B<M1>& a, const B<M2>& b) {
  return !a.value() && b.value();
}

template<class M1, class M2>
CUDA constexpr bool operator>=(const B<M1>& a, const B<M2>& b) {
  return b <= a;
}

template<class M1, class M2>
CUDA constexpr bool operator>(const B<M1>& a, const B<M2>& b) {
  return b < a;
}

template<class M1, class M2>
CUDA constexpr bool operator==(const B<M1>& a, const B<M2>& b) {
  return a.value() == b.value();
}

template<class M1, class M2>
CUDA constexpr bool operator!=(const B<M1>& a, const B<M2>& b) {
  return a.value() != b.value();
}

template<class M>
std::ostream& operator<<(std::ostream &s, const B<M> &a) {
  s << a.value();
  return s;
}

} // namespace lala

#endif
