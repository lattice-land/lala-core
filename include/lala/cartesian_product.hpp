// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_CARTESIAN_PRODUCT_HPP
#define LALA_CORE_CARTESIAN_PRODUCT_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "logic/logic.hpp"
#include "universes/primitive_upset.hpp"

namespace lala {

template<class... As>
class CartesianProduct;

namespace impl {
  template<class... As>
  CUDA constexpr auto index_sequence_of(const CartesianProduct<As...>&) {
    return std::index_sequence_for<As...>{};
  }

  template<class... As>
  CUDA constexpr auto index_sequence_of(const battery::tuple<As...>&) {
    return std::index_sequence_for<As...>{};
  }

  template <class A, class B>
  CUDA constexpr auto index_sequence_of(const A& a, const B& b) {
    static_assert(decltype(impl::index_sequence_of(a))::size() == decltype(impl::index_sequence_of(b))::size());
    return impl::index_sequence_of(a);
  }

  template<class... Os>
  CUDA constexpr typename CartesianProduct<Os...>::local_type make_cp(Os... os) {
    return typename CartesianProduct<Os...>::local_type(os...);
  }

  template<class A>
  struct is_product {
    static constexpr bool value = false;
  };

  template<class... As>
  struct is_product<CartesianProduct<As...>> {
    static constexpr bool value = true;
  };

  template<class... As>
  struct is_product<battery::tuple<As...>> {
    static constexpr bool value = true;
  };

  template<class A>
  inline static constexpr bool is_product_v = is_product<A>::value;
}

/** The Cartesian product abstract domain is a _domain transformer_ combining several abstract domains.
Concretization function: \f$ \gamma((a_1, \ldots, a_n)) = \sqcap_{i \leq n} \gamma_i(a_i) \f$. */
template<class... As>
class CartesianProduct {
public:
  template<size_t i>
  using type_of = typename battery::tuple_element<i, battery::tuple<As...>>::type;
  constexpr static size_t n = battery::tuple_size<battery::tuple<As...>>{};
  static_assert(n > 0, "CartesianProduct must not be empty.");
  using this_type = CartesianProduct<As...>;
  using local_type = CartesianProduct<typename As::local_type...>;
  using memory_type = typename type_of<0>::memory_type;

  template<class...Bs> friend class CartesianProduct;

  template<class F>
  using iresult = IResult<local_type, F>;

  using value_type = battery::tuple<typename As::value_type...>;

  constexpr static const bool is_abstract_universe = true;
  constexpr static const bool sequential = (... && As::sequential);
  constexpr static const char* name = "CartesianProduct";

private:
  battery::tuple<As...> val;

public:
  /** Initialize a Cartesian product to bottom using default constructors. */
  constexpr CartesianProduct() = default;
  CUDA constexpr CartesianProduct(const As&... as): val(battery::make_tuple(as...)) {}
  CUDA constexpr CartesianProduct(As&&... as): val(battery::make_tuple(std::forward<As>(as)...)) {}
  CUDA constexpr CartesianProduct(typename As::value_type... vs): val(battery::make_tuple(As(vs)...)) {}

  template<class... Bs>
  CUDA constexpr CartesianProduct(const CartesianProduct<Bs...>& other): val(other.val) {}

  template<class... Bs>
  CUDA constexpr CartesianProduct(CartesianProduct<Bs...>&& other): val(std::move(other.val)) {}

  /** The assignment operator can only be used in a sequential context.
   * It is monotone but not extensive. */
  template <class... Bs>
  CUDA constexpr this_type& operator=(const CartesianProduct<Bs...>& other) {
    if constexpr(sequential) {
      val = other.val;
      return *this;
    }
    else {
      static_assert(sequential, "operator= seq (CartesianProduct).");
    }
  }

  CUDA constexpr this_type& operator=(const this_type& other) {
    if constexpr(sequential) {
      val = other.val;
      return *this;
    }
    else {
      static_assert(sequential, "operator= seq (CartesianProduct).");
    }
  }

  /** Cartesian product initialized to \f$ (\bot_1, \ldots, \bot_n) \f$. */
  CUDA static constexpr local_type bot() {
    return local_type(As::bot()...);
  }

  /** Cartesian product initialized to \f$ (\top_1, \ldots, \top_n) \f$. */
  CUDA static constexpr local_type top() {
    return local_type(As::top()...);
  }

private:
  /** Interpret the formula in the component `i`. */
  template<bool is_tell, size_t i, class F, class Env>
  CUDA NI static iresult<F> interpret_one(const F& f, const Env& env) {
    auto one = is_tell ? type_of<i>::interpret_tell(f, env) :  type_of<i>::interpret_ask(f, env);
    if(one.has_value()) {
      auto res = bot();
      res.template project<i>().tell(one.value());
      return one.map(std::move(res));
    }
    else {
      auto res = iresult<F>(IError<F>(true, name, "The requested component of this Cartesian product cannot interpret this formula.", f));
      return res.join_errors(std::move(one));
    }
  }

  template<bool is_tell, size_t i = 0, class F, class Env>
  CUDA NI static IResult<bool, F> interpret_all(const F& f, local_type& res, bool empty, const Env& env) {
    if constexpr(i == n) {
      if(empty) {
        return IResult<bool, F>(IError<F>(true, name, "No component of this Cartesian product can interpret this formula.", f));
      }
      else {
        return IResult<bool, F>(true);
      }
    }
    else {
      auto one = is_tell ? type_of<i>::interpret_tell(f, env) : type_of<i>::interpret_ask(f, env);
      if(one.has_value()) {
        res.template project<i>().tell(one.value());
        return std::move(interpret_all<is_tell, i+1>(f, res, false, env).join_warnings(std::move(one)));
      }
      else {
        auto r = interpret_all<is_tell, i+1>(f, res, empty, env);
        if(!r.has_value()) {
          r.join_errors(std::move(one));
        }
        return std::move(r);
      }
    }
  }

  template<bool is_tell, class F, class Env>
  CUDA NI static iresult<F> interpret(const F& f, const Env& env) {
    local_type cp = bot();
    if(f.is(F::Seq) && f.sig() == AND) {
      iresult<F> res(bot());
      for(int i = 0; i < f.seq().size(); ++i) {
        auto r = interpret_all<is_tell>(f.seq(i), cp, true, env);
        if(!r.has_value()) {
          return std::move(r).template map_error<local_type>();
        }
        res.join_warnings(std::move(r));
      }
      return std::move(res).map(std::move(cp));
    }
    return interpret_all<is_tell>(f, cp, true, env).map(std::move(cp));
  }

public:
  template<size_t i, class F, class Env>
  CUDA static iresult<F> interpret_one_tell(const F& f, const Env& env) {
    return interpret_one<true, i>(f, env);
  }

  template<size_t i, class F, class Env>
  CUDA static iresult<F> interpret_one_ask(const F& f, const Env& env) {
    return interpret_one<false, i>(f, env);
  }

  /** Interpret the formula `f` in all sub-universes in which `f` is interpretable. */
  template<class F, class Env>
  CUDA static iresult<F> interpret_tell(const F& f, const Env& env) {
    return interpret<true>(f, env);
  }

  template<class F, class Env>
  CUDA static iresult<F> interpret_ask(const F& f, const Env& env) {
    return interpret<false>(f, env);
  }

private:
  // The non-const version must stay private, otherwise it violates the PCCP model since the caller might not check if the updated value is strictly greater w.r.t. lattice order.
  template<size_t i>
  CUDA constexpr type_of<i>& project() {
    return battery::get<i>(val);
  }

  template<size_t... I>
  CUDA constexpr value_type value_(std::index_sequence<I...>) const {
    return value_type(project<I>().value()...);
  }

  template<size_t... I>
  CUDA constexpr local::BInc is_top_(std::index_sequence<I...>) const {
    return (... || project<I>().is_top());
  }

  template<size_t... I>
  CUDA constexpr local::BDec is_bot_(std::index_sequence<I...>) const {
    return (... && project<I>().is_bot());
  }

public:
  template<size_t i>
  CUDA constexpr const type_of<i>& project() const {
    return battery::get<i>(val);
  }

  CUDA constexpr value_type value() const {
    return value_(std::index_sequence_for<As...>{});
  }

  /** `true` if \f$ \exists{j \geq i},~\gamma(a_j) = \top^\flat \f$, `false` otherwise. */
  CUDA constexpr local::BInc is_top() const {
    return is_top_(std::index_sequence_for<As...>{});
  }

  /** `true` if \f$ \forall{j \geq i},~\gamma(a_j) = \bot^\flat \f$, `false` otherwise. */
  CUDA constexpr local::BDec is_bot() const {
    return is_bot_(std::index_sequence_for<As...>{});
  }

private:
  template<size_t i = 0, class M, class... Bs>
  CUDA constexpr this_type& tell_(const CartesianProduct<Bs...>& other, BInc<M>& has_changed) {
    if constexpr (i < n) {
      project<i>().tell(other.template project<i>(), has_changed);
      return tell_<i+1>(other, has_changed);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0, class... Bs>
  CUDA constexpr this_type& tell_(const CartesianProduct<Bs...>& other) {
    if constexpr (i < n) {
      project<i>().tell(other.template project<i>());
      return tell_<i+1>(other);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0, class M, class... Bs>
  CUDA constexpr this_type& dtell_(const CartesianProduct<Bs...>& other, BInc<M>& has_changed) {
    if constexpr (i < n) {
      project<i>().dtell(other.template project<i>(), has_changed);
      return dtell_<i+1>(other, has_changed);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0, class... Bs>
  CUDA constexpr this_type& dtell_(const CartesianProduct<Bs...>& other) {
    if constexpr (i < n) {
      project<i>().dtell(other.template project<i>());
      return dtell_<i+1>(other);
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0>
  CUDA constexpr this_type& dtell_bot_() {
    if constexpr (i < n) {
      project<i>().dtell_bot();
      return dtell_bot_<i+1>();
    }
    else {
      return *this;
    }
  }

  template<size_t i = 0, class... Bs>
  CUDA constexpr bool extract_(CartesianProduct<Bs...>& ua) {
    if constexpr (i < n) {
      bool is_under = project<i>().extract(ua.template project<i>());
      return is_under && extract_<i+1>(ua);
    }
    else {
      return true;
    }
  }

public:
  /** \f$ \top \f$ is only told in the first component (since it is sufficient for the whole Cartesian product to become top as well).  */
  CUDA constexpr this_type& tell_top() {
    project<0>().tell_top();
    return *this;
  }

  template <class M, class... Bs>
  CUDA constexpr this_type& tell(const CartesianProduct<Bs...>& other, BInc<M>& has_changed) {
    return tell_(other, has_changed);
  }

  template<size_t i, class Ai, class M>
  CUDA constexpr this_type& tell(const Ai& a, BInc<M>& has_changed) {
    project<i>().tell(a, has_changed);
    return *this;
  }

  template <class... Bs>
  CUDA constexpr this_type& tell(const CartesianProduct<Bs...>& other) {
    return tell_(other);
  }

  template<size_t i, class Ai>
  CUDA constexpr this_type& tell(const Ai& a) {
    project<i>().tell(a);
    return *this;
  }

  CUDA constexpr this_type& dtell_bot() {
    dtell_bot_();
    return *this;
  }

  template <class M, class... Bs>
  CUDA constexpr this_type& dtell(const CartesianProduct<Bs...>& other, BInc<M>& has_changed) {
    return dtell_(other, has_changed);
  }

  template<size_t i, class Ai, class M>
  CUDA constexpr this_type& dtell(const Ai& a, BInc<M>& has_changed) {
    project<i>().dtell(a, has_changed);
    return *this;
  }

  template <class... Bs>
  CUDA constexpr this_type& dtell(const CartesianProduct<Bs...>& other) {
    return dtell_(other);
  }

  template<size_t i, class Ai>
  CUDA constexpr this_type& dtell(const Ai& a) {
    project<i>().dtell(a);
    return *this;
  }

  /** For correctness, the parameter `ua` must be stored in a local memory. */
  template <class... Bs>
  CUDA constexpr bool extract(CartesianProduct<Bs...>& ua) const {
    return extract_(ua);
  }

// Implementation of the logical signature.

private:
  template<Sig sig, class A, size_t... I>
  CUDA static constexpr auto fun_(const A& a, std::index_sequence<I...>)
  {
    return impl::make_cp((As::template fun<sig>(a.template project<I>()))...);
  }

  template<Sig sig, class A, class B, size_t... I>
  CUDA static constexpr auto fun_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return impl::make_cp((As::template fun<sig>(a.template project<I>(), b.template project<I>()))...);
  }

  template<Sig sig, class A, class B, size_t... I>
  CUDA static constexpr auto fun_left(const A& a, const B& b, std::index_sequence<I...>)
  {
    return impl::make_cp((As::template fun<sig>(a.template project<I>(), b))...);
  }

  template<Sig sig, class A, class B, size_t... I>
  CUDA static constexpr auto fun_right(const A& a, const B& b, std::index_sequence<I...>)
  {
    return impl::make_cp((As::template fun<sig>(a, b.template project<I>()))...);
  }

public:
  CUDA static constexpr bool is_supported_fun(Sig sig) {
    return (... && As::is_supported_fun(sig));
  }

  /** Given a product \f$ (x_1, \ldots, x_n) \f$, returns \f$ (f(x_1), \ldots, f(x_n)) \f$. */
  template<Sig sig, class... Bs>
  CUDA static constexpr auto fun(const CartesianProduct<Bs...>& a) {
    return fun_<sig>(a, impl::index_sequence_of(a));
  }

  /** Given two product \f$ (x_1, \ldots, x_n) \f$ and \f$ (y_1, \ldots, y_n) \f$, returns \f$ (f(x_1, y_1), \ldots, f(x_n, y_n)) \f$.
      If either the left or right operand is not a product, returns \f$ (f(x_1, c), \ldots, f(x_n, c)) \f$ or  \f$ (f(c, y_1), \ldots, f(c, y_n)) \f$. */
  template<Sig sig, class... As2, class... Bs>
  CUDA static constexpr auto fun(const CartesianProduct<As2...>& a, const CartesianProduct<Bs...>& b) {
    return fun_<sig>(a, b, impl::index_sequence_of(a, b));
  }

  template<Sig sig, class... As2, class B>
  CUDA static constexpr auto fun(const CartesianProduct<As2...>& a, const B& b) {
    return fun_left<sig>(a, b, impl::index_sequence_of(a));
  }

  template<Sig sig, class A, class... Bs>
  CUDA static constexpr auto fun(const A& a, const CartesianProduct<Bs...>& b) {
    return fun_right<sig>(a, b, impl::index_sequence_of(b));
  }

private:
  template<size_t i, class Env, class Allocator = typename Env::allocator_type>
  CUDA NI TFormula<Allocator> deinterpret_(AVar x,
    typename TFormula<Allocator>::Sequence& seq, const Env& env) const
  {
    if constexpr(i < n) {
      auto f = project<i>().deinterpret(x, env);
      if(!f.is_true()) {
        seq.push_back(project<i>().deinterpret(x, env));
      }
      return deinterpret_<i+1, Env>(x, seq, env);
    }
    else {
      if(seq.size() == 1) {
        return std::move(seq[0]);
      }
      else {
        return TFormula<Allocator>::make_nary(
          AND,
          std::move(seq));
      }
    }
  }

public:
  template<class Env>
  CUDA TFormula<typename Env::allocator_type> deinterpret(AVar x, const Env& env) const {
    using allocator_t = typename Env::allocator_type;
    typename TFormula<allocator_t>::Sequence seq(env.get_allocator());
    return deinterpret_<0, Env>(x, seq, env);
  }

private:
  template<size_t i = 0>
  CUDA NI void print_() const {
    if constexpr(i < n) {
      project<i>().print();
      if constexpr(i < n - 1) {
        printf("\n");
        print_<i+1>();
      }
    }
  }

public:
  CUDA void print() const {
    print_();
  }
};

/// Similar to `cp.template project<i>()`, just to avoid the ".template" syntax.
template<size_t i, class... As>
CUDA constexpr const typename CartesianProduct<As...>::template type_of<i>&
project(const CartesianProduct<As...>& cp) {
  return cp.template project<i>();
}

// Lattice operators
namespace impl {
  template<class A, class B, size_t... I>
  CUDA constexpr auto join_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return make_cp(join(project<I>(a), project<I>(b))...);
  }

  template<class A, class B, size_t... I>
  CUDA constexpr auto meet_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return make_cp(meet(project<I>(a), project<I>(b))...);
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool eq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return (... && (project<I>(a) == project<I>(b)));
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool neq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return (... || (project<I>(a) != project<I>(b)));
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool leq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return (... && (project<I>(a) <= project<I>(b)));
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool lt_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return leq_(a, b, std::index_sequence<I...>{}) && neq_(a, b, std::index_sequence<I...>{});
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool geq_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return (... && (project<I>(a) >= project<I>(b)));
  }

  template<class A, class B, size_t... I>
  CUDA constexpr bool gt_(const A& a, const B& b, std::index_sequence<I...>)
  {
    return geq_(a, b, std::index_sequence<I...>{}) && neq_(a, b, std::index_sequence<I...>{});
  }
}

/** \f$ (a_1, \ldots, a_n) \sqcup (b_1, \ldots, b_n) = (a_1 \sqcup_1 b_1, \ldots, a_n \sqcup_n b_n) \f$ */
template<class... As, class... Bs>
CUDA constexpr auto join(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::join_(a, b, impl::index_sequence_of(a, b));
}

/** \f$ (a_1, \ldots, a_n) \sqcap (b_1, \ldots, b_n) = (a_1 \sqcap_1 b_1, \ldots, a_n \sqcap_n b_n) \f$ */
template<class... As, class... Bs>
CUDA constexpr auto meet(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::meet_(a, b, impl::index_sequence_of(a, b));
}

/** \f$ (a_1, \ldots, a_n) \leq (b_1, \ldots, b_n) \f$ holds when \f$ \forall{i \leq n},~a_i \leq_i b_i \f$. */
template<class... As, class... Bs>
CUDA constexpr bool operator<=(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::leq_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator<(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::lt_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator>=(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::geq_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator>(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::gt_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator==(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::eq_(a, b, impl::index_sequence_of(a, b));
}

template<class... As, class... Bs>
CUDA constexpr bool operator!=(const CartesianProduct<As...>& a, const CartesianProduct<Bs...>& b)
{
  return impl::neq_(a, b, impl::index_sequence_of(a, b));
}

namespace impl {
  template<size_t i = 0, class... As>
  void std_print(std::ostream &s, const CartesianProduct<As...> &cp) {
    if constexpr(i < CartesianProduct<As...>::n) {
      s << project<i>(cp);
      if constexpr(i < CartesianProduct<As...>::n - 1) {
        s << ", ";
        std_print<i+1>(s, cp);
      }
    }
  }
}

template<class A, class... As>
std::ostream& operator<<(std::ostream &s, const CartesianProduct<A, As...> &cp) {
// There is a weird compilation bug with `template<class... As>` where the compiler tries to instantiate << with an empty sequence of templates.
// Forcing at least one template solves the problem.
  s << "(";
  impl::std_print<0>(s, cp);
  s << ")";
  return s;
}

} // namespace lala

#endif
