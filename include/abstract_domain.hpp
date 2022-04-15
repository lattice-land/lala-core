// Copyright 2021 Pierre Talbot

#ifndef ABSTRACT_DOMAIN_HPP
#define ABSTRACT_DOMAIN_HPP

#include "thrust/optional.h"
#include "utility.hpp"

namespace lala {

/** An abstract domain is an extension of an abstract universe to formula with multiple variables.
 * You should first read `AbstractUniverse`.
*/
template <typename T, typename A, typename U, typename Alloc>
class AbstractDomain {
public:
  /** `TellElement` is an intermediate representation between a logical formula (see `Formula`) and an abstract element.
  The design rational is that we want to avoid manipulating `Formula` during solving for efficiency purposes (since Formula contains dynamic arrays, string representation of operators, ...).
  Therefore, we can create the tell elements at the beginning, and adding them in the abstract element later on, when appropriate. */
  typedef T TellElement;

  /** Similar to `TellElement` but with the purpose to be used with the function `entailment`.
  The terminology, ask and tell, comes from concurrent constraint programming, where a tell operation adds constraints into a global and shared store, and an ask operation query the store to check if a constraint is entailed (can be deduced from what we already know).
  Here the store is the abstract element. */
  typedef A AskElement;

  /** The underlying representation of a single variable inside this abstract element, given by an abstract universe.
   For instance, it can be an interval or a set of values for instance. */
  typedef U Universe;

  /** The memory allocator used for creating new elements in this abstract domain. */
  typedef Alloc Allocator;

  typedef AbstractDomain<T,A,U,Alloc> this_type;

  /** \return The least element \f$\bot\f$ of this abstract domain. */
  CUDA /* static */ this_type* bot(AType uid = UNTYPED) const = 0;

  /** \return The largest element \f$\top\f$ of this abstract domain. */
  CUDA /* static */ this_type* top(AType uid = UNTYPED) const = 0;

  /** See `AbstractUniverse.interpret` and `TellElement`.
  \return An empty optional if the formula cannot be interpreted in the abstract domain, or if \f$\bot\f$ would be trivially returned in case of over-approximation (dually for \f$ \top \f$ and under-approximation).
  Otherwise, it returns the interpreted formula.
  The returned tell element must be joined later in the current abstract element `this` and not in another abstract element. */
  CUDA virtual thrust::optional<TellElement> interpret(const Formula& f) = 0;

  /** Similar to `interpret` but for the ask queries.
  A reasonable default implementation is `return interpret(UNDER, f)`, with `AskElement = TellElement`.
  If `f` is under-approximated, then \f$ entailment(f) \f$ will hold only if the solutions of `this` are included in the solution of `f`.
  See also `AskElement`. */
  CUDA virtual thrust::optional<AskElement> interpret_ask(const Formula& f) = 0;

  /** Compute \f$ a \sqcup b \f$ where \f$a\f$ (`this`) is the current element and \f$ b \f$ (`other`) another abstract element. */
  CUDA virtual this_type& join(const this_type& other) = 0;

  /** Compute \f$ a \sqcap b \f$, see also `join`. */
  CUDA virtual this_type& meet(const this_type& other) = 0;

  /** Similar to `join`, but in addition set `has_changed` to `true` if \f$ a \sqcup b \neq a \f$, that is, `this` has changed.
  Also it only performs a write operation into `this` if \f$ a \sqcup b \neq a \f$, which is an important property for convergence. */
  CUDA virtual this_type& tell(const TellElement& other, bool& has_changed) = 0;

  /** `refine` is an extensive function (\f$\forall{a \in A},~\mathit{refine}(a) \geq a \f$) refining an abstract element \f$a\f$.
  It can have additional properties such as being under- or over-approximating depending on the abstract domain.
  \return `true` if the abstract element has changed and `false` if we reached a fixed point. */
  CUDA virtual bool refine() = 0;

  /** The entailment, formally written \f$a \models \varphi\f$, is `true` whenever we can deduce a formula \f$\varphi\f$ from an abstract element \f$a\f$, i.e., \f$\gamma(a) \subseteq [\![\varphi]\!]\f$.
  Note that if it returns `false`, it can either mean \f$\lnot\varphi\f$ is entailed, or that we do not know yet if it is entailed or not.
  Therefore, to test for _disentailment_, you should ask if the negation of the formula is entailed. */
  CUDA virtual bool entailment(const AskElement& element) const = 0;

  /** The projection of term onto the underlying abstract universe `Universe`.
  A common example is to project the domain of a variable `x` or a term such as `x + y` onto an interval or set variable domain.
  If you want to project a formula onto a Boolean, you should use `entailment` instead. */
  CUDA virtual Universe project(const TellElement& x) const = 0;

  /** The function `embed(x, dom)` is similar to \f$ a \sqcup [\![\varphi]\!] \f$ where \f$\varphi\f$ is a formula with a single variable equals to \f$ x \f$ and interpretable in `Universe`.
   Here, the underlying element `Universe` has already been created. */
  CUDA virtual void embed(AVar x, const Universe& dom) = 0;

  /** See `AbstractUniverse.split`. */
  CUDA virtual battery::vector<TellElement, Allocator> split(/*const SearchStrategy& strat*/) const = 0;

  /** See `AbstractUniverse.reset`. */
  CUDA virtual void reset(const this_type& b) = 0;

  /** See `AbstractUniverse.clone`. */
  CUDA virtual this_type* clone() const = 0;

  /** This function is the inverse of `interpret`, but directly maps to a general `Formula`.
      Let \f$ a = [\![\varphi]\!]_A \f$, then we must have \f$ \gamma(a) = [\![[\![a]\!]^{-1}]\!]^\flat \f$. */
  CUDA virtual Formula deinterpret() const = 0;

  /** This function is similar to `deinterpret` but for a specific tell element, that is not necessarily in the abstract element yet. */
  CUDA virtual Formula deinterpret_tell(const TellElement& element) const = 0;

  /** This function is similar to `deinterpret` but for a specific ask element. */
  CUDA virtual Formula deinterpret_ask(const AskElement& element) const = 0;

  /** Print the current element with the logical name of the variables. */
  CUDA virtual void print() const = 0;
};

} // namespace lala

#endif