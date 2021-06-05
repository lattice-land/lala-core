// Copyright 2021 Pierre Talbot

#ifndef ABSTRACT_DOMAIN_HPP
#define ABSTRACT_DOMAIN_HPP

#include "thrust/optional.h"
#include "ecuda/ecuda.hpp"
#include "cuda_helper.hpp"

namespace lala {

/** An abstract domain is a [lattice](https://en.wikipedia.org/wiki/Lattice_(order)) with additional operations including an interpretation function, a refinement operator and a split operator.
We now explain the main idea of this abstract framework, although it is not possible to be very precise here, for more information on the theory, please contact Pierre Talbot (pierre.talbot@uni.lu).
We have three entities playing a role in our framework: logical formulas, abstract domains and the concrete domain.
The concrete domain is a mathematical object, possibly infinite and not computable, which is not explicitly represented in the code.
The concrete domain is however crucial to establish the proofs of soundness and completeness of our abstract domains.
We will write the concrete domain \f$ D^\flat \f$ and an abstract domain \f$ A \f$.
We can interpret a logical formula \f$ \varphi \f$ in the concrete domain with \f$[\![\varphi]\!]^\flat\f$, and this element contains all the solutions of \f$\varphi\f$.
We connect the concrete and abstract worlds using a monotone concretization function \f$\gamma: A \to D^\flat\f$ which turns an abstract element into a concrete element.
The concretization is useful to establish two properties.
Let \f$ \varphi \f$ be a formula and  \f$ a \in A \f$ an abstract element, then:
  - \f$a\f$ is an under-approximation if \f$\gamma(a) \subseteq [\![\varphi]\!]^\flat \f$ (soundness).
  - \f$a\f$ is an over-approximation if \f$\gamma(a) \supseteq [\![\varphi]\!]^\flat \f$ (completeness).
  - \f$a \f$ is an exact representation of \f$\varphi\f$ if it is both an under- and over-approximation.

In brief, an under-approximating element guarantees we have only solutions of \f$\varphi\f$ represented in the abstract element \f$a\f$, but not necessarily all.
Whereas an over-approximating element guarantees we have all solutions of \f$\varphi\f$ but possibly with extra non-solution elements.

This class exists only for documentation purposes as abstract domains will be combined using templates, and not by relying on inheritance from an abstract class.
Ideally, `AbstractDomain` should be a C++20 concept, but it is not yet possible because neither NVCC or Doxygen support concepts yet.
*/
template <typename LE, typename VD>
class AbstractDomain {
public:
  /** `LogicalElement` is an intermediate representation between a logical formula (see `Formula`) and an abstract element.
   The design rational is that we want to avoid manipulating `Formula` during solving for efficiency purposes (since Formula contains dynamic arrays, string representation of operators, ...).
   Therefore, we can create the logical elements at the beginning, and adding them in the abstract element later on, when appropriate. */
  typedef LE LogicalElement;

  /** A variable domain (`VarDom`) is the underlying representation of a single variable inside this abstract element.
   It can be an interval or a set of values for instance. */
  typedef VD VarDom;

  /** A partial interpretation function \f$[\![\varphi]\!]_a^\updownarrow \f$ turns the logical formula \f$\varphi\f$ (`f`) into a logical element according to the current abstract element \f$a\f$ and the approximation kind \f$\updownarrow\f$ (`appx`).
  See also `LogicalElement`.
  The approximation kind is not necessarily bound to an abstract element.
  For instance, let an abstract element \f$a\f$ under-approximating a logical formula \f$\varphi\f$.
  We can still add over-approximating _redundant constraints_ in \f$a\f$, which will not impact the under-approximating property of \f$a\f$ w.r.t. \f$\varphi\f$.
  \return An empty optional if the formula cannot be interpreted in the abstract domain. Otherwise, it returns the interpreted formula. */
  CUDA virtual thrust::optional<LogicalElement> interpret(Approx appx, const Formula& f) = 0;

  /** Compute \f$ a \sqcup [\![\varphi]\!] \f$ where \f$a\f$ is the current element and \f$ [\![\varphi]\!] \f$ an interpreted formula. */
  CUDA virtual void join(const LogicalElement& other) = 0;

  /** Compute \f$ a \sqcap [\![\varphi]\!] \f$, see also `join`. */
  CUDA virtual void meet(const LogicalElement& other) = 0;

  /** `refine` is an extensive function (\f$\forall{a \in A},~\mathit{refine}(a) \geq a \f$) refining an abstract element \f$a\f$.
  It can have additional properties such as under- or over-approximation depending on the abstract domain.
  \return `true` if the abstract element has changed and `false` if we reached a fixed point. */
  CUDA virtual bool refine() = 0;

  /** The entailment, formally written \f$a \models \varphi\f$, is `true` whenever we can deduce a formula \f$\varphi\f$ from an abstract element \f$a\f$, i.e., \f$\gamma(a) \subseteq [\![\varphi]\!]\f$.
  Note that if it returns `false`, it can either mean \f$\lnot\varphi\f$ is entailed, or that we do not know yet if it is entailed or not.
  Therefore, to test for _disentailment_, you should ask if the negation of the formula is entailed. */
  CUDA virtual bool entailment(const LogicalElement& element) const = 0;

  /** The projection of term onto the underlying variable domain `VarDom`.
  A common example is to project the domain of a variable `x` or a term such as `x + y` onto an interval or set variable domain.
  If you want to project a formula onto a Boolean, you should use `entailment` instead. */
  CUDA virtual VarDom project(const LogicalElement& x) const = 0;

  /** The function `embed(x, dom)` is similar to \f$ a \sqcup [\![\varphi]\!] \f$ where \f$\varphi\f$ is a formula with a single variable equals to \f$ x \f$ and interpretable in `VarDom`.
   Here, the underlying element `VarDom` has already been created. */
  CUDA virtual void embed(AVar x, const VarDom& dom) const = 0;

  /** `split` is an extensive function, i.e., \f$ \forall{a \in A},~\forall{b \in \mathit{split}(a)},~a \leq b \f$, that divides an abstract element into a set of subelements.
  We call _unsplittable elements_ the elements such that \f$\mathit{split}(a) \f$ is a singleton.
  We require \f$\mathit{split}(a) = \{a\} \f$ for all unsplittable elements \f$a \in A \f$.
  An additional usage of `split` is to detect unsatisfiability of over-approximation, and satisfiability of under-approximation:
    - In case of an over-approximating element \f$a\f$, we have \f$\mathit{split}(a) = \{\} \Rightarrow \gamma(a) = \{\} \f$.
    - In case of an under-approximating element \f$a\f$, we have \f$\mathit{split}(a) \neq \{\} \Rightarrow \gamma(a) \neq \{\} \land \gamma(a) \subseteq [\![\varphi]\!]^\flat\f$.

  \return A list of logical elements (possibly complementary, but not necessarily) that can be joined in an abstract element to further refine its state.
  */
  CUDA virtual ecuda::vector<LogicalElement> split(/*const SearchStrategy& strat*/) const = 0;

  /** An abstract domain can be _eventually under-approximating_ which means that after a sufficient number of split and refine operations, it always reach an under-approximating element.
   \return `true` if \f$\gamma(a) \subseteq [\![\varphi]\!]^\flat\f$. `false` is returned whenever `a` is an over-approximation or if we do not know whether `a` is an under-approximation. */
  CUDA virtual bool is_underappx() const = 0;

  /** This method resets the current abstract element to an anterior state \f$b \f$.
      Therefore, this operation is similar to computing \f$ a \sqcap b \f$ where \f$ a \geq b \f$. */
  CUDA virtual void reset(const AbstractDomain& b) = 0;

  /** \return A copy of the current abstract element. */
  CUDA virtual AbstractDomain clone() const = 0;

  /** This function is the inverse of `interpret`, but directly maps to a general `Formula`.
      Let \f$ a = [\![\varphi]\!]_A \f$, then we must have \f$ \gamma(a) = [\![[\![a]\!]^{-1}]\!]^\flat \f$. */
  CUDA virtual Formula deinterpret() const = 0;

  /** Print the current element with the logical name of the variables. */
  CUDA virtual void print() const = 0;
};

} // namespace lala

#endif