// Copyright 2021 Pierre Talbot

#ifndef ABSTRACT_UNIVERSE_HPP
#define ABSTRACT_UNIVERSE_HPP

#include "thrust/optional.h"
#include "darray.hpp"
#include "utility.hpp"

namespace lala {

/** An abstract universe is a [lattice](https://en.wikipedia.org/wiki/Lattice_(order)) with additional operations including an interpretation function from a logical statement to an element of the lattice.
We now explain the main idea of this abstract framework, although it is not possible to be very precise here, for more information on the theory, please contact Pierre Talbot (pierre.talbot@uni.lu).
We have three entities playing a role in our framework: logical formulas, abstract universes (and their extensions: abstract domains) and the concrete domain.
The concrete domain is a mathematical object, possibly infinite and not computable, which is not explicitly represented in the code.
The concrete domain is however crucial to establish the proofs of soundness and completeness of our abstract domains.
We will write the concrete domain \f$ D^\flat \f$ and an abstract universe \f$ A_x \f$.
An abstract universe represents the values, taken in a universe of discourse, of a single variable \f$ x \f$.
Hence, the interpretation of a formula \f$ \varphi \f$ is only possible if \f$ \varphi \f$ has a single variable.
See the concept of abstract domain, extending the one of abstract universe, for multiple variables.
Formally, we write \f$ A_x \f$ an abstract universe with \f$ x \f$ the name of the variable; but in practice, we will not explicitly represent \f$ x \f$.
We can interpret a logical formula \f$ \varphi \f$ in the concrete domain with \f$[\![\varphi]\!]^\flat\f$, and this element contains all the solutions of \f$\varphi\f$.

We connect the concrete and abstract worlds using a monotone concretization function \f$\gamma: A_x \to D^\flat\f$ which turns an abstract element into a concrete element.
The concretization is useful to establish two properties.
Let \f$ \varphi \f$ be a formula and  \f$ a \in A_x \f$ an abstract element, then:
  - \f$a\f$ is an under-approximation if \f$\gamma(a) \subseteq [\![\varphi]\!]^\flat \f$ (soundness).
  - \f$a\f$ is an over-approximation if \f$\gamma(a) \supseteq [\![\varphi]\!]^\flat \f$ (completeness).
  - \f$a \f$ is an exact representation of \f$\varphi\f$ if it is both an under- and over-approximation.

In brief, soundness guarantees we have only solutions of \f$\varphi\f$ represented in the abstract element \f$a\f$, but not necessarily all.
Whereas completeness guarantees we have all solutions of \f$\varphi\f$ but possibly with extra non-solution elements.

This class exists only for documentation purposes as abstract universes will be combined using templates, and not by relying on inheritance from an abstract class.
Ideally, `AbstractUniverse` and `AbstractDomain` should be C++20 concepts, but it is not yet possible because neither NVCC or Doxygen support concepts yet.
*/
template <typename T, typename A, typename Alloc>
class AbstractUniverse {
public:
  /** The memory allocator used for creating new elements in this abstract universe. */
  typedef Alloc Allocator;

  typedef AbstractUniverse<T,A,Alloc> this_type;

  /** \return The smallest element of this abstract universe, formally written \f$\bot\f$. */
  CUDA static this_type bot() const { return this_type(); }

  /** \return The largest element of this abstract universe, formally written \f$\top\f$. */
  CUDA static this_type top() const { return this_type(); }

  /** A partial interpretation function \f$[\![\varphi]\!]_a^{\mathit{appx}} \f$ turns the logical formula \f$\varphi\f$ (`f`) into an abstract element according to the approximation kind \f$\mathit{appx}\f$.
  The approximation kind is not necessarily bound to an abstract element.
  For instance, let an abstract element \f$a\f$ under-approximating a logical formula \f$\varphi\f$.
  We can still add over-approximating _redundant constraints_ in \f$a\f$, which will not impact the under-approximating property of \f$a\f$ w.r.t. \f$\varphi\f$.
  \return An empty optional if the formula cannot be interpreted in the abstract universe, or if \f$\bot\f$ would be trivially returned in case of over-approximation (dually for \f$ \top \f$ and under-approximation).
  Otherwise, it returns the interpreted formula. */
  CUDA static thrust::optional<this_type> interpret(Approx appx, const Formula& f) { return {}; }

  /** Compute \f$ a \sqcup b \f$ where \f$a\f$ (`this`) is the current element and \f$ b \f$ another element. */
  CUDA virtual this_type& join(const this_type& b) = 0;

  /** Compute \f$ a \sqcap b \f$, see also `join`. */
  CUDA virtual this_type& meet(const this_type& b) = 0;

  /** \return `true` if \f$ a \leq b \f$, and `false` otherwise. */
  CUDA virtual bool order(const this_type& b) const = 0;

  /** `split` is an extensive function, i.e., \f$ \forall{a \in A},~\forall{b \in \mathit{split}(a)},~a \leq b \f$, that divides an abstract element into a set of subelements.
  We call _unsplittable elements_ the elements such that \f$\mathit{split}(a) \f$ is a singleton.
  We require \f$\mathit{split}(a) = \{a\} \f$ for all unsplittable elements \f$a \in A \f$.
  An additional usage of `split` is to detect unsatisfiability of over-approximation, and satisfiability of under-approximation:
    - In case of an over-approximating element \f$a\f$, we have \f$\mathit{split}(a) = \{\} \Rightarrow \gamma(a) = \{\} \f$.
    - In case of an under-approximating element \f$a\f$, we have \f$\mathit{split}(a) \neq \{\} \Rightarrow \gamma(a) \neq \{\} \land \gamma(a) \subseteq [\![\varphi]\!]^\flat\f$.

  For the special case of _eventually under-approximating_ abstract domain, \f$a\f$ is an under-approximation whenever \f$\mathit{split}(a) = \{a\}\f$.

  \return A list of abstract elements (possibly complementary, but not necessarily) that can be joined in an abstract element to further refine its state.
  */
  CUDA virtual DArray<this_type, Allocator> split(const Alloc& allocator = Alloc()) const = 0;

  /** This method resets the current abstract element to an anterior state \f$b \f$.
      Therefore, this operation is similar to computing \f$ a \sqcap b \f$ where \f$ a \geq b \f$. */
  CUDA virtual void reset(const this_type& b) = 0;

  /** \return A copy of the current abstract element. */
  CUDA virtual this_type* clone() const = 0;

  /** This function is the inverse of `interpret`, and directly maps to a `Formula`.
      Let \f$ a = [\![\varphi]\!]_A \f$, then we must have \f$ \gamma(a) = [\![[\![a]\!]^{-1}]\!]^\flat \f$.
      `x` is the name of the variable represented by this abstract universe. */
  CUDA virtual Formula deinterpret(AVar x) const = 0;

  /** Print the current element with the logical name `x` of the variable. */
  CUDA virtual void print(const LVar<Allocator>& x) const = 0;
};

} // namespace lala

#endif