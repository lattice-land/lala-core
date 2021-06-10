// Copyright 2021 Pierre Talbot

#ifndef AST_HPP
#define AST_HPP

#include "utility.hpp"

namespace lala {

/** Each abstract domain is uniquely identified by an UID. */
typedef int AD_UID;
/** This value means a formula is not typed in a particular abstract domain and its type should be inferred. */
#define UNTYPED_AD (-1)

/** A "logical variable" is just the name of the variable. */
typedef char* LVar;

/** We call an "abstract variable" the representation of this variable in an abstract domain.
It is simply an integer containing the UID of the abstract element and an internal integer variable identifier proper to the abstract domain.
The mapping between logical variables and abstract variables is maintained in `Environment` below.
An abstract variable always has a single name (or no name if it is not explicitly represented in the initial formula).
However, a logical variable can be represented by several abstract variables when the variable occurs in different domains. */
typedef int AVar;
#define AID(v) (v & ((1 << 8) - 1))
#define VID(v) (v >> 8)

CUDA AVar make_var(int ad_uid, int var_id);

/** The approximation of a formula in an abstract domain w.r.t. the concrete domain.
* `UNDER`: An under-approximating element contains only solutions but not necessarily all.
* `OVER`: An over-approximating element contains all solutions but not necessarily only solutions.
* `EXACT`: An exact element is both under- and over-approximating; it exactly represents the set of solutions. */
enum Approx {
  UNDER,
  OVER,
  EXACT
};

/** We represent everything at the same level (terms, formula, predicate, variable, constant).
This is general convenient when modelling to avoid creating intermediate boolean variables when reifying.
We can have `x + (x > y \/ y > x + 4)` and this expression is true if the value is != 0. */
struct Formula {
  AD_UID ad_uid;

  enum Type {
    ///@{
    LONG, DOUBLE,                    ///< Constant in the domain of discourse that can be represented exactly.
    ///@}
    AVAR,                          ///< Abstract variable
    ///@{
    ADD, SUB, MUL, DIV, MOD, POW,  ///< Terms
    ///@}
    ///@{
    EQ, LEQ, GEQ, NEQ, GT, LT,     ///< Predicates
    ///@}
    ///@{
    TRUE, FALSE, AND, OR, IMPLY, EQUIV, NOT,    ///< Formulas
    ///@}
    RAW                            ///< General tag for extension purposes.
  };

  Type tag;

  /** The name of the variable, term, function or predicate is represented by a string.
  This struct can also be used for representing constant such as real numbers.
  NOTE: We sometimes cannot use a float type because it cannot represent all real numbers, it is up to the abstract domain to under- or over-approximate it, or choose an exact representation such as rational. */
  struct Raw {
    char* name;
    Formula* children;
    size_t n;
  };

  union {
    long long int i;    // LONG
    double f;           // DOUBLE
    AVar v;   // AVAR
    struct {  // ADD, SUB, ..., EQ, ..., AND, .., NOT
      Formula* children;
      size_t n;
    };
    Raw raw;  // LVar, global constraints, predicates, real numbers, ...
  };
};

#define SHAPE(f,a,b,c) (f.tag == (a) && f.children[0].tag == (b) && f.children[1].tag == (c))

template<typename Allocator>
Formula make_x_op_i(Allocator& allocator, Formula::Type op, AVar x, long long int i) {
  Formula* children = new(allocator) Formula[2];
  children[0].tag = Formula::AVAR;
  children[0].v = x;
  children[1].tag = Formula::LONG;
  children[1].i = i;
  Formula f;
  f.ad_uid = UNTYPED_AD;
  f.tag = op;
  f.children = children;
  f.n = 2;
  return f;
}

struct SolveMode {
  enum {
    MINIMIZE,
    MAXIMIZE,
    SATISFY
  } tag;

  union {
    LVar lv;  ///< The logical variable to optimize.
    AVar av;  ///< The abstract variable to optimize. (We use this one, whenever the variable has been added to an abstract element).
    int num_sols; ///< How many solutions should we compute (SATISFY mode).
  };
};

/** An environment is a formula with an optimization mode and the mapping between logical variables and abstract variables. */
struct Environment {
  SolveMode mode;
  Formula formula;
  /** Given an abstract variable `v`, `avar2lvar[AD_UID(v)][VAR_ID(v)]` is the name of the variable. */
  struct VarArray {
    LVar* data;
    size_t n;
  };
  VarArray* avar2lvar;
  size_t n;
};

} // namespace lala
#endif
