// Copyright 2026 Pierre Talbot

#ifndef LALA_CORE_WELLFORMED_HPP
#define LALA_CORE_WELLFORMED_HPP

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include "ast.hpp"
#include "env.hpp"

/** This file checks that a formula is well-formed with respect to the sorted signature `Sig`.
 * The sort of each variable is given by an existential quantifier occurring in the formula, or by the environment.
 * There is no implicit coercion between sorts: Booleans, integers and real numbers must be converted explicitly (`BOOL2INT`, `INT2BOOL`, `INT2REAL`, `FLOOR`, `CEIL`, `TRUNC`).
 * Polymorphic symbols (`EQ`, `LEQ`, ..., set symbols and `ITE`) only require their arguments to have the same sort; they are not resolved to monomorphic symbols.
 * Note that the ternary form produced by `ternarize` is not well-formed in general (Booleans are integers in {0,1} there).
 */

namespace lala {

/** Explain why a formula is not well-formed: `subformula` is the smallest subformula exhibiting the problem. */
template <class Allocator>
struct WFDiagnostic {
  TFormula<Allocator> subformula;
  std::string message;

  void print() const {
    printf("%% [error] Ill-formed formula.\n");
    printf("%%   Formula: ");
    subformula.print();
    printf("\n%%   Description: %s\n", message.c_str());
  }
};

template <class Allocator>
std::string string_of_sort(const Sort<Allocator>& sort) {
  switch(sort.tag) {
    case Sort<Allocator>::Bool: return "Bool";
    case Sort<Allocator>::Int: return "Int";
    case Sort<Allocator>::Real: return "Real";
    case Sort<Allocator>::Set: return "Set(" + string_of_sort(*sort.sub) + ")";
  }
  return "<unknown sort>";
}

namespace impl {

template <class F, class Env>
class WellformedChecker {
public:
  using allocator_type = typename F::allocator_type;
  using sort_type = Sort<allocator_type>;
  using diagnostic_type = WFDiagnostic<allocator_type>;

private:
  /** The sort of a term: either a known sort, or the sort `Set(T)` for an unknown `T` (e.g., the empty set). */
  struct SortInfo {
    std::optional<sort_type> sort;
    bool any_set = false;

    SortInfo() = default;
    SortInfo(const sort_type& s): sort(s) {}
    static SortInfo make_any_set() { SortInfo i; i.any_set = true; return i; }
    bool is(typename sort_type::Tag tag) const { return sort.has_value() && sort->tag == tag; }
    bool is_set() const { return any_set || is(sort_type::Set); }
    std::string to_string() const { return sort.has_value() ? string_of_sort(*sort) : "Set(?)"; }
  };

  const Env& env;
  /** The sorts of the variables declared in the formula, by the identifier of their symbol. */
  std::unordered_map<int, sort_type> scope;
  diagnostic_type* diag;

  bool error(const F& f, const std::string& msg) {
    if(diag != nullptr) {
      diag->subformula = f;
      diag->message = msg;
    }
    return false;
  }

  static std::string name_of(const F& f) {
    return f.is(F::Seq) ? std::string(string_of_sig_txt(f.sig())) : std::string("<extended symbol>");
  }

  static sort_type make_sort(typename sort_type::Tag tag) { return sort_type(tag); }

  /** \return The name of the conversion from `from` to `to`, if any. */
  static std::string conversion_hint(const SortInfo& from, typename sort_type::Tag to) {
    if(from.is(sort_type::Bool) && to == sort_type::Int) { return " (use `bool2int` to convert a Boolean into an integer)"; }
    if(from.is(sort_type::Int) && to == sort_type::Bool) { return " (use `int2bool` to convert an integer into a Boolean)"; }
    if(from.is(sort_type::Int) && to == sort_type::Real) { return " (use `int2real` to convert an integer into a real number)"; }
    if(from.is(sort_type::Bool) && to == sort_type::Real) { return " (use `int2real(bool2int(...))` to convert a Boolean into a real number)"; }
    if(from.is(sort_type::Real) && to == sort_type::Int) { return " (use `floor`, `ceil` or `trunc` to convert a real number into an integer)"; }
    return "";
  }

  static std::string tag_name(typename sort_type::Tag tag) {
    switch(tag) {
      case sort_type::Bool: return "Bool";
      case sort_type::Int: return "Int";
      case sort_type::Real: return "Real";
      default: return "Set";
    }
  }

  /** Two sorts are compatible if they are equal, or if one is `Set(?)` and the other is a set. */
  static bool compatible(const SortInfo& a, const SortInfo& b) {
    if(a.sort.has_value() && b.sort.has_value()) {
      return *a.sort == *b.sort;
    }
    return a.is_set() && b.is_set();
  }

  /** The most precise of two compatible sorts. */
  static SortInfo meet(const SortInfo& a, const SortInfo& b) {
    return a.sort.has_value() ? a : b;
  }

  bool check_arity(const F& f, int expected) {
    int n = static_cast<int>(f.seq().size());
    if(n != expected) {
      return error(f, "`" + name_of(f) + "` expects " + std::to_string(expected) + " argument(s), but got " + std::to_string(n) + ".");
    }
    return true;
  }

  bool check_min_arity(const F& f, int expected) {
    int n = static_cast<int>(f.seq().size());
    if(n < expected) {
      return error(f, "`" + name_of(f) + "` expects at least " + std::to_string(expected) + " argument(s), but got " + std::to_string(n) + ".");
    }
    return true;
  }

  bool check_arg(const F& f, int i, const SortInfo& actual, typename sort_type::Tag expected) {
    if(actual.is(expected)) {
      return true;
    }
    return error(f, "`" + name_of(f) + "` expects an argument of sort " + tag_name(expected) + ", but argument "
      + std::to_string(i+1) + " has sort " + actual.to_string() + conversion_hint(actual, expected) + ".");
  }

  bool check_all_args(const F& f, const std::vector<SortInfo>& args, typename sort_type::Tag expected) {
    for(int i = 0; i < args.size(); ++i) {
      if(!check_arg(f, i, args[i], expected)) {
        return false;
      }
    }
    return true;
  }

  bool check_set_arg(const F& f, int i, const SortInfo& actual) {
    if(actual.is_set()) {
      return true;
    }
    return error(f, "`" + name_of(f) + "` expects an argument of sort Set, but argument " + std::to_string(i+1) + " has sort " + actual.to_string() + ".");
  }

  /** All the arguments in `[from, args.size())` must have the same sort. */
  std::optional<SortInfo> check_same_sort(const F& f, const std::vector<SortInfo>& args, int from = 0) {
    SortInfo res = args[from];
    for(int i = from + 1; i < args.size(); ++i) {
      if(!compatible(res, args[i])) {
        error(f, "`" + name_of(f) + "` expects arguments of the same sort, but argument " + std::to_string(from+1) + " has sort "
          + res.to_string() + " and argument " + std::to_string(i+1) + " has sort " + args[i].to_string() + ".");
        return {};
      }
      res = meet(res, args[i]);
    }
    return res;
  }

  bool declare(const F& f) {
    const auto& [lvar, sort] = f.exists();
    const std::string& name = lvar.str();
    sort_type s(sort);
    auto it = scope.find(lvar.id());
    if(it != scope.end()) {
      if(!(it->second == s)) {
        return error(f, "The variable `" + name + "` is declared with the sorts " + string_of_sort(it->second) + " and " + string_of_sort(s) + ".");
      }
      return true;
    }
    auto var = env.variable_of(lvar);
    if(var.has_value() && !(sort_type(var->get().sort) == s)) {
      return error(f, "The variable `" + name + "` is declared with the sort " + string_of_sort(s) + " but has the sort " + string_of_sort(sort_type(var->get().sort)) + " in the environment.");
    }
    scope.emplace(lvar.id(), s);
    return true;
  }

public:
  WellformedChecker(const Env& env, diagnostic_type* diag): env(env), diag(diag) {}

  /** First pass: collect the existentially quantified variables of `f`. */
  bool collect_existentials(const F& f) {
    switch(f.index()) {
      case F::E: return declare(f);
      case F::Seq: {
        for(int i = 0; i < f.seq().size(); ++i) {
          if(!collect_existentials(f.seq(i))) {
            return false;
          }
        }
        return true;
      }
      default: return true;
    }
  }

  /** Second pass: infer the sort of `f` and check the sorts of all its subformulas.
   * \return The sort of `f` or `{}` if `f` is not well-formed. */
  std::optional<SortInfo> infer(const F& f) {
    switch(f.index()) {
      case F::B: return SortInfo(make_sort(sort_type::Bool));
      case F::Z: return SortInfo(make_sort(sort_type::Int));
      case F::R: return SortInfo(make_sort(sort_type::Real));
      case F::S: return infer_set(f);
      case F::LV: {
        auto it = scope.find(f.lv().id());
        if(it != scope.end()) {
          return SortInfo(it->second);
        }
        auto var = env.variable_of(f.lv());
        if(var.has_value()) {
          return SortInfo(sort_type(var->get().sort));
        }
        error(f, "Undeclared variable `" + f.lv().str() + "`.");
        return {};
      }
      case F::E: return SortInfo(make_sort(sort_type::Bool));
      case F::ESeq: return SortInfo(make_sort(sort_type::Bool)); // Extended symbols (e.g., search annotations) are opaque.
      case F::Seq: return infer_seq(f);
      default:
        error(f, "Unknown kind of formula.");
        return {};
    }
  }

  /** Check that `f` is a formula, i.e., a well-formed term of sort `Bool`.
   * For instance, the integer term `bool2int(b)` is not a formula, hence it cannot occur at toplevel. */
  bool check_formula(const F& f) {
    auto s = infer(f);
    if(!s.has_value()) {
      return false;
    }
    if(!s->is(sort_type::Bool)) {
      return error(f, "A formula must have the sort Bool, but this term has sort " + s->to_string() + conversion_hint(*s, sort_type::Bool) + ".");
    }
    return true;
  }

private:
  std::optional<SortInfo> infer_set(const F& f) {
    const auto& set = f.s();
    if(set.size() == 0) {
      return SortInfo::make_any_set();
    }
    std::optional<SortInfo> elem;
    for(int i = 0; i < set.size(); ++i) {
      for(const F* bound : {&battery::get<0>(set[i]), &battery::get<1>(set[i])}) {
        auto s = infer(*bound);
        if(!s.has_value()) { return {}; }
        if(!elem.has_value()) {
          elem = std::move(s);
        }
        else if(!compatible(*elem, *s)) {
          error(f, "The elements of a set must have the same sort, but we found the sorts " + elem->to_string() + " and " + s->to_string() + ".");
          return {};
        }
        else {
          elem = meet(*elem, *s);
        }
      }
    }
    if(!elem->sort.has_value()) {
      return SortInfo::make_any_set(); // Set of empty sets: we do not track nested unknown sorts.
    }
    return SortInfo(sort_type(sort_type::Set, sort_type(*elem->sort), f.s().get_allocator()));
  }

  std::optional<SortInfo> infer_seq(const F& f) {
    std::vector<SortInfo> args;
    args.reserve(f.seq().size());
    for(int i = 0; i < f.seq().size(); ++i) {
      auto s = infer(f.seq(i));
      if(!s.has_value()) { return {}; }
      args.push_back(std::move(*s));
    }
    const auto Bool = sort_type::Bool;
    const auto Int = sort_type::Int;
    const auto Real = sort_type::Real;
    Sig sig = f.sig();
    auto result = [&](typename sort_type::Tag tag) { return std::optional<SortInfo>(SortInfo(make_sort(tag))); };
    switch(sig) {
      case ZNEG: case ZABS:
        if(!check_arity(f, 1) || !check_all_args(f, args, Int)) { return {}; }
        return result(Int);
      case ZSUB: case ZPOW:
      case ZTDIV: case ZTMOD: case ZFDIV: case ZFMOD: case ZCDIV: case ZCMOD: case ZEDIV: case ZEMOD:
        if(!check_arity(f, 2) || !check_all_args(f, args, Int)) { return {}; }
        return result(Int);
      case ZADD: case ZMUL: case ZMIN: case ZMAX:
        if(!check_min_arity(f, 1) || !check_all_args(f, args, Int)) { return {}; }
        return result(Int);
      case RNEG: case RABS: case RSQRT: case REXP: case RLN:
      case RSIN: case RCOS: case RTAN: case RASIN: case RACOS: case RATAN:
      case RSINH: case RCOSH: case RTANH: case RASINH: case RACOSH: case RATANH:
        if(!check_arity(f, 1) || !check_all_args(f, args, Real)) { return {}; }
        return result(Real);
      case RSUB: case RDIV: case RPOW: case RLOG:
        if(!check_arity(f, 2) || !check_all_args(f, args, Real)) { return {}; }
        return result(Real);
      case RNROOT:
        if(!check_arity(f, 2) || !check_arg(f, 0, args[0], Real) || !check_arg(f, 1, args[1], Int)) { return {}; }
        return result(Real);
      case RADD: case RMUL: case RMIN: case RMAX:
        if(!check_min_arity(f, 1) || !check_all_args(f, args, Real)) { return {}; }
        return result(Real);
      case FLOOR: case CEIL: case TRUNC:
        if(!check_arity(f, 1) || !check_all_args(f, args, Real)) { return {}; }
        return result(Int);
      case INT2REAL:
        if(!check_arity(f, 1) || !check_all_args(f, args, Int)) { return {}; }
        return result(Real);
      case BOOL2INT:
        if(!check_arity(f, 1) || !check_all_args(f, args, Bool)) { return {}; }
        return result(Int);
      case INT2BOOL:
        if(!check_arity(f, 1) || !check_all_args(f, args, Int)) { return {}; }
        return result(Bool);
      case ZEQ: case ZNEQ: case ZLEQ: case ZGEQ: case ZLT: case ZGT:
        if(!check_arity(f, 2) || !check_all_args(f, args, Int)) { return {}; }
        return result(Bool);
      case REQ: case RNEQ: case RLEQ: case RGEQ: case RLT: case RGT:
        if(!check_arity(f, 2) || !check_all_args(f, args, Real)) { return {}; }
        return result(Bool);
      case EQ: case NEQ: case LEQ: case GEQ: case LT: case GT:
        if(!check_arity(f, 2) || !check_same_sort(f, args).has_value()) { return {}; }
        return result(Bool);
      case SUBSET: case SUBSETEQ: case SUPSET: case SUPSETEQ:
        if(!check_arity(f, 2) || !check_set_arg(f, 0, args[0]) || !check_set_arg(f, 1, args[1]) || !check_same_sort(f, args).has_value()) { return {}; }
        return result(Bool);
      case UNION: case INTERSECTION: case DIFFERENCE: case SYMMETRIC_DIFFERENCE: {
        if((sig == UNION || sig == INTERSECTION) ? !check_min_arity(f, 1) : !check_arity(f, 2)) { return {}; }
        for(int i = 0; i < args.size(); ++i) {
          if(!check_set_arg(f, i, args[i])) { return {}; }
        }
        return check_same_sort(f, args);
      }
      case COMPLEMENT: case HULL:
        if(!check_arity(f, 1) || !check_set_arg(f, 0, args[0])) { return {}; }
        return args[0];
      case CONVEX:
        if(!check_arity(f, 1) || !check_set_arg(f, 0, args[0])) { return {}; }
        return result(Bool);
      case CARD:
        if(!check_arity(f, 1) || !check_set_arg(f, 0, args[0])) { return {}; }
        return result(Int);
      case IN: {
        if(!check_arity(f, 2) || !check_set_arg(f, 1, args[1])) { return {}; }
        if(args[1].sort.has_value()) {
          SortInfo elem(*args[1].sort->sub);
          if(!compatible(elem, args[0])) {
            error(f, "`in` expects an element of sort " + elem.to_string() + ", but the element has sort " + args[0].to_string() + ".");
            return {};
          }
        }
        return result(Bool);
      }
      case AND: case OR:
        if(!check_all_args(f, args, Bool)) { return {}; }
        return result(Bool);
      case EQUIV: case XOR:
        if(!check_min_arity(f, 2) || !check_all_args(f, args, Bool)) { return {}; }
        return result(Bool);
      case IMPLY:
        if(!check_arity(f, 2) || !check_all_args(f, args, Bool)) { return {}; }
        return result(Bool);
      case NOT:
        if(!check_arity(f, 1) || !check_all_args(f, args, Bool)) { return {}; }
        return result(Bool);
      case ITE:
        if(!check_arity(f, 3) || !check_arg(f, 0, args[0], Bool)) { return {}; }
        return check_same_sort(f, args, 1);
      case MAXIMIZE: case MINIMIZE:
        if(!check_arity(f, 1)) { return {}; }
        if(!args[0].is(Int) && !args[0].is(Real)) {
          error(f, "`" + name_of(f) + "` expects an argument of sort Int or Real, but the argument has sort " + args[0].to_string() + ".");
          return {};
        }
        return result(Bool);
      default:
        error(f, "Unknown symbol `" + name_of(f) + "`.");
        return {};
    }
  }
};

} // namespace impl

/** \return `true` if `f` is a well-formed formula with respect to the sorted signature, where the sorts of the variables are given by the existential quantifiers in `f` or by `env`.
 * A formula has the sort `Bool`; use `sort_of_term` for terms of other sorts.
 * If `f` is not well-formed and `diag` is not null, `diag` explains the first error found. */
template <class F, class Env = VarEnv<battery::standard_allocator>>
bool is_wellformed(const F& f, const Env& env, WFDiagnostic<typename F::allocator_type>* diag) {
  impl::WellformedChecker<F, Env> checker(env, diag);
  return checker.collect_existentials(f) && checker.check_formula(f);
}

template <class F, class Env = VarEnv<battery::standard_allocator>>
bool is_wellformed(const F& f, const Env& env, WFDiagnostic<typename F::allocator_type>& diag) {
  return is_wellformed(f, env, &diag);
}

template <class F, class Env = VarEnv<battery::standard_allocator>>
bool is_wellformed(const F& f, const Env& env = Env()) {
  return is_wellformed(f, env, static_cast<WFDiagnostic<typename F::allocator_type>*>(nullptr));
}

/** \return The sort of the term `t` if it is well-formed, where the sorts of the variables are given by `env` (and the existential quantifiers in `t`).
 * Contrarily to `TFormula::sort()`, the sort of logical variables is known, and existential quantifiers have the sort `Bool`.
 * The result is `{}` if `t` is not well-formed, or if its sort is a set of unknown elements (e.g., the empty set). */
template <class F, class Env = VarEnv<battery::standard_allocator>>
std::optional<Sort<typename F::allocator_type>> sort_of_term(const F& t, const Env& env = Env()) {
  impl::WellformedChecker<F, Env> checker(env, nullptr);
  if(!checker.collect_existentials(t)) { return {}; }
  auto s = checker.infer(t);
  if(s.has_value() && s->sort.has_value()) {
    return std::move(s->sort);
  }
  return {};
}

} // namespace lala

#endif // LALA_CORE_WELLFORMED_HPP
