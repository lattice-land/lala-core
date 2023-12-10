// Copyright 2021 Pierre Talbot

#ifndef LALA_CORE_IDIAGNOSTICS_HPP
#define LALA_CORE_IDIAGNOSTICS_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "ast.hpp"

namespace lala {

/** `IDiagnostics` is used in abstract domains to diagnose why a formula cannot be interpreted (error) or if it was interpreted by under- or over-approximation (warnings).
    If the abstract domain cannot interpret the formula, it must explain why.
    This is similar to compilation errors in compiler. */
template<class F, class Allocator = typename F::allocator_type>
class IDiagnostics {
public:
  using allocator_type = typename F::allocator_type;
  using this_type = IDiagnostics<F>;

private:
  battery::string<allocator_type> ad_name;
  battery::string<allocator_type> description;
  F uninterpretable_formula;
  AType aty;
  battery::vector<IDiagnostics<F>, allocator_type> suberrors;
  bool fatal;

  CUDA void print_indent(int indent) const {
    for(int i = 0; i < indent; ++i) {
      printf(" ");
    }
  }

  CUDA void print_line(const char* line, int indent) const {
    print_indent(indent);
    printf("%s", line);
  }

public:
  CUDA NI IDiagnostics(): fatal(false), aty(-2) {} // special value indicating it is a top-level diagnostics.

  // If fatal is false, it is considered as a warning.
  CUDA NI IDiagnostics(bool fatal,
    battery::string<allocator_type> ad_name,
    battery::string<allocator_type> description,
    F uninterpretable_formula,
    AType aty = UNTYPED)
   : fatal(fatal),
     ad_name(std::move(ad_name)),
     description(std::move(description)),
     uninterpretable_formula(std::move(uninterpretable_formula)),
     aty(aty)
  {}

  CUDA NI this_type& add_suberror(IDiagnostics<F>&& suberror) {
    fatal |= suberror.is_fatal();
    suberrors.push_back(std::move(suberror));
    return *this;
  }

  CUDA size_t num_suberrors() const {
    return suberrors.size();
  }

  /** This operator moves all `suberrors[i..(n-1)]` as a suberror of `suberrors[i-1]`.
   * If only warnings are present, `suberrors[i-1]` is converted into a warning.
   * If `succeeded` is true, then all suberrors are erased.
   */
  CUDA void merge(bool succeeded, size_t i) {
    assert(i > 0);
    assert(i <= suberrors.size());
    suberrors[i-1].fatal = !succeeded;
    for(int j = i; j < suberrors.size(); ++j) {
      // In case of success, we erase the fatal suberrors.
      if(!succeeded || !suberrors[j].is_fatal()) {
        suberrors[i-1].add_suberror(std::move(suberrors[j]));
      }
    }
    suberrors.resize((suberrors[i-1].num_suberrors() == 0 && succeeded) ? i-1 : i);
    fatal = false;
    for(int i = 0; i < suberrors.size(); ++i) {
      if(suberrors[i].is_fatal()) {
        fatal = true;
        return;
      }
    }
  }

  CUDA NI void print(int indent = 0) const {
    // If it is not a top-level error, we print it, otherwise all errors are listed as `suberrors`.
    if(aty != -2) {
      if(fatal) {
        print_line("[error] ", indent);
      }
      else {
        print_line("[warning] ", indent);
      }
      printf("Uninterpretable formula.\n");
      print_indent(indent);
      printf("  Abstract domain: %s\n", ad_name.data());
      print_line("  Abstract type: ", indent);
      if(aty == UNTYPED) {
        printf("untyped\n");
      }
      else {
        printf("%d\n", aty);
      }
      print_line("  Formula: ", indent);
      uninterpretable_formula.print(true);
      printf("\n");
      print_indent(indent);
      printf("  Description: %s\n", description.data());
    }
    else {
      indent -= 2;
    }
    for(int i = 0; i < suberrors.size(); ++i) {
      suberrors[i].print(indent + 2);
      printf("\n");
    }
  }

  CUDA bool is_fatal() const { return fatal; }
  CUDA bool has_warning() const {
    for(int i = 0; i < suberrors.size(); ++i) {
      if(!suberrors[i].is_fatal()) {
        return true;
      }
    }
    return false;
  }
};

#define RETURN_INTERPRETATION_ERROR(MSG) \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics<F>(true, name, (MSG), f)); \
  } \
  return false;

#define RETURN_INTERPRETATION_WARNING(MSG) \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics<F>(false, name, (MSG), f)); \
  } \
  return true;


/** This macro creates a high-level error message that is possibly erased if `call` does not lead to any error.
 * If `call` leads to errors, these errors are moved as suberrors of the high-level error message.
 * Additionally, `merge` is executed if `call` does not lead to any error.
 */
#define CALL_WITH_ERROR_CONTEXT_WITH_MERGE(MSG, CALL, MERGE) \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics<F>(false, name, (MSG), f)); \
  } \
  size_t error_context = diagnostics.num_suberrors(); \
  bool res = CALL; \
  if constexpr(diagnose) { \
    diagnostics.merge(res, error_context); \
  } \
  if(res) { MERGE; } \
  return res;

#define CALL_WITH_ERROR_CONTEXT(MSG, CALL) \
  CALL_WITH_ERROR_CONTEXT_WITH_MERGE(MSG, CALL, {})

}

#endif
