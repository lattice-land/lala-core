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
    suberrors.push_back(std::move(suberror));
    return *this;
  }

  CUDA size_t num_suberrors() const {
    return suberrors.size();
  }

  /** This operator moves all `suberrors[i..(n-1)]` as a suberror of `suberrors[i-1]`.
   * If only warnings are present, the `suberrors[i-1]` is converted into a warning.
   */
  CUDA void merge(size_t i) {
    assert(i > 0);
    assert(i <= suberrors.size());
    if(i > suberrors.size()) {
      suberrors[i-1].fatal = false;
    }
    for(int j = i; j < suberrors.size(); ++j) {
      suberrors[i-1].fatal |= suberrors[j].is_fatal();
      suberrors[i-1].add_suberror(std::move(suberrors[j]));
    }
    suberrors.resize(i);
  }

  CUDA NI void print(int indent = 0) const {
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
    for(int i = 0; i < suberrors.size(); ++i) {
      suberrors[i].print(indent + 2);
      printf("\n");
    }
  }

  CUDA bool is_fatal() const { return fatal; }
};

#define RETURN_INTERPRETATION_ERROR(msg) \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics<F>(true, name, (msg), f)); \
  } \
  return false;

#define RETURN_INTERPRETATION_WARNING(msg) \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics<F>(false, name, (msg), f)); \
  } \
  return true;


/** This macro creates a high-level error message that is possibly erased if `call` does not lead to any error.
 * If `call` leads to errors, these errors are moved as suberrors of the high-level error message.
 * Additionally, `merge` is executed if `call` does not lead to any error.
 */
#define CALL_WITH_ERROR_CONTEXT_WITH_MERGE(msg, call, merge) \
  size_t error_context = diagnostics.num_suberrors(); \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostic<F>(true, name, (msg), f)); \
  } \
  bool res = call; \
  if constexpr(diagnose) { \
    diagnostics.merge(error_context); \
  } \
  if(res) { merge; } \
  return res;

#define CALL_WITH_ERROR_CONTEXT(msg, call) \
  CALL_WITH_ERROR_CONTEXT_WITH_MERGE(msg, call, {})

}

#endif
