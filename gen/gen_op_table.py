import csv

name2idx = {}
idx2name = {}
m = []

def add_monotone_entry(output, key, g):
  if g is not None and g != "-" and key[0] != "-" and key[1] != "-":
    # print(key[0] + " |_| " + key[1] + " = " + g)
    output[key] = g

def create_basic(output):
  for x in name2idx:
    for y in name2idx:
      add_monotone_entry(output, (x,y), m[name2idx[x]][name2idx[y]])

def generalize(x, y):
  if x == None or y == None:
    return None
  if x == "-" or y == "-":
    return None
  if x == y:
    return x
  if x[-1] == "-" or x[-1] == "+":
    x2 = x[:-1]
  else:
    x2 = x
  if y[-1] == "-" or y[-1] == "+":
    y2 = y[:-1]
  else:
    y2 = y
  if x2 == y2:
    return x2
  else:
    return None

def add_signed_version(output):
  for i in range(0, len(m), 2):
    for j in range(0, len(m[0]), 2):
      gcol = generalize(idx2name[i], idx2name[i+1])
      gline = generalize(idx2name[j], idx2name[j+1])
      add_monotone_entry(output, (idx2name[i], gline), generalize(m[i][j], m[i][j+1]))
      add_monotone_entry(output, (idx2name[i+1], gline), generalize(m[i+1][j], m[i+1][j+1]))
      add_monotone_entry(output, (gcol, idx2name[j]), generalize(m[i][j], m[i+1][j]))
      add_monotone_entry(output, (gcol, idx2name[j+1]), generalize(m[i][j+1], m[i+1][j+1]))
      add_monotone_entry(output, (gcol, gline), generalize(generalize(m[i][j], m[i][j+1]), generalize(m[i+1][j], m[i+1][j+1])))

name2cpp = {
  "z": (False, "V"),
  "z+": (False, "spos<V>"),
  "z-": (False, "sneg<V>"),
  "zi": (True, "ZInc<V>"),
  "zi+": (True, "ZPInc<V>"),
  "zi-": (True, "ZNInc<V>"),
  "zd": (True, "ZDec<V>"),
  "zd+": (True, "ZPDec<V>"),
  "zd-": (True, "ZNDec<V>"),
  "U": (False, "typename L::ValueType"),
  "L": (False, "L"),
  "LD": (False, "typename L::dual_type"),
  "B": (False, "bool"),
  "BI": (True, "BInc"),
  "BD": (True, "BDec")
}

def generate_arith_binop_cpp(o, binop):
  print("\ntemplate<typename L, typename K> struct " + binop + " {};")
  for (a,b) in o:
    print("template<class V> struct " + binop + "<" + name2cpp[a][1] + ", " + name2cpp[b][1] + "> { using type = " + name2cpp[o[(a,b)]][1] + "; };")

def read_file(filename):
  reader = csv.reader(open(filename), delimiter=" ", skipinitialspace=True)
  data = []
  for row in reader:
    data.append(row)
  return data

def populate_name_idx_map(data):
  name2idx.clear()
  idx2name.clear()
  i = 0
  for c in data[0][1:]:
    name2idx[c] = i
    idx2name[i] = c
    i = i + 1

def create_binary_fun_matrix(filename):
  data = read_file(filename)
  populate_name_idx_map(data)
  # strip row and column labels
  m.clear()
  for row in data[1:]:
    m.append(row[1:])

def monotone_binary_function(filename, binop):
  create_binary_fun_matrix(filename)
  o = {}
  create_basic(o)
  add_signed_version(o)
  generate_arith_binop_cpp(o, binop)

def generate_unop_cpp(o, unop):
  print("\ntemplate<typename L, typename = void> struct " + unop + " {};")
  for a in o:
    if name2cpp[a][1] == "V": # Becasue a template specialization must be constrained.
      print("template<class V> struct " + unop + "<V, std::enable_if_t<std::is_arithmetic_v<V>>> { using type = V; };")
    else:
      print("template<class V> struct " + unop + "<" + name2cpp[a][1] + "> { using type = " + name2cpp[o[a]][1] + "; };")

def monotone_unary_function(filename, unop):
  data = read_file(filename)
  populate_name_idx_map(data)
  m = data[1][1:]
  o = {}
  # create_basic
  for x in name2idx:
    o[x] = m[name2idx[x]]
  # add_signed_version
  for i in range(0, len(m), 2):
    gcol = generalize(idx2name[i], idx2name[i+1])
    r = generalize(m[i], m[i+1])
    if gcol is not None and r is not None:
      o[gcol] = r
  generate_unop_cpp(o, unop)

def combine_unary_binary(unop_filename, binop_filename, binop_res):
  data = read_file(unop_filename)
  unop = data[1][1:]
  data = read_file(binop_filename)
  populate_name_idx_map(data)
  binop = []
  for row in data[1:]:
    binop.append(row[1:])
  for x in range(0, len(binop)):
    for y in range(0, len(binop[0])):
      m[x][y] = binop[x][name2idx[unop[y]]]
  o = {}
  create_basic(o)
  add_signed_version(o)
  generate_arith_binop_cpp(o, binop_res)

def generate_lattice_binop_cpp(o, binop, with_lattice_referential):
  if with_lattice_referential:
    print("\ntemplate<class O, class L, class K> struct " + binop + " {};")
  else:
    print("\ntemplate<class L, class K> struct " + binop + " {};")
  for (a,b) in o:
    if with_lattice_referential:
      print("template<class L> struct " + binop + "<L, " + name2cpp[a][1] + ", " + name2cpp[b][1] + "> { using type = " + name2cpp[o[(a,b)]][1] + "; };")
    else:
      print("template<class L> struct " + binop + "<" + name2cpp[a][1] + ", " + name2cpp[b][1] + "> { using type = " + name2cpp[o[(a,b)]][1] + "; };")

def generate_logic_binop_cpp(o, binop):
  print("\ntemplate<class L, class K> struct " + binop + " {};")
  for (a,b) in o:
    print("template<> struct " + binop + "<" + name2cpp[a][1] + ", " + name2cpp[b][1] + "> { using type = " + name2cpp[o[(a,b)]][1] + "; };")

def monotone_binary_lattice_op(filename, binop, with_lattice_referential):
  create_binary_fun_matrix(filename)
  o = {}
  create_basic(o)
  generate_lattice_binop_cpp(o, binop, with_lattice_referential)

# Subtraction `a - b` defined as `a + neg(b)`.
def subtraction_function(binop):
  combine_unary_binary("neg.txt", "add.txt", binop)

# Division `a / b` defined as `a * 1/b`
def division_function(binop):
  combine_unary_binary("inv.txt", "mul.txt", binop)

dual_of = {
  "-": "-",
  "z": "z",
  "z+": "z+",
  "z-": "z-",
  "zi": "zd",
  "zi+": "zd+",
  "zi-": "zd-",
  "zd": "zi",
  "zd+": "zi+",
  "zd-": "zi-",
  "U": "U",
  "L": "typename L::dual_type",
  "LD": "typename LD::dual_type",
  "B": "B",
  "BI": "BD",
  "BD": "BI"
}

def dualize(o):
  for k in o:
    o[k] = dual_of[o[k]]

def monotone_binary_lattice_op_dual(filename, binop, with_lattice_referential):
  create_binary_fun_matrix(filename)
  o = {}
  create_basic(o)
  dualize(o)
  generate_lattice_binop_cpp(o, binop, with_lattice_referential)

def monotone_unary_logic_function(filename, unop):
  data = read_file(filename)
  populate_name_idx_map(data)
  m = data[1][1:]
  o = {}
  # create_basic
  for x in name2idx:
    o[x] = m[name2idx[x]]
  # generate c++
  print("\ntemplate<class L> struct " + unop + " {};")
  for a in o:
    print("template<> struct " + unop + "<" + name2cpp[a][1] + "> { using type = " + name2cpp[o[a]][1] + "; };")

def monotone_binary_logic_function(filename, binop):
  create_binary_fun_matrix(filename)
  o = {}
  create_basic(o)
  generate_logic_binop_cpp(o, binop)


monotone_unary_function("neg.txt", "neg_z")
monotone_unary_function("abs.txt", "abs_z")
monotone_binary_function("add.txt", "add_z")
subtraction_function("sub_z")
monotone_binary_function("mul.txt", "mul_z")
division_function("div_z")
monotone_unary_function("sqr.txt", "sqr_z")
monotone_binary_function("pow.txt", "pow_z")

monotone_binary_lattice_op("join.txt", "join_t", False)
monotone_binary_lattice_op("meet.txt", "meet_t", False)
monotone_binary_lattice_op("leq.txt", "leq_t", True)
monotone_binary_lattice_op("leq.txt", "lt_t", True)
monotone_binary_lattice_op_dual("leq.txt", "geq_t", True)
monotone_binary_lattice_op_dual("leq.txt", "gt_t", True)

monotone_unary_logic_function("not.txt", "not_t")
monotone_binary_logic_function("and.txt", "and_t")
monotone_binary_logic_function("or.txt", "or_t")
monotone_binary_logic_function("equiv.txt", "equiv_t")
monotone_binary_logic_function("imply.txt", "imply_t")
monotone_binary_logic_function("xor.txt", "xor_t")
