# ---------- 1. স্টেটমেন্ট ডিকশনারি ----------
statements = {
    "s": {"subject": "students", "predicate": "brilliant", "singular": False},
    "a": {"subject": "Real CR", "predicate": "student", "singular": True},
    "b": {"subject": "Real CR", "predicate": "lazy", "singular": True},
}

# ---------- 2. হেল্পার ফাংশন ----------
def print_pos(k):
    st = statements[k]
    if st["singular"]:
        print(f"{st['subject']} is {st['predicate']}", end="")
    else:
        print(f"{st['subject']} are {st['predicate']}", end="")

def print_neg(k):
    st = statements[k]
    if st["singular"]:
        print(f"{st['subject']} is not {st['predicate']}", end="")
    else:
        print(f"{st['subject']} are not {st['predicate']}", end="")

# ---------- 3. লজিক → বাক্য ----------
def build_statement(text):
    text = text.replace(" ", "").replace(",", "")
    neg = False
    only_sub = False

    for ch in text:
        if ch == "&":
            print(" and ", end="")
        elif ch == "|":
            print(" or ", end="")
        elif ch == "~":
            print(" implies ", end="")
        elif ch == "*":
            print(" if and only if ", end="")
        elif ch == "!":
            neg = True
        elif ch == "A":
            print("For all ", end="")
            only_sub = True
        elif ch == "E":
            print("For some ", end="")
            only_sub = True
        elif ch in ["(", ")"]:
            continue
        else:           # স্টেটমেন্টের কী (s, a, b)
            if only_sub:
                print(statements[ch]["subject"], end=", ")
                only_sub = False
            elif neg:
                print_neg(ch)
                neg = False
            else:
                print_pos(ch)
    print(".")  # বাক্য শেষ

from itertools import product

build_statement("A(s) s")
# build_statement("a & !b ~ a")


def evaluate(a, b, op):
    if op == "!": return 0 if a == 1 else 1
    if op == "&": return 1 if a == 1 and b == 1 else 0
    if op == "|": return 1 if a == 1 or b == 1 else 0
    if op == "~": return 0 if a == 1 and b == 0 else 1  # implies
    if op == "*": return 1 if a == b else 0             # iff

def truth_table(op, p_name="P", q_name="Q"):
    print(f"\nTruth table for {p_name} {op} {q_name}")
    print(f"{p_name}\t{q_name}\tResult")
    for p, q in product([0, 1], repeat=2):
        print(f"{p}\t{q}\t{evaluate(p, q, op)}")

# ---------- 5️⃣ main ----------
print("► Sentence building:")
build_statement("A(s) s")
build_statement("a & !b ~ a")

print("\n► Truth tables:")
truth_table("&", "a", "b")   # AND
# truth_table("|", "a", "b")   # OR
# truth_table("~", "a", "b")   # IMPLIES
# truth_table("*", "a", "b")   # IFF