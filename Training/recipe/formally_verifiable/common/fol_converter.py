import re
from typing import List, Set, Tuple, Union
from z3 import (
    And, Or, Not, Implies, ForAll, Exists, Xor,
    BoolSort, DeclareSort, Const, Function,
    Solver, unsat, is_expr,
)


class FOLToken:
    IDENT = "IDENT"
    FORALL = "FORALL"
    EXISTS = "EXISTS"
    ARROW = "ARROW"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"
    COMMA = "COMMA"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"


def tokenize_fol(text: str) -> List[Tuple[str, str]]:

    tokens = []
    i = 0
    text = text.strip()
    while i < len(text):
        c = text[i]

        if c.isspace():
            i += 1
            continue

        if c == '∀':
            tokens.append((FOLToken.FORALL, c)); i += 1; continue
        if c == '∃':
            tokens.append((FOLToken.EXISTS, c)); i += 1; continue
        if c == '→':
            tokens.append((FOLToken.ARROW, c)); i += 1; continue
        if c == '∧':
            tokens.append((FOLToken.AND, c)); i += 1; continue
        if c == '∨':
            tokens.append((FOLToken.OR, c)); i += 1; continue
        if c == '¬':
            tokens.append((FOLToken.NOT, c)); i += 1; continue
        if c == '⊕':
            tokens.append((FOLToken.XOR, c)); i += 1; continue
        if c == ',':
            tokens.append((FOLToken.COMMA, c)); i += 1; continue
        if c == '(':
            tokens.append((FOLToken.LPAREN, c)); i += 1; continue
        if c == ')':
            tokens.append((FOLToken.RPAREN, c)); i += 1; continue

        if c.isalpha() or c == '_':
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            tokens.append((FOLToken.IDENT, text[i:j]))
            i = j
            continue

        raise ValueError(f"Unexpected character '{c}' at position {i} in: {text}")
    return tokens


class ASTNode:
    pass

class VarNode(ASTNode):

    def __init__(self, name: str):
        self.name = name

class ConstNode(ASTNode):

    def __init__(self, name: str):
        self.name = name

class AppNode(ASTNode):

    def __init__(self, func: str, args: List[ASTNode]):
        self.func = func
        self.args = args

class NotNode(ASTNode):

    def __init__(self, body: ASTNode):
        self.body = body

class BinOpNode(ASTNode):

    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

class QuantNode(ASTNode):

    def __init__(self, quant: str, var: str, body: ASTNode):
        self.quant = quant
        self.var = var
        self.body = body


class FOLParser:


    def __init__(self, tokens: List[Tuple[str, str]], bound_vars: Set[str]):
        self.tokens = tokens
        self.pos = 0
        self.bound_vars = bound_vars


    def peek(self) -> Tuple[str, str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ("EOF", "")


    def consume(self, expected_type: str = None) -> Tuple[str, str]:
        tok = self.peek()
        if expected_type and tok[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok


    def parse_expr(self) -> ASTNode:

        tok = self.peek()
        if tok[0] == FOLToken.FORALL:
            return self.parse_quant()
        if tok[0] == FOLToken.EXISTS:
            return self.parse_quant()
        return self.parse_implies()


    def parse_quant(self) -> ASTNode:
        tok = self.consume()
        quant = "forall" if tok[0] == FOLToken.FORALL else "exists"
        var_tok = self.consume(FOLToken.IDENT)
        var_name = var_tok[1]

        if self.peek()[0] == FOLToken.COMMA:
            self.consume()
        self.bound_vars.add(var_name)
        body = self.parse_expr()
        self.bound_vars.discard(var_name)
        return QuantNode(quant, var_name, body)


    def parse_implies(self) -> ASTNode:
        left = self.parse_or()
        while self.peek()[0] == FOLToken.ARROW:
            self.consume()
            right = self.parse_or()
            left = BinOpNode("implies", left, right)
        return left


    def parse_or(self) -> ASTNode:
        left = self.parse_xor()
        while self.peek()[0] == FOLToken.OR:
            self.consume()
            right = self.parse_xor()
            left = BinOpNode("or", left, right)
        return left


    def parse_xor(self) -> ASTNode:
        left = self.parse_and()
        while self.peek()[0] == FOLToken.XOR:
            self.consume()
            right = self.parse_and()
            left = BinOpNode("xor", left, right)
        return left


    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        while self.peek()[0] == FOLToken.AND:
            self.consume()
            right = self.parse_not()
            left = BinOpNode("and", left, right)
        return left


    def parse_not(self) -> ASTNode:
        if self.peek()[0] == FOLToken.NOT:
            self.consume()
            body = self.parse_not()
            return NotNode(body)
        return self.parse_app()


    def parse_app(self) -> ASTNode:
        primary = self.parse_primary()
        args = []

        while True:
            tok = self.peek()
            if tok[0] in (FOLToken.IDENT, FOLToken.LPAREN, FOLToken.NOT,
                          FOLToken.FORALL, FOLToken.EXISTS):


                if tok[0] in (FOLToken.FORALL, FOLToken.EXISTS):
                    break
                arg = self.parse_primary()
                args.append(arg)
            else:
                break

        if args:

            if isinstance(primary, VarNode):
                return AppNode(primary.name, args)
            elif isinstance(primary, ConstNode):
                return AppNode(primary.name, args)
            else:


                raise ValueError(f"Cannot apply arguments to non-identifier: {primary}")
        return primary


    def parse_primary(self) -> ASTNode:
        tok = self.peek()
        if tok[0] == FOLToken.LPAREN:
            self.consume()
            expr = self.parse_expr()
            self.consume(FOLToken.RPAREN)
            return expr
        if tok[0] == FOLToken.IDENT:
            self.consume()
            name = tok[1]
            if name in self.bound_vars:
                return VarNode(name)


            if len(name) == 1 and name.islower():
                return VarNode(name)
            return ConstNode(name)
        raise ValueError(f"Unexpected token in primary: {tok}")


def collect_bound_vars(tokens: List[Tuple[str, str]]) -> Set[str]:

    bound = set()
    i = 0
    while i < len(tokens):
        if tokens[i][0] in (FOLToken.FORALL, FOLToken.EXISTS):
            i += 1
            if i < len(tokens) and tokens[i][0] == FOLToken.IDENT:
                bound.add(tokens[i][1])
                i += 1
            continue
        i += 1
    return bound


class FOLToZ3Converter:


    def __init__(self):
        self.Entity = DeclareSort('Entity')
        self.bound_vars: dict = {}
        self.consts: dict = {}
        self.preds: dict = {}
        self.funcs: dict = {}


    def reset(self):
        self.bound_vars = {}
        self.consts = {}
        self.preds = {}
        self.funcs = {}


    def get_or_create_const(self, name: str):
        if name not in self.consts:
            self.consts[name] = Const(name, self.Entity)
        return self.consts[name]


    def get_or_create_var(self, name: str):
        if name not in self.bound_vars:
            self.bound_vars[name] = Const(name, self.Entity)
        return self.bound_vars[name]


    def get_or_create_pred(self, name: str, arity: int):
        key = (name, arity)
        if key not in self.preds:
            sorts = [self.Entity] * arity + [BoolSort()]
            self.preds[key] = Function(name, *sorts)
        return self.preds[key]


    def to_z3(self, node: ASTNode) -> is_expr:
        if isinstance(node, VarNode):
            return self.get_or_create_var(node.name)
        if isinstance(node, ConstNode):
            return self.get_or_create_const(node.name)
        if isinstance(node, AppNode):
            pred = self.get_or_create_pred(node.func, len(node.args))
            z3_args = [self.to_z3(a) for a in node.args]
            return pred(*z3_args)
        if isinstance(node, NotNode):
            return Not(self.to_z3(node.body))
        if isinstance(node, BinOpNode):
            left = self.to_z3(node.left)
            right = self.to_z3(node.right)
            if node.op == "and":
                return And(left, right)
            if node.op == "or":
                return Or(left, right)
            if node.op == "implies":
                return Implies(left, right)
            if node.op == "xor":
                return Xor(left, right)
            raise ValueError(f"Unknown binop: {node.op}")
        if isinstance(node, QuantNode):
            var = self.get_or_create_var(node.var)
            body = self.to_z3(node.body)
            if node.quant == "forall":
                return ForAll([var], body)
            if node.quant == "exists":
                return Exists([var], body)
            raise ValueError(f"Unknown quant: {node.quant}")
        raise ValueError(f"Unknown AST node: {node}")


    def convert(self, text: str) -> is_expr:

        self.reset()
        tokens = tokenize_fol(text)
        bound_vars = collect_bound_vars(tokens)
        parser = FOLParser(tokens, bound_vars)
        ast = parser.parse_expr()
        if parser.peek()[0] != "EOF":
            remaining = " ".join([t[1] for t in parser.tokens[parser.pos:]])
            raise ValueError(f"Unexpected trailing tokens: {remaining}")
        return self.to_z3(ast)


def verify_implication(premises_z3: List, conclusion_z3, timeout_ms: int = 5000) -> Tuple[bool, str]:


    s = Solver()
    s.set("timeout", timeout_ms)
    for p in premises_z3:
        s.add(p)
    s.add(Not(conclusion_z3))
    result = s.check()
    if result == unsat:
        return True, "UNSAT: conclusion follows from premises"
    if result == unsat:
        return True, "UNSAT"

    if str(result) == "unknown":
        return False, "UNKNOWN (possibly timeout)"
    return False, f"SAT: counterexample exists ({result})"
