import ply.yacc as yacc

from lexer import tokens


def p_statements(p):
    'statement : statement statement'
    p[0] = p[2]


def p_statement_instruction(p):
    'statement : instruction'
    p[0] = p[1]


def p_expression_term(p):
    'expression : term'
    p[0] = p[1]


def p_expression_plus(p):
    'expression : expression PLUS term'
    p[0] = p[1] + p[3]


def p_expression_minus(p):
    'expression : expression MINUS term'
    p[0] = p[1] - p[3]


def p_term_factor(p):
    'term : factor'
    p[0] = p[1]


def p_term_times(p):
    'term : term TIMES factor'
    p[0] = p[1] * p[3]


def p_term_div(p):
    'term : term DIVIDE factor'
    p[0] = p[1] / p[3]


def p_factor_num(p):
    'factor : NUMBER'
    p[0] = p[1]


def p_factor_expr(p):
    'factor : LPAREN expression RPAREN'
    p[0] = p[2]


def p_error(p):
    print("Syntax error in input!")


def p_empty(p):
    'empty :'
    pass


def p_assignments(p):
    'assignment : assignment COMMA assignment'
    p[0] = p[1] + p[3]


def p_assignment_empty(p):
    'assignment : empty'
    p[0] = []


def p_assignment_number(p):
    'assignment : NUMBER'
    p[0] = [p[1]]


def p_assignment_identifier(p):
    'assignment : IDENTIFIER'
    p[0] = [p[1]]


def p_assignment(p):
    'assignment : IDENTIFIER ASSIGN expression'
    p[0] = [(p[1], p[3])]


def p_slice_predicate_slice_number(p):
    'slice_predicate : SLICE DOT NUMBER DOT'
    print('Slicing number with value: %s' % p[3])
    p[0] = p[3]


def p_instruction_slice_claim_bar(p):
    'instruction : slice_predicate CLAIM_BAR LPAREN assignment RPAREN'
    print('Claiming barrier with assignment: %s' % p[4])


def p_instruction_slice_spm_alocate(p):
    'instruction : slice_predicate SPM_ALOCATE LPAREN assignment RPAREN'
    print('Alocating SPM with assignment: %s' % p[4])


def p_tma_predicate(p):
    'tma_predicate : TMA DOT'


def p_instruction_tma_load_slice(p):
    'instruction : tma_predicate LOAD DOT SLICE DOT NUMBER LPAREN assignment RPAREN'
    print('Loading TMA with assignment: %s' % p[8])


def p_instruction_tma_load_multicast(p):
    'instruction : tma_predicate LOAD DOT MULTICAST LPAREN assignment RPAREN'
    print('Loading TMA multicast with assignment: %s' % p[6])


def p_instruction_tma_store_slice(p):
    'instruction : tma_predicate STORE DOT SLICE DOT NUMBER LPAREN assignment RPAREN'
    print('Storing TMA with assignment: %s' % p[8])


def p_instruction_tma_store_multicast(p):
    'instruction : tma_predicate STORE DOT MULTICAST LPAREN assignment RPAREN'
    print('Storing TMA multicast with assignment: %s' % p[6])


parser = yacc.yacc()

if __name__ == '__main__':

    program = open('program.txt').read()
    res = parser.parse(program)
    print(res)
