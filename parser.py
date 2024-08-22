import ply.yacc as yacc

from lexer import tokens


def p_statements(p):
    'statement : statement statement'
    p[0] = p[1] + p[2]


def p_statement_instruction(p):
    'statement : instruction'
    p[0] = [p[1]]


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


def p_factor_identifier(p):
    'factor : IDENTIFIER'
    p[0] = p[1]


def p_factor_expr(p):
    'factor : LPAREN expression RPAREN'
    p[0] = p[2]


def p_error(p):
    print("Syntax error in input!")


def p_empty(p):
    'empty :'
    pass


def p_assignment_bracket(p):
    'assignment : IDENTIFIER LBRACKET assignment RBRACKET'
    p[0] = [(p[1], p[3])]

##################


def p_list_bracket(p):
    'list : LBRACKET bare_list RBRACKET'
    p[0] = p[2]


def p_bare_list_expression(p):
    'bare_list : expression'
    p[0] = [p[1]]


def p_bare_list_expression_comma(p):
    'bare_list : bare_list COMMA expression'
    p[0] = p[1] + [p[3]]


def p_expression_list(p):
    'expression : list'
    p[0] = p[1]

##################


def p_assignment_comma(p):
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
    print('Parser: Slicing number with value: %s' % p[3])
    p[0] = [p[3]]


def p_instruction_slice_claim_bar(p):
    'instruction : slice_predicate CLAIM_BAR LPAREN assignment RPAREN'
    print('Parser: Claiming barrier with assignment: %s' % p[4])
    p[0] = ('claim_barrier', p[1] + p[4])


def p_instruction_slice_spm_alocate(p):
    'instruction : slice_predicate SPM_ALOCATE LPAREN assignment RPAREN'
    print('Parser: Alocating SPM with assignment: %s' % p[4])
    p[0] = ('spm_allocate', p[1] + p[4])


def p_tma_predicate(p):
    'tma_predicate : TMA DOT'


def p_slice_suffix_slice_number(p):
    'slice_suffix : DOT SLICE DOT NUMBER'
    print('Parser: Slicing number with value: %s' % p[4])
    p[0] = [p[4]]


def p_instruction_tma_load_slice(p):
    'instruction : tma_predicate LOAD slice_suffix LPAREN assignment RPAREN'
    print('Parser: Loading TMA with assignment: %s' % p[5])
    p[0] = ('tma_load_slice', p[3] + p[5])


def p_instruction_tma_load_multicast(p):
    'instruction : tma_predicate LOAD DOT MULTICAST LPAREN assignment RPAREN'
    print('Parser: Loading TMA multicast with assignment: %s' % p[6])
    p[0] = ('tma_load_multicast', p[6])


def p_instruction_tma_store_slice(p):
    'instruction : tma_predicate STORE slice_suffix LPAREN assignment RPAREN'
    print('Parser: Storing TMA with assignment: %s' % p[5])
    p[0] = ('tma_store_slice', p[3] + p[5])


def p_instruction_tma_store_multicast(p):
    'instruction : tma_predicate STORE DOT MULTICAST LPAREN assignment RPAREN'
    print('Parser: Storing TMA multicast with assignment: %s' % p[6])
    p[0] = ('tma_store_multicast', p[6])


parser = yacc.yacc()

if __name__ == '__main__':

    program = open('program.txt').read()
    res = parser.parse(program)
