import ply.lex as lex

tokens = (
    'NUMBER', 'CLAIM_BAR', 'COMMA', 'IDENTIFIER', 'LPAREN', 'RPAREN', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'DOT',
    'SLICE', 'SPM_ALOCATE', 'ASSIGN', 'TMA', 'MULTICAST', 'LOAD', 'STORE', 'LBRACKET', 'RBRACKET'
)

reserved = {
    'claim_bar': 'CLAIM_BAR',
    'slice': 'SLICE',
    'spm_allocate': 'SPM_ALOCATE',
    'tma': 'TMA',
    'multicast': 'MULTICAST',
    'load': 'LOAD',
    'store': 'STORE'
}

t_COMMA = r','
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'\/'
t_DOT = r'\.'
t_ASSIGN = r'='


def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t


def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count('\n')


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


t_ignore = ' \t'

lexer = lex.lex()

if __name__ == '__main__':

    program = open('program.txt').read()
    res = lexer.input(program)
    for token in lexer:
        print(token)
