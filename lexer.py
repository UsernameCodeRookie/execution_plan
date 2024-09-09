import ply.lex as lex

tokens = (
    'NUMBER', 'CLAIM_BAR', 'COMMA', 'IDENTIFIER', 'LRB', 'RRB', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'DOT',
    'SLICE', 'SPM_ALOCATE', 'ASSIGN', 'TMA', 'MULTICAST', 'LOAD', 'STORE', 'LSB', 'RSB', 'LAB', 'RAB',
    'DOUBLE_COLON', 'MAKE_TENSOR', 'GEMM', 'D', 'SPM_FREE'
)

reserved = {
    'claim_bar': 'CLAIM_BAR',
    'slice': 'SLICE',
    'spm_allocate': 'SPM_ALOCATE',
    'spm_free': 'SPM_FREE',
    'tma': 'TMA',
    'multicast': 'MULTICAST',
    'load': 'LOAD',
    'store': 'STORE',
    'make_tensor': 'MAKE_TENSOR',
    'gemm': 'GEMM',
    'D': 'D'
}

t_COMMA = r'\,'
t_LRB = r'\('
t_RRB = r'\)'
t_LSB = r'\['
t_RSB = r'\]'
t_LAB = r'\<'
t_RAB = r'\>'
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'\/'
t_DOT = r'\.'
t_ASSIGN = r'\='
t_DOUBLE_COLON = r'\:\:'


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

    program = open('resource/program.txt').read()
    res = lexer.input(program)
    for token in lexer:
        print(token)
