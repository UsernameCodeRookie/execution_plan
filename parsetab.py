
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'ASSIGN CLAIM_BAR COMMA D DIVIDE DOT DOUBLE_COLON GEMM IDENTIFIER LAB LOAD LRB LSB MAKE_TENSOR MINUS MULTICAST NUMBER PLUS RAB RRB RSB SLICE SPM_ALOCATE STORE TIMES TMAstatement : statement statementstatement : instructionexpression : termexpression : expression PLUS termexpression : expression MINUS termterm : factorterm : term TIMES factorterm : term DIVIDE factorfactor : NUMBERfactor : IDENTIFIERfactor : LRB expression RRBempty :assignment : IDENTIFIER LSB assignment RSBlist : LSB bare_list RSBbare_list : expressionbare_list : bare_list COMMA expressionexpression : listassignment : assignment COMMA assignmentassignment : emptyassignment : NUMBERassignment : listassignment : IDENTIFIERassignment : IDENTIFIER ASSIGN expressionslice_predicate : SLICE DOT NUMBER DOTinstruction : slice_predicate CLAIM_BAR LRB assignment RRBinstruction : slice_predicate SPM_ALOCATE LRB assignment RRBtma_predicate : TMA DOTslice_suffix : DOT SLICE DOT NUMBERinstruction : tma_predicate LOAD slice_suffix LRB assignment RRBinstruction : tma_predicate LOAD DOT MULTICAST LRB assignment RRBinstruction : tma_predicate STORE slice_suffix LRB assignment RRBinstruction : tma_predicate STORE DOT MULTICAST LRB assignment RRBtemplate_suffix : DOUBLE_COLON LAB assignment RABtemplate_suffix : DOUBLE_COLON LAB NUMBER D RABinstruction : MAKE_TENSOR template_suffix LRB assignment RRBinstruction : slice_predicate GEMM template_suffix LRB assignment RRB'
    
_lr_action_items = {'MAKE_TENSOR':([0,1,2,8,45,57,64,77,78,81,91,92,],[5,5,-2,5,-25,-26,-35,-36,-29,-31,-30,-32,]),'SLICE':([0,1,2,8,22,24,45,57,64,77,78,81,91,92,],[6,6,-2,6,38,38,-25,-26,-35,-36,-29,-31,-30,-32,]),'TMA':([0,1,2,8,45,57,64,77,78,81,91,92,],[7,7,-2,7,-25,-26,-35,-36,-29,-31,-30,-32,]),'$end':([1,2,8,45,57,64,77,78,81,91,92,],[0,-2,-1,-25,-26,-35,-36,-29,-31,-30,-32,]),'CLAIM_BAR':([3,44,],[9,-24,]),'SPM_ALOCATE':([3,44,],[10,-24,]),'GEMM':([3,44,],[11,-24,]),'LOAD':([4,17,],[12,-27,]),'STORE':([4,17,],[13,-27,]),'DOUBLE_COLON':([5,11,],[15,15,]),'DOT':([6,7,12,13,27,38,],[16,17,22,24,44,61,]),'LRB':([9,10,14,20,21,23,30,37,40,48,56,65,71,72,73,74,75,80,83,],[18,19,25,35,36,39,56,60,63,56,56,-33,56,56,56,56,56,-28,-34,]),'LAB':([15,],[26,]),'NUMBER':([16,18,19,25,26,30,35,36,39,46,47,48,56,60,61,63,71,72,73,74,75,],[27,32,32,32,43,54,32,32,32,32,32,54,54,32,80,32,54,54,54,54,54,]),'IDENTIFIER':([18,19,25,26,30,35,36,39,46,47,48,56,60,63,71,72,73,74,75,],[29,29,29,29,55,29,29,29,29,29,55,55,29,29,55,55,55,55,55,]),'RRB':([18,19,25,28,29,31,32,33,34,35,36,39,41,46,51,52,53,54,55,58,59,60,62,63,67,69,70,76,79,82,84,86,87,88,89,90,],[-12,-12,-12,45,-22,-19,-20,-21,57,-12,-12,-12,64,-12,-3,-17,-6,-9,-10,77,78,-12,81,-12,-18,-23,-14,90,91,92,-13,-4,-5,-7,-8,-11,]),'COMMA':([18,19,25,26,28,29,31,32,33,34,35,36,39,41,42,43,46,47,49,50,51,52,53,54,55,58,59,60,62,63,67,68,69,70,79,82,84,85,86,87,88,89,90,],[-12,-12,-12,-12,46,-22,-19,-20,-21,46,-12,-12,-12,46,46,-20,-12,-12,71,-15,-3,-17,-6,-9,-10,46,46,-12,46,-12,46,46,-23,-14,46,46,-13,-16,-4,-5,-7,-8,-11,]),'LSB':([18,19,25,26,29,30,35,36,39,46,47,48,56,60,63,71,],[30,30,30,30,47,30,30,30,30,30,30,30,30,30,30,30,]),'MULTICAST':([22,24,],[37,40,]),'RAB':([26,29,31,32,33,42,43,46,51,52,53,54,55,66,67,69,70,84,86,87,88,89,90,],[-12,-22,-19,-20,-21,65,-20,-12,-3,-17,-6,-9,-10,83,-18,-23,-14,-13,-4,-5,-7,-8,-11,]),'RSB':([29,31,32,33,46,47,49,50,51,52,53,54,55,67,68,69,70,84,85,86,87,88,89,90,],[-22,-19,-20,-21,-12,-12,70,-15,-3,-17,-6,-9,-10,-18,84,-23,-14,-13,-16,-4,-5,-7,-8,-11,]),'ASSIGN':([29,],[48,]),'D':([43,],[66,]),'PLUS':([50,51,52,53,54,55,69,70,76,85,86,87,88,89,90,],[72,-3,-17,-6,-9,-10,72,-14,72,72,-4,-5,-7,-8,-11,]),'MINUS':([50,51,52,53,54,55,69,70,76,85,86,87,88,89,90,],[73,-3,-17,-6,-9,-10,73,-14,73,73,-4,-5,-7,-8,-11,]),'TIMES':([51,53,54,55,86,87,88,89,90,],[74,-6,-9,-10,74,74,-7,-8,-11,]),'DIVIDE':([51,53,54,55,86,87,88,89,90,],[75,-6,-9,-10,75,75,-7,-8,-11,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'statement':([0,1,8,],[1,8,8,]),'instruction':([0,1,8,],[2,2,2,]),'slice_predicate':([0,1,8,],[3,3,3,]),'tma_predicate':([0,1,8,],[4,4,4,]),'template_suffix':([5,11,],[14,20,]),'slice_suffix':([12,13,],[21,23,]),'assignment':([18,19,25,26,35,36,39,46,47,60,63,],[28,34,41,42,58,59,62,67,68,79,82,]),'empty':([18,19,25,26,35,36,39,46,47,60,63,],[31,31,31,31,31,31,31,31,31,31,31,]),'list':([18,19,25,26,30,35,36,39,46,47,48,56,60,63,71,],[33,33,33,33,52,33,33,33,33,33,52,52,33,33,52,]),'bare_list':([30,],[49,]),'expression':([30,48,56,71,],[50,69,76,85,]),'term':([30,48,56,71,72,73,],[51,51,51,51,86,87,]),'factor':([30,48,56,71,72,73,74,75,],[53,53,53,53,53,53,88,89,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> statement","S'",1,None,None,None),
  ('statement -> statement statement','statement',2,'p_statements','parser.py',8),
  ('statement -> instruction','statement',1,'p_statement_instruction','parser.py',13),
  ('expression -> term','expression',1,'p_expression_term','parser.py',18),
  ('expression -> expression PLUS term','expression',3,'p_expression_plus','parser.py',23),
  ('expression -> expression MINUS term','expression',3,'p_expression_minus','parser.py',28),
  ('term -> factor','term',1,'p_term_factor','parser.py',33),
  ('term -> term TIMES factor','term',3,'p_term_times','parser.py',38),
  ('term -> term DIVIDE factor','term',3,'p_term_div','parser.py',43),
  ('factor -> NUMBER','factor',1,'p_factor_num','parser.py',48),
  ('factor -> IDENTIFIER','factor',1,'p_factor_identifier','parser.py',53),
  ('factor -> LRB expression RRB','factor',3,'p_factor_expr','parser.py',58),
  ('empty -> <empty>','empty',0,'p_empty','parser.py',67),
  ('assignment -> IDENTIFIER LSB assignment RSB','assignment',4,'p_assignment_bracket','parser.py',72),
  ('list -> LSB bare_list RSB','list',3,'p_list_bracket','parser.py',79),
  ('bare_list -> expression','bare_list',1,'p_bare_list_expression','parser.py',84),
  ('bare_list -> bare_list COMMA expression','bare_list',3,'p_bare_list_expression_comma','parser.py',89),
  ('expression -> list','expression',1,'p_expression_list','parser.py',94),
  ('assignment -> assignment COMMA assignment','assignment',3,'p_assignment_comma','parser.py',101),
  ('assignment -> empty','assignment',1,'p_assignment_empty','parser.py',106),
  ('assignment -> NUMBER','assignment',1,'p_assignment_number','parser.py',111),
  ('assignment -> list','assignment',1,'p_assignment_list','parser.py',116),
  ('assignment -> IDENTIFIER','assignment',1,'p_assignment_identifier','parser.py',121),
  ('assignment -> IDENTIFIER ASSIGN expression','assignment',3,'p_assignment','parser.py',126),
  ('slice_predicate -> SLICE DOT NUMBER DOT','slice_predicate',4,'p_slice_predicate_slice_number','parser.py',131),
  ('instruction -> slice_predicate CLAIM_BAR LRB assignment RRB','instruction',5,'p_instruction_slice_claim_bar','parser.py',137),
  ('instruction -> slice_predicate SPM_ALOCATE LRB assignment RRB','instruction',5,'p_instruction_slice_spm_alocate','parser.py',143),
  ('tma_predicate -> TMA DOT','tma_predicate',2,'p_tma_predicate','parser.py',149),
  ('slice_suffix -> DOT SLICE DOT NUMBER','slice_suffix',4,'p_slice_suffix_slice_number','parser.py',153),
  ('instruction -> tma_predicate LOAD slice_suffix LRB assignment RRB','instruction',6,'p_instruction_tma_load_slice','parser.py',159),
  ('instruction -> tma_predicate LOAD DOT MULTICAST LRB assignment RRB','instruction',7,'p_instruction_tma_load_multicast','parser.py',165),
  ('instruction -> tma_predicate STORE slice_suffix LRB assignment RRB','instruction',6,'p_instruction_tma_store_slice','parser.py',171),
  ('instruction -> tma_predicate STORE DOT MULTICAST LRB assignment RRB','instruction',7,'p_instruction_tma_store_multicast','parser.py',177),
  ('template_suffix -> DOUBLE_COLON LAB assignment RAB','template_suffix',4,'p_template_suffix','parser.py',183),
  ('template_suffix -> DOUBLE_COLON LAB NUMBER D RAB','template_suffix',5,'p_make_tensor_template_suffix','parser.py',189),
  ('instruction -> MAKE_TENSOR template_suffix LRB assignment RRB','instruction',5,'p_instruction_make_tensor','parser.py',195),
  ('instruction -> slice_predicate GEMM template_suffix LRB assignment RRB','instruction',6,'p_instruction_slice_gemm','parser.py',201),
]
