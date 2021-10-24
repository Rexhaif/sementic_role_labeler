from yargy.morph import Case, Gender, Number, Grams, Form
from yargy.token import Token
from yargy.span import Span
import spacy_udpipe
import re
from yargy.tokenizer import Tokenizer, RULES
from pymorphy2 import MorphAnalyzer
from functools import lru_cache
class UdCase(Case):
    
    def __init__(self, grams):
        self.mask = [
            (_ in grams)
            for _ in ['Nom', 'Gen', 'Dat', 'Acc', 'Abl', 'Loc', 'Voc']
        ]
        self.fixed = False
        

class UdGender(Gender):
    def __init__(self, grams):
        self.male = 'Masc' in grams
        self.female = 'Fem' in grams
        self.neutral = 'Neut' in grams
        # https://github.com/OpenCorpora/opencorpora/issues/795
        self.bi = False
        self.general = 'Com' in grams
        
        
class UdNumber(Number):
    def __init__(self, grams):
        self.single = 'Sing' in grams
        self.plural = 'Plur' in grams
        self.only_single = 'Stan' in grams # not actually exists in universal dependencies
        self.only_plural = 'Ptan' in grams
        
        
class UdGrams(Grams):

    def __init__(self, values):
        self.values = values

    @property
    def gender(self):
        return UdGender(self)

    @property
    def number(self):
        return UdNumber(self)

    @property
    def case(self):
        return UdCase(self)

    def __contains__(self, value):
        return value in self.values

    def __repr__(self):
        values = sorted(self.values)
        return 'Grams({values})'.format(
            values=','.join(values)
        )

    def _repr_pretty_(self, printer, cycle):
        printer.text(repr(self))
        
        
class UdForm(Form):
    
    def __init__(self, normalized, grams):
        self.normalized = normalized
        self.grams = grams

    def __repr__(self):
        return 'Form({self.normalized!r}, {self.grams!r})'.format(self=self)

    def _repr_pretty_(self, printer, cycle):
        printer.text(repr(self))
        
        
class MorphImitator():
    
    def __init__(self):
        self.morph = MorphAnalyzer()
    
    def check_gram(self, gram):
        '''
        For some reason yargy uses implementation-specific morphology verification, so we have to imitate it's existence
        '''
        pass
    
    @lru_cache(maxsize=100000)
    def normalized(self, word):
        parse = self.morph.parse(word)
        return {_.normal_form for _ in parse}


class UdMorphTokenizer(Tokenizer):
    def __init__(self, model):
        super(UdMorphTokenizer, self).__init__(RULES)
        self.model = model
        self.morph = MorphImitator()
        
    def get_token_type(self, token):
        if token.is_ascii:
            return "LATIN"
        elif token.is_digit:
            return "INT"
        elif token.is_punct:
            return "PUNCT"
        elif re.match(r'[\n\r]+', token.text) is not None:
            return "EOL"
        elif re.match(r'[а-яёЁА-Я]+', token.text) is not None:
            return "RU"
        else:
            return "OTHER"
        
    def tokenize(self, doc):
        for token in doc:
            span = Span(token.idx, token.idx + len(token.text))
            token_type = self.get_token_type(token)
            yield token, Token(token.text, span, token_type)
            
    def get_morph(self, token):
        pos = token.pos_
        norm = token.lemma_
        features = set(token.morph.to_dict().values())
        features.add(pos)
        grams = UdGrams(features)
        return UdForm(norm, grams)
        

    def __call__(self, text):
        tokens = self.tokenize(self.model(text))
        for spacy_token, yargy_token in tokens:
            if yargy_token.type == "RU":
                forms = self.get_morph(spacy_token)
                yield yargy_token.morphed([forms])
            else:
                yield yargy_token