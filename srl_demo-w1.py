#!/usr/bin/env python
# coding: utf-8
import json
import os
from collections import defaultdict
from functools import lru_cache
from typing import *

import joblib as jb
import pandas as pd
import razdel
import yargy as y
import yargy.morph as ytm
import yargy.pipelines as pipelines
import yargy.predicates as yp
import yargy.tokenizer as yt
from isanlp import PipelineCommon
from isanlp.processor_remote import ProcessorRemote
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd
from isanlp.ru.processor_mystem import ProcessorMystem
from navec import Navec
from predpatt import PredPatt, PredPattOpts, load_conllu
from predpatt.patt import Token
from predpatt.util.ud import dep_v1, dep_v2
from pyhash import city_32
from pymorphy2 import MorphAnalyzer
from rich import inspect, print
from slovnet import Syntax
from tqdm.auto import tqdm
from ufal.udpipe import Model, Pipeline, ProcessingError

CACHE_SIZE = 10000


class MostProbMorphAnalyzer(ytm.MorphAnalyzer):
    def __call__(self, word):
        records = self.raw.parse(word)
        max_score = max(x.score for x in records)
        records = list(filter(lambda x: x.score == max_score, records))
        return [ytm.prepare_form(record) for record in records]


class CachedMostProbMorphAnalyzer(MostProbMorphAnalyzer):
    def __init__(self):
        super(CachedMostProbMorphAnalyzer, self).__init__()

    __call__ = lru_cache(CACHE_SIZE)(MostProbMorphAnalyzer.__call__)


def create_predicate_rule(
    require_deverbal_noun: str,
    require_reflexive: str,
    require_status_category: str,
    predicate: str,
    predicate_type: str,
    **kwargs,
):
    rule_id = f"predicate={predicate},deverbal={require_deverbal_noun},reflexive={require_reflexive},status_category={require_status_category},predicate_type={predicate_type}"
    return rule_id, y.rule(
        y.and_(
            req_predicate(predicate, predicate_type),
            req_deverbal(require_deverbal_noun),
            req_reflexive(require_reflexive),
        )
    )


def create_argument_role(argument_type: str, case: str, preposition: str, **kwargs):
    rule_id = f"argument_type={argument_type},case={case},preposition={preposition}"
    arg = y.and_(req_argument(), req_animacy(argument_type), req_case(case))
    internal = y.or_(
        y.and_(yp.gram("ADJF"), y.or_(yp.normalized("этот"), yp.normalized("тот"))),
        y.not_(yp.gram("ADJF")),
    )

    rule = y.or_(
        y.rule(req_preposition(preposition), arg),
        y.rule(req_preposition(preposition), internal, arg),
    )
    return rule_id, rule


def req_deverbal(require_deverbal_noun: str = "?"):
    if require_deverbal_noun == "1":  ## strictly deverbal noun
        return y.and_(yp.gram("NOUN"), yp.in_caseless(deverbal_nouns))
    elif require_deverbal_noun == "0":  ## strictly regular verb
        return y.or_(yp.gram("VERB"), yp.gram("INFN"))
    elif require_deverbal_noun == "?":  ## anything
        return y.or_(
            y.and_(yp.gram("NOUN"), yp.in_caseless(deverbal_nouns)),
            yp.gram("VERB"),
            yp.gram("INFN"),
        )
    else:
        raise ValueError("Incorrect deverbal status")


def req_reflexive(reflexive_status: str = "?"):
    def is_reflexive_verb(verb: str):
        return verb.endswith("ся") or verb.endswith("сь")

    if reflexive_status == "1":
        return yp.custom(is_reflexive_verb)
    if reflexive_status == "0":
        return y.not_(yp.custom(is_reflexive_verb))
    elif reflexive_status == "?":
        return yp.true()
    else:
        raise ValueError("Incorrect reflexive status")


def req_animacy(animacy: str = "любой"):
    if animacy == "любой":
        return yp.true()
    elif animacy == "одуш.":
        return y.or_(
            y.not_(yp.gram("inan")), yp.gram("anim"), yp.gram("NPRO"), yp.gram("ADJF")
        )
    elif animacy == "неодуш.":
        return y.or_(yp.gram("inan"), yp.gram("anim"), yp.gram("NPRO"), yp.gram("ADJF"))
    else:
        raise ValueError("Incorrect Animacy Type")


def req_argument():
    return y.and_(
        y.not_(
            y.or_(  ## prohibits arguments from being any of following parts-of-speech
                yp.gram("PREP"),
                yp.gram("CONJ"),
                yp.gram("PRCL"),
                yp.gram("INTJ"),
                yp.gram("ADJF"),
            )
        ),
        y.or_(yp.gram("NOUN"), yp.gram("NPRO")),
    )


def req_predicate(word: str = "?", predicate_type: str = "глаг"):
    # add predicate_type handling
    if predicate_type == "глаг":
        predicate = y.or_(yp.gram("VERB"), yp.gram("INFN"))
    elif predicate_type == "сущ":
        predicate = y.or_(yp.gram("INFN"), yp.gram("NOUN"))
    elif predicate_type == "любой":
        predicate = y.or_(yp.gram("VERB"), yp.gram("INFN"), yp.gram("NOUN"))
    else:
        raise ValueError("predicate_type must be глаг or сущ or любой")
    if word != "?":
        if "|" not in word:
            # single-word scope
            predicate = y.and_(yp.normalized(word), predicate)
        else:
            predicate_words = word.split("|")
            scope_rule = list(map(yp.normalized, predicate_words))
            scope_rule = y.or_(*scope_rule)
            predicate = y.and_(scope_rule, predicate)

    return predicate


def req_case(case: str = "в"):
    if case == "в":
        pred = yp.gram("accs")
    elif case == "т":
        pred = yp.gram("ablt")
    elif case == "д":
        pred = yp.gram("datv")
    elif case == "р":
        pred = yp.gram("gent")
    elif case == "и":
        pred = yp.gram("nomn")
    elif case == "п":
        pred = yp.gram("loct")
    else:
        raise ValueError("Incorrect Case")

    return y.or_(pred)


def req_preposition(preposition: str = None):
    if preposition == "None":
        return y.empty()
    else:
        return y.or_(
            y.and_(yp.gram("PREP"), yp.eq(preposition)), y.not_(yp.gram("PREP"))
        )


def soft_parser_pass(parser, text):
    matches = []
    for match in parser.findall(text):
        matches.append(
            {
                "text": " ".join([x.value for x in match.tokens]),
                "span": tuple(match.span),
            }
        )

    return matches


def strict_parser_pass(parser, text):
    match = parser.match(text)
    matches.append(
        {"text": " ".join([x.value for x in match.tokens]), "span": tuple(match.span)}
    )

    return [match]


def create_rules(**kwargs):
    predicate_rule_id, predicate_rule = create_predicate_rule(**kwargs)
    argument_rule_id, argument_rule = create_argument_role(**kwargs)
    return {
        "predicate_id": predicate_rule_id,
        "argument_id": argument_rule_id,
        "predicate_parser": y.Parser(
            predicate_rule, yt.MorphTokenizer(morph=CachedMostProbMorphAnalyzer())
        ),
        "argument_parser": y.Parser(
            argument_rule, yt.MorphTokenizer(morph=CachedMostProbMorphAnalyzer())
        ),
    }


def check_parseable(text, parser):
    return len(list(parser.findall(text))) > 0


class ArgumentExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def extract(self, sentence: str) -> List[Dict[str, Any]]:
        pass


class PredPattArgumentExtractor(ArgumentExtractor):
    def __init__(
        self,
        path_to_udpipe: str,
        resolve_relcl: bool = True,
        resolve_appos: bool = True,
        resolve_amod: bool = True,
        resolve_conj: bool = True,
        resolve_poss: bool = True,
        ud=dep_v2.VERSION,
    ):
        super().__init__()
        self.model = Model.load(path_to_udpipe)
        self.pipeline = Pipeline(
            self.model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
        )
        self._error = ProcessingError()
        self._opts = PredPattOpts(
            resolve_relcl=resolve_relcl,
            resolve_appos=resolve_appos,
            resolve_amod=resolve_amod,
            resolve_conj=resolve_conj,
            resolve_poss=resolve_poss,
            ud=ud,
        )

    @lru_cache(maxsize=100000)
    def extract(self, sentence: str) -> List[Dict[str, Any]]:
        processed = self.pipeline.process(sentence, self._error)
        if self._error.occurred():
            print(f"=== Error occurred: {self._error.message}")
            self._error = ProcessingError()
            return None
        else:
            conll_example = [ud_parse for sent_id, ud_parse in load_conllu(processed)][
                0
            ]
            ppatt = PredPatt(conll_example, opts=self._opts)
            result = []
            for predicate in ppatt.instances:
                structure = {
                    "predicate": predicate.tokens,
                    "arguments": [x.tokens for x in predicate.arguments],
                }
                result.append(structure)

            return result


class MainPhraseExtractor:
    def __init__(self, syntax_parser, pymorphy_analyzer):
        self.syntax = syntax_parser
        self.morph = pymorphy_analyzer

    def get_main_phrase(self, words, get_prep=False, verbose=False):
        markup = next(self.syntax.map([words]))
        forward = {}
        backward = defaultdict(list)
        token_map = {}
        candidates = []
        for token in markup.tokens:
            if token.head_id not in backward:
                backward[token.head_id] = []

            token_map[token.id] = token
            forward[token.id] = token.head_id
            backward[token.head_id].append(token.id)

            if token.id == token.head_id or token.head_id == "0":
                candidates.append(token.id)

        if verbose:
            print("forward ", forward)
            print("backward ", backward)
            print("candidates ", candidates)

        if len(candidates) == 0:
            return markup.tokens

        candidate = sorted(candidates, key=lambda x: len(backward[x]))[-1]
        if get_prep:
            prep_candidates = backward[candidate]
            prep_candidates = list(
                filter(
                    lambda x: self.morph.tag(token_map[x].text)[0].POS == "PREP",
                    prep_candidates,
                )
            )
            if len(prep_candidates) == 0:
                return [token_map[candidate]]

            prep = sorted(prep_candidates, key=lambda x: abs(int(x) - int(candidate)))[
                0
            ]
            return (token_map[prep], token_map[candidate])

        return [token_map[candidate]]


class RstClauseSeparator:
    def __init__(
        self,
        udpipe=("tsa05.isa.ru", 3334),
        rst=("papertext.ru", 5555),
        cache_path="./rst-cache.pkl",
    ):
        udpipe_host, udpipe_port = udpipe
        rst_host, rst_port = rst
        self.cache_path = cache_path
        self.ppl = PipelineCommon(
            [
                (
                    ProcessorRemote(udpipe_host, udpipe_port, "0"),
                    ["text"],
                    {
                        "sentences": "sentences",
                        "tokens": "tokens",
                        "lemma": "lemma",
                        "syntax_dep_tree": "syntax_dep_tree",
                        "postag": "ud_postag",
                    },
                ),
                (
                    ProcessorMystem(delay_init=False),
                    ["tokens", "sentences"],
                    {"postag": "postag"},
                ),
                (
                    ConverterMystemToUd(),
                    ["postag"],
                    {"morph": "morph", "postag": "postag"},
                ),
                (
                    ProcessorRemote(rst_host, rst_port, "default"),
                    [
                        "text",
                        "tokens",
                        "sentences",
                        "postag",
                        "morph",
                        "lemma",
                        "syntax_dep_tree",
                    ],
                    {"clauses": "clauses"},
                ),
            ]
        )
        self.__cache = {}
        self.__hasher = city_32()
        if os.path.exists(self.cache_path):
            self.__cache = jb.load(self.cache_path)

    def extract(self, text):
        text_hash = self.__hasher(text)
        if text_hash in self.__cache:
            return self.__cache[text_hash]
        else:
            result = self.ppl(text)
            clauses = [x.text for x in result["clauses"]]
            self.__cache[text_hash] = clauses
            return clauses

    def flush(self):
        jb.dump(self.__cache, self.cache_path)


class RoleLabeler:
    def __init__(
        self,
        argument_extractor: ArgumentExtractor,
        main_phrase_extractor: MainPhraseExtractor,
        filter_pipeline,
        predicate_ruleset,
        mode: str = "soft",
        extend_arguments: bool = False,
    ):

        self.argument_extractor = argument_extractor
        self.main_phrase_extractor = main_phrase_extractor
        self.filter_pipeline = filter_pipeline
        self.ruleset = predicate_ruleset
        self.extend_arguments = extend_arguments
        if mode == "soft":
            self.pass_fn = soft_parser_pass
        elif mode == "strict":
            self.pass_fn = strict_parser_pass
        else:
            raise ValueError(f"Incorrect mode = {mode}, can be 'soft' or 'strict'")

    def check_parse(self, text, parser):
        return len(self.pass_fn(parser, text)) > 0

    def run(self, sentence):
        tokenized = map(lambda x: x.text, razdel.tokenize(sentence))
        words = list(map(lambda x: Token(x[0], x[1], None), enumerate(tokenized)))
        arg_groups = self.argument_extractor.extract(sentence)
        arg_groups = list(
            filter(
                lambda x: check_parseable(
                    " ".join([token.text for token in x["predicate"]]),
                    self.filter_pipeline,
                ),
                arg_groups,
            )
        )
        result = []
        for group in arg_groups:

            predicate_txt = " ".join([token.text for token in group["predicate"]])
            predicate_tokens = [token.text for token in group["predicate"]]
            predicate_main = " ".join(
                [
                    x.text
                    for x in self.main_phrase_extractor.get_main_phrase(
                        predicate_tokens
                    )
                ]
            )
            forward_map = {
                " ".join([token.text for token in argument]): argument
                for argument in group["arguments"]
            }
            group_name = (
                f"predicate={predicate_txt},arguments=[{','.join(forward_map.keys())}]"
            )
            group_result = []
            for predicate in self.ruleset.values():
                if self.check_parse(predicate_main, predicate["predicate_parser"]):
                    predicate_result = {
                        "predicate": predicate_txt,
                        "predicate_analyzed": predicate_main,
                        "predicate_tokens": group["predicate"],
                        "arguments": [],
                    }
                    for argument in forward_map.keys():
                        argument_tokens = [x.text for x in forward_map[argument]]
                        offset = min(x.position for x in forward_map[argument]) - 1
                        argument_main_phrase = (
                            self.main_phrase_extractor.get_main_phrase(
                                argument_tokens, True
                            )
                        )
                        argument_main = " ".join([x.text for x in argument_main_phrase])
                        argument_word = argument_main

                        token_positions = [
                            (offset + int(x.id)) for x in argument_main_phrase
                        ]
                        try:
                            if (
                                self.extend_arguments
                            ):  # extending argument with up to 2 previous tokens to ensure preposition included
                                min_pos = min(token_positions)
                                if min_pos >= 2:
                                    argument_main = f"{words[min_pos - 2].text} {words[min_pos - 1].text} {argument_main}"
                                elif min_pos == 1:
                                    argument_main = f"{words[0].text} {argument_main}"
                        except IndexError as e:
                            print(f"Index error for {token_positions} at {words}")

                        roles = [
                            rule["role"]
                            for rule in predicate["arguments"]
                            if self.check_parse(argument_main, rule["argument_parser"])
                        ]
                        if len(roles) > 0:
                            predicate_result["arguments"].append(
                                {
                                    "argument": argument,
                                    "argument_analyzed": argument_word,
                                    "argument_tokens": forward_map[argument],
                                    "roles": tuple(roles),
                                }
                            )
                    if len(predicate_result["arguments"]) > 0:
                        predicate_result["arguments"] = tuple(
                            predicate_result["arguments"]
                        )
                        group_result.append(predicate_result)
            result.append({"group": group_name, "parses": group_result})
        return result


class ConstraintEnforcer:
    def __init__(self, constraints=None):
        if constraints is None:
            constraints = list()

        self.constraints = constraints

    def add(self, constraint):
        self.constraints.append(constraint)

    def enforce(self, parse):
        a_parse = parse.copy()
        for constraint in self.constraints:
            a_parse = constraint(a_parse)
            if len(a_parse) == 0:
                return a_parse

        return a_parse


def enforce_parseable_predicate(parse):
    if check_parseable(parse["predicate_analyzed"], filter_pipeline):
        return parse
    else:
        return {}


def reduce_duplicate_roles(parse):
    new_args = []
    for arg in parse["arguments"]:
        arg["roles"] = tuple(set(arg["roles"]))
        new_args.append(arg)
    parse["arguments"] = new_args
    return parse


def resolve_multiple_expirirencers(parse):
    if len(parse["arguments"]) >= 2:
        parse_roles = set(arg["roles"] for arg in parse["arguments"])
        if ("экспериенцер",) in parse_roles:
            new_args = []
            for arg in parse["arguments"]:
                if len(arg["roles"]) >= 2:
                    new_roles = list(arg["roles"])
                    if "экспериенцер" in new_roles:
                        new_roles.remove("экспериенцер")
                    arg["roles"] = tuple(new_roles)
                new_args.append(arg)
            parse["arguments"] = new_args
    return parse


def resolve_single_expiriencer(parse):
    parse_roles = [arg["roles"] for arg in parse["arguments"] if len(arg["roles"]) >= 2]
    if len(parse_roles) > 0:
        n_exp = 0
        for role in parse_roles:
            if "экспериенцер" in role:
                n_exp += 1

        if n_exp == 1:
            new_args = []
            for arg in parse["arguments"]:
                if len(arg["roles"]) >= 2 and "экспериенцер" in arg["roles"]:
                    arg["roles"] = ("экспериенцер",)
                new_args.append(arg)
            parse["arguments"] = new_args
    return parse
