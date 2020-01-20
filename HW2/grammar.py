"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
Student: Nick Gupta, UNI: ng2528
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        
        for gram_rules in self.lhs_to_rules.gram_rules():
            all_words = self.lhs_to_rules[gram_rules]
            prob_of_word = []
            for word in all_words:
        #using only right hand side and assuming that both RHS and LHS are consistent.
                right_side = word[1]
                #right_side = right hand side; it has nothing to do with right or wrong.
                if not(len(right_side) == 1 or len(right_side) == 2):
                    return False
                    break
            if(len(right_side) == 2):
                if not (right_side[0].isupper() and right_side[1].isupper()):
            #checks if nonterminal symbols are upper-case or not
                    return  False
                    break
            elif(len(right_side) == 1):
                if not(right_side[0].islower()):
            #terminal symbols are lower case
                    return False
                    break
            prob_of_word.append(word[2])
            round(fsum(prob_of_word), 1)
            #rounds the probability upto 1
            #for example, probability of 0.9 gets rounded to 1.0
            if fsum(prob_of_word) != 1.0:
                #checks if the tatal pobability is 1
                return False
                #returns false if the total probability is 1
                break
                return True
                #returns true if the total probability is not 1
        


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        result = grammar.verify_grammar()
        print(result)
       
