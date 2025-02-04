import re
from pathlib import Path


def load_stopwords():
    f = Path("data/stopwords.txt")
    if f.exists():
        return set([line.replace('\n', '') for line in open(str(f), 'r', encoding='utf-8').readlines()])
    return set()

#********************************************************************************************************

def is_digit(word: str):
    w = re.sub(r'[$]?[-+]?[\d]*[.,\:]?[\d]+[ %\"\'\)\+]*', "", word)
    return not w

def is_complex_digit(word: str):
    w = re.sub(r'[$]?[-+]?[\d]*[.,\:]?[\d]+[ %\"\'\)\+]*[A-Za-z0-9]?', "", word)
    return not w

def is_date(value: str):
    # check formats: DD:MM:YYYY, DD.MM.YYYY, DD-MM-YYYY, DD/MM/YYYY
    pattern = r"^\d{1,2}(:|\.|\-|\/)\d{1,2}\1\d{4}$"
    return re.match(pattern, value) != None


# original modified: #removed defis
def str_tokenize_words(s: str):
    s = re.findall("(\.?\w[\w'\.&]*\w|\w\+*#?)", str.lower(s))
    if s: return s
    return []

"""
def str_tokenize_words(s: str):
    s = re.findall("(\.?\w[\w'\.&-]*\w|\w\+*#?)", s)
    if s: return s
    return [] 
"""

def str_tokenize(sentence, stopwords: set):
    return [ word for word in str_tokenize_words(sentence.lower()) if word not in stopwords ] 


#********************************************************************************************************

def test_grammar():

    print(is_date("00.01.2000"), is_date("00-01-2000"), is_date("00/01/2000"), is_date("00:01:2000"))

    d_test = [ "160", "160)", "160.0", "+160", "+160.0", "$0.2%", "$.225%", "$.225%", 
                "$.225%", "$.225%%", "$+.225%", "$,225%", "$:225%", "$+55%%%" ]
    for i in d_test: print(is_digit(i))

    for i in d_test: print(is_complex_digit(i + "v"))

    s = "John's mom went there, but he wasn't know c++, c#, .net, Q&A/Q-A, #nope i_t IT at-all'. So' she said: 'Where are& viix.co. !!' 'A a'"
    
    list_1 = str_tokenize_words(s)

    print(list_1)


"""
if __name__ == "__main__":
    test_grammar()
"""
