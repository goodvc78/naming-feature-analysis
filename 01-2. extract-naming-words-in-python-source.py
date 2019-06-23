
# coding: utf-8

# ### Python souce를 parsing하고 변수명을 추출한다. 

# * sample : tensorflow (https://github.com/tensorflow/tensorflow)
# * 참고 : https://www.python.org/dev/peps/pep-0008/#naming-conventions

# In[ ]:

import os
import re
import pandas as pd 
get_ipython().magic('matplotlib inline')
import re 


# * source down load list 
# <pre>
# git clone https://github.com/tensorflow/tensorflow
# git clone https://github.com/django/django.git
# git clone https://github.com/scikit-learn/scikit-learn.git
# git clone https://github.com/pallets/flask.git
# git clone https://github.com/ansible/ansible.git
# git clone https://github.com/odoo/odoo.git
# git clone https://github.com/scrapy/scrapy.git
# git clone https://github.com/rg3/youtube-dl.git
# git clone https://github.com/kennethreitz/requests.git
# git clone https://github.com/tornadoweb/tornado.git
# git clone https://github.com/fchollet/keras.git
# git clone https://github.com/ipython/ipython.git
# git clone https://github.com/pandas-dev/pandas.git
# git clone https://github.com/numpy/numpy.git
# git clone https://github.com/matplotlib/matplotlib.git
# git clone https://github.com/mwaskom/seaborn.git
# git clone https://github.com/networkx/networkx.git
# git clone https://github.com/scipy/scipy.git
# 
# </pre>

# In[ ]:

def readSource(path):
    try :
        with open(path) as f:
            content = f.read()
        return content
    except :
        return 'file exception'
            
    return ''


def removeComments(source):
    """
    replacde '/* */' and '//' style comments
    reference : http://blog.ostermiller.org/find-comment 
    comment의 열림만 있고 닫힘이 없는 경우 regex에서 hang이 걸리는 경우가 발생함 
    이름 방지하기 위해 사용 
    """
    source += '*/'

    # java comment
    # p = re.compile('(\/\*([^*]|[\r\n]|(\*([^/]|[\r\n])))*\*\/)|(\/\/.*)')
    p = re.compile('(\"((?:.|\n)*?)\")|(#.*)')
    
    output = p.sub("", source)
    
    return output


def cleaningSource(source):
    ## remove ccomments 
    source = removeComments(source)
    
    ## remove "\r"
    source = source.replace("\r","")
    
    ## split by lines
    source_lines = source.split("\n")
    
    return source_lines

def parseSourceLines(lines):
    variable_names = []
    variable_set = set()
    for line in lines:
        vals = parseLine(line)
        if  (vals==None): 
            continue
        
        for val in vals :
            ## unique check
            if val[0] not in variable_set:
                variable_names.append(val)
                variable_set.add(val[0])
    return variable_names


# In[ ]:

## test code 

sample_path = "/Users/goodvc/Data/naming-recsys/resource/python/tensorflow/tensorflow/python/framework/common_shapes.py"
sample_src = readSource(sample_path)

cleaningSource(sample_src)


# In[ ]:

print(sample_src)


# In[ ]:

reserved_words = set('and del from not while as elif global or with assert else if pass yield break except default abstract import print class exec in raise continue finally is return def for lambda try '.split())     


# In[ ]:

## Java 예약어 
#reserved_words = set('abstract default package synchronized boolean do if private this break double implements protected throw byte else import public throws switch enum instanceof return try catch extends int short char final interface static void class finally long strictfp volatile float native super while continue for new case goto* null transient const operator future generic ineer outer rest var from'.split())

def variableValidation(name):
    name = name.lower()
    ## check reserved words 
    if name in reserved_words:
        return False
    
    ## check start-char is number
    if name[0].isnumeric():
        return False
    
    ## check test code's name
    if name.find("test") > -1:
        return False
    
    ## __{keyword}__ pattern 
    if name[:2]=='__' and name[-2:]=='__':
        return False
    
    return True


# In[ ]:

##### Variable Extract Rule ###############################

def ev_equal_rule(line):
    """
    변수는 변할수 있는 값이라는 전제로 값을 변하게 하는 equal(=) 연산자의 left token을 변수를 함 
    """
    line = line.replace('==',' ')
    equal_pos = line.find('=')
    if equal_pos < 0:
        return None
    line = ''.join( c if (c.isalnum()) | (c=='_') else ' ' for c in line[:equal_pos] )
    val = line.split()[-1]
    if len(val)>0:
        return [val]
    return None

def ev_equal_regex_rule(line):
    """ 
    '=' 이전에 공백이나 탭이 0~3개 까지 올수 있고 
    '=' 이전에 A-Z|a-z|0-9|_ 로 구성된 문자가 오고 
    '=' 다음에 '<>!' 문자가 안오는 경우 
    """
    regex = r"([A-Za-z0-9\_]+)[ \t]{0,3}\=[^<>!]"
    line += ' '
    matches = re.finditer(regex, line)
    names = []
    for match in matches:
        names.append(match.group(1))
    return names


##### Class Extract Rule ###############################
def ec_prefix_regex_rule(line):
    """
    'class' 단어가 오고 
    ' '이 1글자 이상오고  
    다음에 a-z, A-Z, _, 0-9 글자로 이루어진 단어를 class명으로 추출 
    """
    regex = r"(class) {1,3}([a-zA-Z_0-9]+)"
    line += ' '
    matches = re.finditer(regex, line)
    names = []
    for match in matches:
        names.append(match.group(2))
        
    return names 

def ev_regex_rule(line, regex, groupid):
    line += ' '
    matches = re.finditer(regex, line)
    names = []
    for match in matches:
        names.append(match.group(groupid))
    return names 

## 함수명을 추출하는 rule, 단순히 '('가 후미에 있고 단어가 a-z, A-Z, 0-9, _ 구성됨
#ef_bracket_regex_rule = lambda line : ev_regex_rule(line, r' {1,3}([a-zA-Z0-9]+) {0,3}\(', 1)
ef_def_regex_rule = lambda line : ev_regex_rule(line, r'(def) {1,3}([a-zA-Z_0-9]+)', 2)


# In[ ]:

def ef_braket_regex_rule(line):
    """
    단순히 '('가 후미에 있고 단어가 a-z, A-Z, 0-9, _ 구성됨
    """
    regex = r"{1,3}([a-zA-Z0-9]+) {0,3}\("
    line += ' '
    matches = re.finditer(regex, line)
    names = []
    for match in matches:
        names.append(match.group(1))
    return names


# In[ ]:

###################################################
def extractVariable(line):
    ## parse rule list 
    parse_rules = [
        ('equal:rule', 'variable', ev_equal_regex_rule), ## equal rule by regex 
        ('class:rule', 'class', ec_prefix_regex_rule), ##
        ('function:rule', 'function', ef_def_regex_rule), ##
    ]
    
    val_names = []
    for (rule, val_type, parsor) in parse_rules:
        names = parsor(line)
        if (names == None):
            continue
        for name in names:
            val_names.append([name,val_type]) 
    
    return val_names

def extractIntent(line):
    intent_char = ' \t'
    pos = 0
    for ch in line:
        if ch not in intent_char:
            break
        pos = pos + (4 if ch == '\t' else 1)
    return pos

def parseLine(line):
    ## check intent
    intent = extractIntent(line)
    
    ## extract variables 
    vals = extractVariable(line)
    if (vals == None) | (len(vals) == 0): 
        return None
    
    ret = []
    for val in vals:
        ## check name validation
        if False==variableValidation(val[0]):
            continue
        ## add intent value 
        val.append(intent)
        ret.append(val)
    return ret


# In[ ]:

def parseSourceCode(src_path):
    source = readSource(src_path)
    source = cleaningSource(source)
    parsed = parseSourceLines(source)
    
    return parsed


# In[ ]:

## test code 
print(sample_path)
parseSourceCode(sample_path)[:3]


# In[ ]:

import fnmatch
import os

def findFiles(path, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

def parseSourceDir(filenames):
    pared_variables = []

    for path in filenames:
        pared_variables.extend(parseSourceCode(path))
    return pared_variables


# In[ ]:

def charType(ch):
    if ch=='_':
        return 'underbar'
    
    if ch.isnumeric():
        return 'numeric'
    
    if not ch.isalpha():
        return 'unknown'
    
    if ch.islower():
        return 'lower'
    else :
        return 'upper'

    return 'unknown'

def chLevel(ch):
    if ch.islower():   return 1
    if ch.isupper():   return 2
    if ch.isnumeric(): return 3
    if ch=='_':        return 4
    if ch=='$':        return 0
    return 5


# In[122]:

def removeNumber(word):
    removed = ''
    for ch in word:
        if not ch.isnumeric():
            removed += ch
    return removed


## https://en.wikipedia.org/wiki/Naming_convention_(programming)#Java 
def tokenizer(sentance):
    """ 
    naming된 sentance를 단어로 tokenizing 한다. 
    java naming convention에 기반하여 check
    UpperCamelCase, lowerCamelCase, lower_delimiter_case, UPPER_DELIMITER_CASE 
    위의 4가지 naming convention 으로 tokenize 실행 
    """
    ### inspection split char posision 
    old_level = 0
    split_pos = []
    last_pos = 0
    for pos, ch in enumerate(sentance):
        cur_level = chLevel(ch)
        if (cur_level < old_level) & (pos>1) :  ## lower edge
            if old_level == 3: 
                split_pos.append(last_pos)
            elif (pos - last_pos)<2:
                split_pos.append(last_pos)
            else :
                split_pos.extend([last_pos,pos-1])
            last_pos = pos
        elif ( cur_level > old_level ): ## upper edge
            #print('set')
            last_pos = pos
        old_level = cur_level
        

    ### word split 
    last_pos = 0
    words = []
    split_pos.append(pos+1)
    for pos in split_pos:
        if sentance[last_pos]=='_':
            last_pos += 1
        w = removeNumber(sentance[last_pos:pos])
        if len(w)>0:
            words.append( w )
        last_pos=pos
    return words 


print(tokenizer('parseDBMXMLFromIPAddress'))


# In[123]:

testSentance = ['test100', 'tokenStats', 'ActiveMQQueueMarshaller', 
                'parseDBMXMLFromIPAddress', 'TestMapFile', 'TEST_NUMVER_AA', 'my_number']
for word in testSentance:
    print(tokenizer(word))


# In[ ]:

for word in testSentance:
    print('^'+word)
    print(''.join(['0']+[ str(chLevel(ch)) for ch in word]))
    


# ### 데이터 수집 및 랭클링 작업 
# * github에서 popular fork repo download  

# In[ ]:

variable_data = {}
variable_meta = {}


# In[ ]:

from os import listdir
from os.path import isfile, join, isdir

home_dir = "/Users/goodvc/Data/naming-recsys/resource/source/python"
            

def checkDirAndParse(output_data, ouput_meta):
    ## check dir 
    base_dir = home_dir
    folders = [f for f in listdir(base_dir) if isdir(join(base_dir, f))]
    
    for topic in folders:
        if topic in output_data:
            continue
        filenames = findFiles( os.path.join(home_dir, topic) , '*.py')
        print(" %s topic %d files parsing start" % (topic, len(filenames)))
        ouput_meta[topic] = {'file_count':len(filenames)}
        output_data[topic] = parseSourceDir(filenames)
    print("parse end")

## 
checkDirAndParse(variable_data, variable_meta)


# In[ ]:

## parsing 한 repository 기본 정보 
repo_meta_ds = pd.DataFrame.from_dict(variable_meta, orient='index').reset_index()
repo_meta_ds.columns = 'topic file_cnt'.split()


# In[ ]:

## 변수 Tokenize 
def tokenizeList(data_list):
    token_list = []
    for val in data_list:
        tokens = tokenizer(val[0])
        token_list.extend([[token, token.lower(),val[1], val[3]] for token in tokens if len(token)>0 ])
    return token_list
## 1207277


# In[ ]:

## variable data merge
data_list = []
for topic in variable_data.keys():
    for info in variable_data[topic]:
        data_list.append( info + [topic] )

## variable data to dataframe 
name_ds = pd.DataFrame(data_list, columns=['name', 'kind', 'intent', 'topic'])

## 단어단위로 분리하고 데이터 셋 만들기 
naming_words = tokenizeList(data_list)
words_ds = pd.DataFrame(naming_words, columns=['word', 'lower', 'kind', 'topic'])


# In[ ]:

unique_name_ds = name_ds.groupby(['name', 'kind']).count().reset_index()[['name','kind','intent']]


# In[ ]:

import nltk

## POS Tagging Dataframe 생성
analyzered = []
for (idx,row) in unique_name_ds.iterrows():
    word = row['name']
    tokens = tokenizer(word)
    tokens = [ token.lower() for token in tokens ]
    tagged = "+".join([ pos for (w,pos) in nltk.pos_tag(tokens) ])
    analyzered.append([len(tokens), tokens, tagged, row['intent']])

pos_tagged_ds = pd.DataFrame(analyzered, columns=['len','tokens', 'pos', 'count'])
pos_tagged_ds['kind'] = unique_name_ds.kind


# In[ ]:

source_type='python'


# In[ ]:

## 데이터셋 저장 
name_ds.to_pickle('./resource/{type}_name_ds.pkl'.format(type=source_type))
words_ds.to_pickle('./resource/{type}_words_ds.pkl'.format(type=source_type))
repo_meta_ds.to_pickle('./resource/{type}_repo_ds.pkl'.format(type=source_type))
pos_tagged_ds.to_pickle('./resource/{type}_pos_tagged_ds.pkl'.format(type=source_type))


# ---

# In[ ]:

## sample 테스트 
words = tokenizer('parseDBMXMLFromIPAddress')
words = [w.lower() for w in words]
print( [ "%s[%s]" %(w,pos) for (w,pos) in nltk.pos_tag(words)])


# ### 추출된 네이밍 데이터 셋 

# In[ ]:

name_ds[102000:102005]


# In[ ]:

pos_tagged_ds[3000:3005]


# In[ ]:



