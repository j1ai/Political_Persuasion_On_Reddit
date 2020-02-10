import numpy as np
import argparse
import json
import re
import csv

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

PRONOUNS_TAGS = {'PRP','PRP$'}
PUNCTUATIONS = {'#', '$','.', ',',':', ':','(', ')','"', '‘', '“', '’','”', '?', '!', '-','/'}
COMMON_NOUNS = {'NN', 'NNS'}
PROPER_NOUNS = {'NNP', 'NNPS'}
ADVERBS = {'RB', 'RBR', 'RBS'}
WHWORDS = {'WDT', 'WP', 'WP$', 'WRB'}
FUTURE_TENSES = {"'ll", 'gonna'}
SKIPABLE_TOKENS = {'//SYM', '//NFP', '/_SP', '//,'}
BristolNormsDict = {}
RatingsWarrinerDict = {}
AltIDDict = {}
CenterIDDict = {}
LeftIDDict = {}
RightIDDict = {}
#Change this to CDF!!!
#AltFeats = np.load("C:\\Users\\LAI\\Desktop\\CSC401\\Political_Persuasion_On_Reddit\\feats\\Alt_feats.dat.npy")
#CenterFeats = np.load("C:\\Users\\LAI\\Desktop\\CSC401\\Political_Persuasion_On_Reddit\\feats\\Center_feats.dat.npy")
#LeftFeats = np.load("C:\\Users\\LAI\\Desktop\\CSC401\\Political_Persuasion_On_Reddit\\feats\\Left_feats.dat.npy")
#RightFeats = np.load("C:\\Users\\LAI\\Desktop\\CSC401\\Political_Persuasion_On_Reddit\\feats\\Right_feats.dat.npy")
AltFeats = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")
CenterFeats = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")
LeftFeats = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
RightFeats = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    feats = np.zeros(174)
    
    num_of_tokens_in_cur_sentence = 0
    num_of_tokens = 0
    num_of_sentence = 0
    num_of_words_in_BristolNormsDict = 0
    num_of_words_in_RatingsWarringerDict = 0
    num_of_going_to_in_this_comment = len(re.compile("go/vbg to/TO [a-zA-Z]+/VB |go/vbg to/in [a-zA-Z]+/VB ").findall(comment))
    feats[6] += num_of_going_to_in_this_comment
    #Num of multi-chars punctuation tokens
    num_of_multi_chars_punctuations = len(re.compile('([?!,;:\.\-`"]{2,})\/').findall(comment))
    feats[8] += num_of_multi_chars_punctuations
    aoa_Norms = []
    img_Norms = []
    fam_Norms = []
    vmean_norms = []
    amean_norms = []
    dmean_norms = []
    
    for token in comment.split():
        #print(token)
        word = ''
        tag = ''
        #Treat //SYM & //NFP as a punctuation
        #Skip this token if word token is multiple spaces (i.e /_SP)
        if token in SKIPABLE_TOKENS:
            continue
        #Multi Character Punctuation
        #if token[0] in PUNCTUATIONS and token[1] in PUNCTUATIONS and len(token) > 3:
            
        split_token = token.split('/')
        if len(split_token) == 2:
            word = split_token[0]
            tag = split_token[1]
        elif len(split_token) > 2:
            #Exclude last split, which is the tag
            word = word.join(split_token[0:-1])
            tag = split_token[-1]
        else:
            continue
        #Num of words in UpperCase
        if len(word) >= 3 and word.isalpha() and word.isupper():
            feats[0] += 1
        #Convert to word to lowercase
        word = word.lower()   
        #Bristol Norms Dictionary LookUp
        #print(word)
        #print(tag)
        if word in BristolNormsDict:
            aoa_Norms.append(BristolNormsDict[word][0])
            img_Norms.append(BristolNormsDict[word][1])
            fam_Norms.append(BristolNormsDict[word][2])
            num_of_words_in_BristolNormsDict += 1
        #Ratings Warringer Dictionary LookUp
        if word in RatingsWarrinerDict:
            vmean_norms.append(RatingsWarrinerDict[word][0])
            amean_norms.append(RatingsWarrinerDict[word][1])
            dmean_norms.append(RatingsWarrinerDict[word][2])
            num_of_words_in_RatingsWarringerDict += 1
        #Num of Sentence
        if tag == '.\n':
            num_of_sentence += 1
            continue
        #Length of all tokens
        if word not in PUNCTUATIONS:
            num_of_tokens += 1
            feats[15] += len(word)
        #Num of first-person pronouns
        if tag in PRONOUNS_TAGS: 
            if word.lower() in FIRST_PERSON_PRONOUNS:
                feats[1] += 1
                continue
            #Num of third-person pronouns
            if word.lower() in THIRD_PERSON_PRONOUNS:
                feats[3] += 1
                continue
        #Num of second-person pronouns
        if word in SECOND_PERSON_PRONOUNS:
            feats[2] += 1
            continue
        #Num of coordinating conjunctions
        if tag in 'CC':
            feats[4] += 1
            continue
        #Num of past tense verbs
        if tag in 'VBD':
            feats[5] += 1
            continue
        #Num of future tense verbs
        if (tag == 'MD' and word == 'will') or word in FUTURE_TENSES:
            feats[6] += 1
            continue
        #Num of commas
        if word == ',' and tag == ',':
            feats[7] += 1
            continue 
        #Num of common nouns
        if tag in COMMON_NOUNS:
            feats[9] += 1
            continue
        #Num of proper nouns
        if tag in PROPER_NOUNS:
            feats[10] += 1
            continue
        #Num of adverbs
        if tag in ADVERBS:
            feats[11] += 1
            continue
        #Num of WH words
        if tag in WHWORDS:
            feats[12] += 1
            continue
        #Num of SLANG 
        if word in SLANG:
            feats[13] += 1
            continue                
    #Average Length of sentences, in tokens
    if num_of_sentence > 0 and num_of_tokens > 0:
       feats[14] = num_of_tokens / num_of_sentence
    elif num_of_sentence == 0 and num_of_tokens > 0:
       feats[14] = num_of_tokens
    #Average Length of tokens
    if num_of_tokens > 0:
        feats[15] = feats[15] / num_of_tokens
    #Number of sentences
    feats[16] = num_of_sentence
    #Average & Standard Deviation of BristolNormsDict
    if num_of_words_in_BristolNormsDict > 0:
        feats[17] = np.average(aoa_Norms)
        feats[18] = np.average(img_Norms) 
        feats[19] = np.average(fam_Norms)
        feats[20] = np.std(aoa_Norms)
        feats[21] = np.std(img_Norms)
        feats[22] = np.std(fam_Norms)
    #Average & Standard Deviation of Rating Warringer
    if num_of_words_in_RatingsWarringerDict > 0:
        feats[23] = np.average(vmean_norms)
        feats[24] = np.average(amean_norms) 
        feats[25] = np.average(dmean_norms)
        feats[26] = np.std(vmean_norms)
        feats[27] = np.std(amean_norms)
        feats[28] = np.std(dmean_norms)
         
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.
    return feats
    
    
def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''
    featured_feats = feats    
    if comment_class == 'Alt':
        index = AltIDDict[comment_id]
        for i in range(0,144):
            featured_feats[29+i] += AltFeats[index][i]
        featured_feats[173] = 3
    elif comment_class == 'Center':
        index = CenterIDDict[comment_id]
        for i in range(0,144):
            featured_feats[29+i] = CenterFeats[index][i]
        featured_feats[173] = 1
    elif comment_class == 'Left':
        index = LeftIDDict[comment_id]
        for i in range(0,144):
            featured_feats[29+i] = LeftFeats[index][i]
        featured_feats[173] = 0
    elif comment_class == 'Right':
        index = RightIDDict[comment_id]
        for i in range(0,144):
            featured_feats[29+i] = RightFeats[index][i]
        featured_feats[173] = 2
    return featured_feats

def loadBristolNormsDict():
    #Change this to CDF !!!!!!!!!!!!
    #data = csv.DictReader(open('BristolNorms+GilhoolyLogie.csv'))
    data = csv.DictReader(open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'))
    for row in data:
        if row['WORD'] != '':
            entry = []
            entry.append(int(row['AoA (100-700)']))
            entry.append(int(row['IMG']))
            entry.append(int(row['FAM']))
            BristolNormsDict[row['WORD']] = entry

def loadRatingsWarriner():
    #Change this to CDF !!!!!!!!!!!!
    #data = csv.DictReader(open('Ratings_Warriner_et_al.csv'))
    data = csv.DictReader(open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'))
    for row in data:
        if row['Word'] != '':
            entry = []
            entry.append(float(row['V.Mean.Sum']))
            entry.append(float(row['A.Mean.Sum']))
            entry.append(float(row['D.Mean.Sum']))
            RatingsWarrinerDict[row['Word']] = entry

def loadIDDicts():
    #Change this to CDF !!!!!!!!!!!!
    #with open ("C:\\Users\\LAI\\Desktop\\CSC401\\Political_Persuasion_On_Reddit\\feats\\Alt_IDs.txt", "r") as altIDfile:
    with open ("/u/cs401/A1/feats/Alt_IDs.txt", "r") as altIDfile:
        counter = 0
        for line in altIDfile:
            AltIDDict[line.strip()] = counter
            counter += 1
    #with open ("C:\\Users\\LAI\\Desktop\\CSC401\\Political_Persuasion_On_Reddit\\feats\\Center_IDs.txt", "r") as centerIDfile:
    with open ("/u/cs401/A1/feats/Center_IDs.txt", "r") as centerIDfile:
        counter = 0
        for line in centerIDfile:
            CenterIDDict[line.strip()] = counter
            counter += 1
    #with open ("C:\\Users\\LAI\\Desktop\\CSC401\\Political_Persuasion_On_Reddit\\feats\\Left_IDs.txt", "r") as leftIDfile:
    with open ("/u/cs401/A1/feats/Left_IDs.txt", "r") as leftIDfile:
        counter = 0
        for line in leftIDfile:
            LeftIDDict[line.strip()] = counter
            counter += 1 
    #with open ("C:\\Users\\LAI\\Desktop\\CSC401\\Political_Persuasion_On_Reddit\\feats\\Right_IDs.txt", "r") as rightIDfile:
    with open ("/u/cs401/A1/feats/Right_IDs.txt", "r") as rightIDfile:
        counter = 0
        for line in rightIDfile:
            RightIDDict[line.strip()] = counter
            counter += 1     

def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    loadBristolNormsDict()
    loadRatingsWarriner()
    loadIDDicts()
    # TODO: Use extract1 to find the first 29 features for each 
    # data point. Add these to feats.
    comment_class = {"Alt", "Center", "Left", "Right"}
    comment_counter = 0
    for comment in data:
        cur_feats = extract1(comment['body'])
        commentID = comment['id']
        cat = comment['cat']
        # TODO: Use extract2 to copy LIWC features (features 30-173)
        # into feats. (Note that these rely on each data point's class,
        # which is why we can't add them in extract1).
        modified_feats = extract2(cur_feats, cat, commentID)
        feats[comment_counter] = modified_feats
        comment_counter += 1
    np.savez_compressed(args.output, feats)
    print('Features Extraction Done')	

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

