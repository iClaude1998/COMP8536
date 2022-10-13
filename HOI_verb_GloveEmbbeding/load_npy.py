'''
Author: Yunxiang Liu u7191378@anu.edu.au
Date: 2022-10-03 20:49:52
LastEditors: Yunxiang Liu u7191378@anu.edu.au
LastEditTime: 2022-10-12 17:17:11
FilePath: \HoiTransformer\HOI_verb_GloveEmbbeding\load_npy.py
Description: word embedding
'''
import json
import numpy as np

hoi_interaction_names = json.loads(
    open('./data/hico/hico_verb_names.json', 'r').readlines()[0])['verb_names']


if __name__ == "__main__":

    '''Load two .npy files as one list and one numpy array'''
    root_dir = 'HOI_verb_GloveEmbbeding'
    wordsList = np.load(f'{root_dir}/HOI_Verb_wordsList.npy')
    wordsList = wordsList.tolist()  # Originally loaded as numpy array
    wordVectors = np.load(f'{root_dir}/HOI_Verb_wordVectors.npy')
    print(wordVectors.shape)

    '''one example'''
    theIndex = wordsList.index('adjust')
    print(wordsList)
    # print(wordVectors.shape)
    # print(theIndex)
    # print(wordVectors[theIndex])

    '''if you like, you can change them into a dictionary: key-value == word-vector'''
    embedding_dict={}
    # print(hoi_interaction_names)
    for i in range(0, len(wordsList)):
        # if 'lie-on' in wordsList[i]:
        #     print(wordsList[i])
        embedding_dict[wordsList[i]] = wordVectors[i].tolist()
    # embedding_dict['__background__'] = embedding_dict.pop('no_interaction')
    json_str = json.dumps(embedding_dict)
    with open('./data/hico/word_embeddings.json', 'w') as jf:
        jf.write(json_str)

    

    # word_dict = json.load(open('./data/hico/word_embeddings.json', 'r'))
    # print(word_dict['__background__'])
    


