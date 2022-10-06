'''
Author: Yunxiang Liu u7191378@anu.edu.au
Date: 2022-10-03 20:49:52
LastEditors: Yunxiang Liu u7191378@anu.edu.au
LastEditTime: 2022-10-06 00:48:01
FilePath: \HoiTransformer\HOI_verb_GloveEmbbeding\load_npy.py
Description: word embedding
'''
import json
import numpy as np




if __name__ == "__main__":

    '''Load two .npy files as one list and one numpy array'''
    root_dir = 'HOI_verb_GloveEmbbeding'
    wordsList = np.load(f'{root_dir}/HOI_Verb_wordsList.npy')
    wordsList = wordsList.tolist()  # Originally loaded as numpy array
    wordVectors = np.load(f'{root_dir}/HOI_Verb_wordVectors.npy')

    '''one example'''
    theIndex = wordsList.index('adjust')
    print(wordVectors.shape)
    print(theIndex)
    print(wordVectors[theIndex])

    '''if you like, you can change them into a dictionary: key-value == word-vector'''
    embedding_dict={}
    for i in range(0, len(wordsList)):
        if 'lie-on' in wordsList[i]:
            print(wordsList[i])
        embedding_dict[wordsList[i]] = wordVectors[i]
    print(embedding_dict.keys())


