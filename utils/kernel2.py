import numpy as np
import pandas as pd
import textblob as tb

def slove_tags_count(row):
    """
    tags count
    """
    return pd.Series([tag[1] \
        for tag in tb.TextBlob(row).tags]).value_counts()

def slove_len(row):
    """
    sentence length
    """
    return len(tb.TextBlob(row).tags)

def tagging_data(path):
    """
    Make tagging
    """
    print('\tRead data')
    frame = pd.read_csv(path).dropna()
    
    print('\tConvert question1')
    quest_1 = frame.ix[:]['question1'].apply(slove_tags_count)
    quest_1.to_csv('./quest_1.csv', sep=',')
    
    print('\tConvert question2')
    quest_2 = frame.ix[:]['question2'].apply(slove_tags_count)
    quest_2.to_csv('./quest_2.csv', sep=',')
    
    print('\tSlove result')
    result = quest_1/quest_2
    
    result['length'] = frame.ix[:]['question1'].apply(slove_len)/\
        frame.ix[:]['question2'].apply(slove_len)
    
    result.to_csv('./fichers_part_one.csv', sep=',')

def mark_broken_question(path):
    """
    Set marker for question with bad content
    """
    print('\tRead data')
    frame = pd.read_csv(path, index_col=[0]).fillna(0)

    print('\tSlove bad rows')
    bad_indexes = frame[frame.T.sum() == 0].index.values
    np.append(bad_indexes, frame[frame.T.sum() == 1].index.values)
    frame['is_broken'] = 0
    frame.set_value(bad_indexes, 'is_broken', 1)

    print('\tSave result')
    frame.to_csv('./with_bad_questions_marker.csv', sep=',')


if __name__ == '__main__':
#    path = './train.csv'
#    tagging_data(path)
