import pandas as pd
import numpy as np
import time
from utils2.utils import remove_nb
from ast import literal_eval


def load_users(_dir, post_size=100, max_posts=2000, nb_depress=500, return_df=False):
    depress = pd.read_csv(_dir + 'final_replace_depress0220.csv', encoding='utf-8')
    control = pd.read_csv(_dir + 'final_replace_control0220.csv', encoding='utf-8')
    # depress = pd.read_csv(_dir + 'final_replace_depress0220token.csv', encoding='utf-8')
    # control = pd.read_csv(_dir + 'final_replace_control0220token.csv', encoding='utf-8')

    print('The number of control: ', len(control.userid.unique()))
    print('The number of depress: ', len(depress.userid.unique()))
    print('Control shape: ', control.shape)
    print('Depress shape: ', depress.shape)

    print('Selecting depressed users with at least {} posts'.format(post_size))

    # start = time.time()
    # included_users = pd.DataFrame(control.groupby(['userid']).size())[
    #     pd.DataFrame(control.groupby(['userid']).size())[0]
    #     >= post_size].index
    # control = control[control.userid.isin(included_users)]
    # print('Control users: ', len(included_users))
    # del included_users

    # remove numbers
    control['clean'] = control['clean'].apply(remove_nb)
    # control['clean'] = control['clean'].apply(literal_eval)
    print('-----------{}-----------'.format(time.time() - start))

    start = time.time()
    included_users = pd.DataFrame(depress.groupby(['userid']).size())[
        pd.DataFrame(depress.groupby(['userid']).size())[0]
        >= post_size].index
    depress = depress[depress.userid.isin(included_users)]
    np.random.seed(0)
    included_users = np.random.choice(depress.userid.unique(), size=nb_depress, replace=False)
    depress = depress[depress.userid.isin(included_users)]
    print('Depress users: ', len(included_users))
    del included_users

    # remove numbers
    depress['clean'] = depress['clean'].apply(remove_nb)
    # depress['clean'] = depress['clean'].apply(literal_eval)
    print('-----------{}-----------'.format(time.time() - start))

    print('Labelling')
    control['label'] = 0
    depress['label'] = 1

    combine, labels = combine_users(control, depress)

    if return_df:
        return combine, labels

    return select_post(combine, labels, max_posts)


def combine_users(control, depress):
    combine = pd.concat([control, depress])
    combine.sort_values(by=['userid'], inplace=True)

    combine = combine.reset_index(drop=True).copy()

    labels = combine[['userid', 'label']].drop_duplicates().reset_index(drop=True).copy()

    assert (combine.userid.unique() == labels.userid.values).all()

    return combine, labels


def select_post(combine, labels, max_posts):
    print('Selecting {} posts from each user'.format(max_posts))
    start = time.time()
    inputs = []
    for userid in combine.userid.unique():
        post = combine[combine.userid == userid].clean.head(max_posts).values
        # sort by ascending
        inputs.append(post)
    print('-----------{}-----------'.format(time.time() - start))

    return inputs, labels


def select_post_ana(combine, labels, max_posts):
    print('Selecting {} posts from each user'.format(max_posts))
    start = time.time()
    inputs = []
    i_inputs = []
    for userid in combine.userid.unique():
        post = combine[combine.userid == userid].clean.head(max_posts).values
        # sort by ascending
        i = combine[combine.userid == userid][['i', 'we', 'you', 'shehe', 'they', 'ipron']].head(max_posts).values
        inputs.append(post)
        i_inputs.append(np.array(i))
        # i_inputs.append(np.array(i).reshape(-1, 1))
    print('-----------{}-----------'.format(time.time() - start))
    return inputs, i_inputs, labels
