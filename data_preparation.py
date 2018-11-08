import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import preprocessing

TEST_SIZE = 0.3

# Make Y and X train and test sets
train = pd.read_csv('train.csv')
Y = train['OpenStatus']
# X_test = pd.read_csv('test.csv')
X = train.drop('OpenStatus', axis=1)

# TODO: analyze dates
# But for now simply drop them
X = X.drop(['PostClosedDate', 'PostCreationDate', 'OwnerUserId', 'PostId'], axis=1)


# prepare account creation dates
def date_string_to_timestamp(date):
    try:
        return datetime.strptime(date, "%m/%d/%Y %H:%M:%S").timestamp()
    except ValueError:  # some dates are in different format
        return datetime.strptime(date, "%Y-%m-%d").timestamp()


creation_timestamps = X["OwnerCreationDate"].map(date_string_to_timestamp)
X["OwnerCreationDate"] = (creation_timestamps - creation_timestamps.mean()) / creation_timestamps.std()

# TODO: analyze titles and bodyMarkdowns considering tags and not only
# But for now simply drop them

text_X = X[['Title', 'BodyMarkdown']]
X = X.drop(['Title', 'BodyMarkdown'], axis=1)

# Creating new column counting tags (more tags -> better post)
for tag_index in ['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']:
    X[tag_index] = X[tag_index].map(lambda x: "" if isinstance(x, float) else x)

tags = (X['Tag1'] + " " + X['Tag2'] + " " + X['Tag3'] + " " + X['Tag4'] + " " + X['Tag5']).str.strip()
text_X['Tags'] = pd.Series(tags)

X['CountTags'] = tags.map(lambda tags_str: len(tags_str.split(" ")))
X = X.drop(['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5'], axis=1)


def get_normalised_numerical_data():
    X_scaled = preprocessing.MinMaxScaler().fit_transform(X.values)
    return train_test_split(pd.DataFrame(X_scaled), Y, test_size=TEST_SIZE)


def get_numerical_data():
    return train_test_split(X, Y, test_size=TEST_SIZE)


def get_text_data():
    return train_test_split(text_X, Y, test_size=TEST_SIZE)
