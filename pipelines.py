






class OpinionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_data, y_data=None):
        return self

    def transform(self, X_data, y_data=None):
        new_opinion_cat = {1: 'Low', 2: 'Low', 3: 'No Idea', 4: 'High', 5: 'High'}
        opinion_variables = [x for x in X_data.columns if x.startswith('opinion')]
        for var in opinion_variables:
            X_data.loc[:, var] = X_data.loc[:, var].map(new_opinion_cat)
        return X_data


class ScoringVariableCreation(BaseEstimator, TransformerMixin):

    def __init__(self, careful_columns, risk_group_columns):
        self.careful_columns = careful_columns
        self.risk_group_columns = risk_group_columns

    def fit(self, X_data, y_data=None):
        return self

    def transform(self, X_data, y_data=None):
        # Careful score
        X_data.loc[:, 'careful_score'] = X_data.loc[:, self.careful_columns].sum(axis=1)

        # Risk group score
        bool_old_people = X_data.loc[:, 'age_group'] == '65+ Years'
        X_data.loc[:, 'risk_group_score'] = X_data.loc[:, self.risk_group_columns].sum(axis=1) + bool_old_people
        return X_data



class TransformerPipeline(BaseEstimator, TransformerMixin):

    def __init__(self, cat_threshold):
        self.cat_threshold = cat_threshold

    def fewer_than_cutoff(column_name, data, n):
        series = data.loc[:, column_name]
        num_categories = len(series.value_counts())
        return num_categories <= n

    def fit(self, X_data, y_data=None):
        self.less_than_six = [x for x in feature_columns if self.fewer_than_cutoff(x, X_data, self.cat_threshold)]
        self.more_than_five = list(set(X_data.columns) - set(features_fewer_than_six))
        return self

    def transform(self, X_data, y_data):
