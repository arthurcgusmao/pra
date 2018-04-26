import pandas as pd
import numpy as np
from tools.feature_matrices import parse_feature_matrices
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score

target_relations = ['cause_of_death',
                    'ethnicity',
                    'gender',
                    'institution',
                    'nationality',
                    'profession',
                    'religion']

def get_reasons(row):
    reasons = row[row != 0]
    output = pd.Series()
    counter = 1
    for reason, relevance in reasons.iteritems():
        output['reason' + str(counter)] = reason
        output['relevance' + str(counter)] = relevance
        counter = counter + 1
        if counter == 10:
            break
    for i in range(counter, 10):
        output['reason' + str(i)] = "n/a"
        output['relevance' + str(i)] = "n/a"
    return output


class Explanator(object):
    def __init__(self, target_relation):
        self.target_relation = target_relation
        # Define the model
        w_l1 = 0.75
        w_l2 = 0.05
        l1_ratio = w_l1 / (w_l1 + w_l2)
        # alpha = w_l1 + w_l2
        alpha = 0.0001

        self.model = SGDClassifier(loss="log", penalty="elasticnet", alpha=alpha, l1_ratio=l1_ratio,
                                   max_iter=100000, tol=1e-3, class_weight="balanced")


    def extract_data(self):
        data_path = './' + self.target_relation
        train_matrix_fpath = data_path + "/train.tsv"
        test_matrix_fpath = data_path + "/test.tsv"
        self.train_data, self.test_data = parse_feature_matrices(train_matrix_fpath, test_matrix_fpath)

        # separate x (features) and y (labels)
        self.train_x = self.train_data.drop(['head', 'tail', 'label'], axis=1)

        self.train_y = self.train_data['label']
        self.test_x = self.test_data.drop(['head', 'tail', 'label'], axis=1)
        self.test_y = self.test_data['label']


    def train(self):
        """ Train and evaluate the model """
        self.extract_data()
        # Train the model
        self.model.fit(self.train_x, self.train_y)

        self.accuracy = self.model.score(self.test_x, self.test_y)
        self.precision_score = precision_score(self.test_y, self.model.predict(self.test_x))
        self.recall_score = recall_score(self.test_y, self.model.predict(self.test_x))
        self.f1_score = f1_score(self.test_y, self.model.predict(self.test_x))

    def explain_per_example(self, data_type):
        coefficients = self.model.coef_.reshape(-1,1)
        if data_type == 'train':
            x = self.train_x
            y = self.train_y
            data = self.train_data
        else:
            x = self.test_x
            y = self.test_y
            data = self.test_data
        final_reasons = pd.DataFrame()
        final_reasons['head'] = data['head']
        final_reasons['tail'] = data['tail']
        repeated_coefficients = np.repeat(coefficients.T, x.shape[0], axis=0)
        weighted_x = x.apply(pd.to_numeric)
        explanations = weighted_x.mul(repeated_coefficients, axis=1)
        motives = explanations.apply(get_reasons, axis=1)
        final_reasons = pd.concat([final_reasons, motives], axis=1)
        answers = self.model.predict_proba(x)[:, 1]
        final_reasons['y_hat'] = answers
        final_reasons['y'] = y
        final_reasons.to_csv('./' + self.target_relation + '/' + self.target_relation + '.csv')
        return final_reasons

    def explain(self):
        """ Explain the model using the coefficients """
        # Extract the coefficients
        self.coefficients = normalize(abs(self.model.coef_), norm='l1', axis=1).reshape(-1,1)

        self.explanation = pd.DataFrame(self.coefficients, columns=['scores'])
        self.explanation['path'] = self.train_x.columns
        self.explanation = self.explanation.sort_values(by="scores", ascending=False)

        self.relevant_relations = self.explanation[self.explanation['scores'] > 0].shape[0]
        self.total_relations = self.explanation.shape[0]

    def report(self):
        file_path = './' + self.target_relation + '/' + self.target_relation + '_explained.txt'
        np.savetxt(file_path, self.explanation.iloc[0:15].values, fmt= '%s')
        with open(file_path, 'a') as f:
            f.write("\n---------------------------------------------")
            f.write("\nAccuracy: %0.2f" % self.accuracy)
            f.write("\nPrecision: %0.2f" % self.precision_score)
            f.write("\nRecall: %0.2f" % self.recall_score)
            f.write("\nF1_score: %0.2f" % self.f1_score)
            f.write("\n---------------------------------------------")
            f.write("\n" + str(self.model.get_params()))

if __name__ == '__main__':
    for target_relation in target_relations:
        print("Training on " + target_relation + " relations")
        exp = Explanator(target_relation)
        exp.train()
        exp.explain()
        exp.report()
        exp.explain_per_example('test')
