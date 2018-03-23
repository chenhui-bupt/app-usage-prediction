# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

SEED = 222
np.random.seed(SEED)

def eval_all():
    data1 = pd.read_csv('./data/Network-Fingerprint-Dataset/data0117.csv', encoding='gbk', usecols=['id', 'app'])
    data2 = pd.read_csv('./data/Network-Fingerprint-Dataset/data0118.csv', encoding='gbk', usecols=['id', 'app'])
    data3 = pd.read_csv('./data/Network-Fingerprint-Dataset/data0119.csv', encoding='gbk', usecols=['id', 'app'])
    data4 = pd.read_csv('./data/Network-Fingerprint-Dataset/data0120.csv', encoding='gbk', usecols=['id', 'app'])

    nb_data1 = pd.read_csv('./output/node/nodesim1.csv',encoding='gbk')
    nb_data2 = pd.read_csv('./output/node/nodesim2.csv',encoding='gbk')
    lp_data1 = pd.read_csv('./output/path/lp1.csv',encoding='gbk')
    lp_data2 = pd.read_csv('./output/path/lp2.csv',encoding='gbk')
    lsp_data1 = pd.read_csv('./output/path/lsp1.csv',encoding='gbk')
    lsp_data2 = pd.read_csv('./output/path/lsp2.csv',encoding='gbk')
    rwr_data1 = pd.read_csv('./output/randomwalk/rwr1.csv',encoding='gbk')
    rwr_data2 = pd.read_csv('./output/randomwalk/rwr2.csv',encoding='gbk')
    rwrr_data1 = pd.read_csv('./output/randomwalk/rwrr1.csv',encoding='gbk')
    rwrr_data2 = pd.read_csv('./output/randomwalk/rwrr2.csv',encoding='gbk')

    def get_edges(data):
        groups = data.groupby(['id','app']).groups
        edges = {g for g in groups}
        return edges

    edges1 = get_edges(data1)
    edges2 = get_edges(data2)
    edges3 = get_edges(data3)
    edges4 = get_edges(data4)

    col1 = ['CN','JC','AA','RA','PA','CS','LHN','HP','HD','SI']
    col2 = ['CN_2','JC_2','AA_2','RA_2','PA_2','CS_2','LHN_2','HP_2','HD_2','SI_2']
    nb11 = np.mat(nb_data1.loc[:,col1])
    nb12 = np.mat(nb_data1.loc[:,col2])
    nb21 = np.mat(nb_data2.loc[:,col1])
    nb22 = np.mat(nb_data2.loc[:,col2])
    nb1 = (nb11+nb12)/2.0
    nb2 = (nb21+nb22)/2.0
    cp1 = (np.mat(nb_data1['CP'])+np.mat(nb_data1['CP_2']))/2.0
    cp2 = (np.mat(nb_data2['CP'])+np.mat(nb_data2['CP_2']))/2.0
    cp1 = cp1.reshape(-1,1)
    cp2 = cp2.reshape(-1,1)
    lp1 = np.mat(lp_data1['LP']+nb_data1['CN_2']).reshape(-1,1)
    lp2 = np.mat(lp_data2['LP']+nb_data2['CN_2']).reshape(-1,1)
    lsp1 = np.mat(lsp_data1['LSP']).reshape(-1,1)
    lsp2 = np.mat(lsp_data2['LSP']).reshape(-1,1)
    rwr1 = np.mat(rwr_data1['RWR']).reshape(-1,1)
    rwr2 = np.mat(rwr_data2['RWR']).reshape(-1,1)
    rwrr1 = np.mat(rwrr_data1['RWRR']).reshape(-1,1)
    rwrr2 = np.mat(rwrr_data2['RWRR']).reshape(-1,1)
    NLR1 = np.column_stack((nb1,lp1,rwr1))
    NLR2 = np.column_stack((nb2,lp2,rwr2))

    xtrain = nb1
    xtest = nb2
    ytrain = nb_data1['link'].values
    ytest = nb_data2['link'].values

    unknown_edges = edges2 - edges1
    # unknown_edges = edges2 - edges1
    def unknown_unexist_data(data):
        unknown_data = list(filter(lambda x: (x[0], x[1]) in unknown_edges, np.array(data)))
        unknown_data = pd.DataFrame(unknown_data, columns=data.columns)
        unexist_data = data.loc[data['link'] == 0].iloc[:len(unknown_data), :]
        data = pd.concat([unknown_data, unexist_data], axis=0, ignore_index=True)
        return data

    def regard_similarity_as_model_directly_and_eval():
        # evaluate Neighbor based similarity
        NB_data = unknown_unexist_data(nb_data1)
        col1 = ['CN', 'JC', 'AA', 'RA', 'PA', 'CS', 'LHN', 'HP', 'HD', 'SI', 'CP']
        col2 = ['CN_2', 'JC_2', 'AA_2', 'RA_2', 'PA_2', 'CS_2', 'LHN_2', 'HP_2', 'HD_2', 'SI_2', 'CP_2']
        cols = col1 + col2
        for col in col1:
            print(col, roc_auc_score(NB_data['link'], NB_data[col] + NB_data[col + '_2']))

        # evaluate LP,LSP,RWR and RWRR
        LP = unknown_unexist_data(lp_data1)
        print(roc_auc_score(LP['link'], LP['LP']))
        LSP = unknown_unexist_data(lsp_data1)
        print(roc_auc_score(LSP['link'], LSP['LSP']))
        RWR = unknown_unexist_data(rwr_data1)
        print(roc_auc_score(RWR['link'], RWR['RWR']))
        RWRR = unknown_unexist_data(rwrr_data1)
        print(roc_auc_score(RWRR['link'], RWRR['RWRR']))

    # ### 1. Neighbor based similarities
    def eval_neighbor_based():
        # nb每一对特征
        cols = ['CN', 'JC', 'AA', 'RA', 'PA', 'CS', 'LHN', 'HP', 'HD', 'SI'] + ['CP']
        ytrain = nb_data1['link'].values
        ytest = nb_data2['link'].values
        Eval = pd.DataFrame()
        for col in cols:
            print(col)
            xtrain = (nb_data1[col] + nb_data1[col + '_2']).reshape(-1, 1)
            xtest = (nb_data2[col] + nb_data2[col + '_2']).reshape(-1, 1)
            eval_df = train_eval()
            Eval = Eval.append(pd.concat([pd.Series([col] * 5, name='similarity'), eval_df], axis=1))

        Eval.to_csv('./output/modelresult2/eval_nb.csv', index=False)
        return Eval

    def plot_neighbor_based():
        Eval = pd.read_csv('./output/modelresult2/eval_nb.csv')
        Eval.head()

        full_name = {
            'PA': 'Preferential Attachment', 'CN': 'Common Neighbors', 'CS': 'Salton', 'JC': 'Jaccard',
            'AA': 'Adamic-Adar', 'RA': 'Resource Allocation', 'SI': 'Sorence', 'LHN': 'Leicht-Holme-Newman',
            'HD': 'Hub Depressed Index', 'HP': 'Hub Promoted Index', 'CP': 'Bidirectional Conditional Probability',
        }

        data = {}
        for i in range(len(Eval)):
            e = Eval.iloc[i, :]
            data.setdefault(e[1], dict())
            #     if e[0]=='SI':
            #         e[0] = 'Sorence'
            e[0] = full_name[e[0]]
            data[e[1]][e[0]] = e[6]

        # xticks = ['PA','CN','CS','JC','AA','RA','Sorence','LHN','HD','HP','CP']
        xticks = ['Preferential Attachment', 'Common Neighbors', 'Salton', 'Jaccard', 'Adamic-Adar',
                  'Resource Allocation',
                  'Sorence', 'Leicht-Holme-Newman', 'Hub Depressed Index', 'Hub Promoted Index',
                  'Bidirectional Conditional Probability']
        df = pd.DataFrame(data).loc[xticks]

        ax = plt.gca()
        ax.set_xlabel('Neighbor based similarity')
        ax.set_ylabel('AUC')
        df.plot(ax=ax, xticks=range(11), legend='reverse', ylim=[0.12, 1.0], rot=90, fontsize=5)
        plt.savefig('./output/pics3/neighbors_based.jpg', dpi=600, bbox_inches='tight')
        plt.savefig('./output/pics3/neighbors_based.eps', dpi=600, bbox_inches='tight')
        plt.savefig('./output/pics3/neighbors_based.pdf', dpi=600, bbox_inches='tight')
        plt.show()

    # ### 2. Path based similarities
    def eval_path_based():
        Eval = pd.DataFrame()
        sim_data = {
            'lp': [lp1, lp2],
            'lsp': [lsp1, lsp2],
            'rwr': [rwr1, rwr2],
            'rwrr': [rwrr1, rwrr2]
        }
        for name, sim in sim_data.items():
            xtrain, xtest = sim
            eval_df = train_eval()
            Eval = Eval.append(pd.concat([pd.Series([name.upper()] * 5, name='similarity'), eval_df], axis=1))

        Eval.to_csv('./output/modelresult2/eval_path.csv', index=False)
        return Eval

    # figure-path-based
    def plot_path_based():
        Eval = pd.read_csv('./output/modelresult2/eval_path.csv')

        full_name = {
            'LP': 'Local Path', 'LSP': 'Local Shortest Path',
            'RWR': 'Random Walk with Restart', 'RWRR': 'Random Walk with Resource Redistribution',
        }

        data = {}
        for i in range(len(Eval)):
            e = Eval.iloc[i, :]
            data.setdefault(e[1], dict())
            #     e[0] = full_name[e[0]]
            e[0] = e[0]
            data[e[1]][e[0]] = e[6]

        # xticks = ['Local Path', 'Local Shortest Path', 'Random Walk with Restart', 'Random Walk with Resource Redistribution']
        xticks = ['LP', 'LSP', 'RWR', 'RWRR']
        df = pd.DataFrame(data).loc[xticks]

        ax = plt.gca()
        ax.set_xlabel('Path based similarity')
        ax.set_ylabel('AUC')
        df.plot(ax=ax, xticks=range(4), legend='reverse')
        plt.savefig('./output/pics3/path_based2.jpg', dpi=600, bbox_inches='tight')
        plt.savefig('./output/pics3/path_based2.eps', dpi=600, bbox_inches='tight')
        plt.savefig('./output/pics3/path_based2.pdf', dpi=600, bbox_inches='tight')
        plt.show()

    # ### 4. Feature combination
    def eval_feature_combination():
        Eval = pd.DataFrame()
        sim_data = {
            'cp': [cp1, cp2],
            'lp': [lp1, lp2],
            'lsp': [lsp1, lsp2],
            'rwr': [rwr1, rwr2],
            'rwrr': [rwrr1, rwrr2]
        }
        comb_name = 'NB'
        ytrain = nb_data1['link'].values
        ytest = nb_data2['link'].values
        xtrain = nb1
        xtest = nb2
        eval_df = train_eval()
        Eval = Eval.append(pd.concat([pd.Series([comb_name] * 5, name='Methods'), eval_df], axis=1))
        for name in ['cp', 'lp', 'lsp', 'rwr', 'rwrr']:
            comb_name += '+' + name.upper()
            print(comb_name)
            train, test = sim_data[name]
            xtrain = np.hstack([xtrain, train])
            xtest = np.hstack([xtest, test])
            eval_df = train_eval()
            Eval = Eval.append(pd.concat([pd.Series([comb_name] * 5, name='Methods'), eval_df], axis=1))

        Eval.to_csv('./output/modelresult2/feature_comb.csv', index=False)

    def eval_feature_combination2():
        Eval = pd.DataFrame()
        sim_data = {
            'nb': [nb1, nb2],
            'nb+cp': [np.hstack([nb1, cp1]), np.hstack([nb2, cp2])],
            'nb+cp+lp': [np.hstack([nb1, cp1, lp1]), np.hstack([nb2, cp2, lp2])],
            'nb+cp+lsp': [np.hstack([nb1, cp1, lsp1]), np.hstack([nb2, cp2, lsp2])],
            'nb+cp+lp+lsp+rwr': [np.hstack([nb1, cp1, lp1, lsp1, rwr1]), np.hstack([nb2, cp2, lp2, lsp2, rwr2])],
            'nb+cp+lp+lsp+rwrr': [np.hstack([nb1, cp1, lp1, lsp1, rwrr1]), np.hstack([nb2, cp2, lp2, lsp2, rwrr2])],
            'nb+lp+rwr': [np.hstack([nb1, lp1, rwr1]), np.hstack([nb2, lp2, rwr2])],
            'nb+cp+lp+lsp+rwr+rwrr': [np.hstack([nb1, cp1, lp1, lsp1, rwr1, rwrr1]),
                                      np.hstack([nb2, cp2, lp2, lsp2, rwr2, rwrr2])],
        }
        ytrain = nb_data1['link'].values
        ytest = nb_data2['link'].values
        for name in ['nb', 'nb+cp', 'nb+cp+lp', 'nb+cp+lsp', 'nb+cp+lp+lsp+rwr', 'nb+cp+lp+lsp+rwrr',
                     'nb+lp+rwr', 'nb+cp+lp+lsp+rwr+rwrr']:
            print(name)
            xtrain, xtest = sim_data[name]
            eval_df = train_eval()
            Eval = Eval.append(pd.concat([pd.Series([name.upper()] * 5, name='Methods'), eval_df], axis=1))

        Eval.to_csv('./output/modelresult2/feature_comb2.csv', index=False)
        return Eval

    def get_models():
        """Generate a library of base learners."""
        nb = GaussianNB()
        knn = KNeighborsClassifier()
        lr = LogisticRegression(penalty='l1')
        rf = RandomForestClassifier(n_estimators=10, random_state=SEED)
        xgb = XGBClassifier()
        models = {
            'knn': knn,
            'naive bayes': nb,
            'logistic': lr,
            'random forest': rf,
            'xgb': xgb,
        }
        return models

    # (Pred == Prob.apply(lambda p: 1*(p > 0.5))).sum() # 所以阈值为大于0.5
    def train_predict(models):
        """Fit models in list on training set and return preds"""
        P = np.zeros((ytest.shape[0], len(models)))
        P = pd.DataFrame(P)  # probability dataframe
        print("Fitting models.")
        cols = list()
        for i, (name, m) in enumerate(models.items()):
            print("%s..." % name, end=" ", flush=False)
            m.fit(xtrain, ytrain)
            P.iloc[:, i] = m.predict_proba(xtest)[:, 1]
            cols.append(name)
            print("done")
        P.columns = cols
        print("Done.\n")
        return P

    def score_models(P, y):
        """Score model in prediction DF"""
        print("Scoring models.")
        eval_df = pd.DataFrame(np.zeros((len(P.columns), 6)),
                               columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
        for i, m in enumerate(P.columns):
            auc = roc_auc_score(y, P.loc[:, m])
            pred = 1 * (P.loc[:, m] > 0.5)  # predict class
            accuracy = accuracy_score(y, pred)
            precision = precision_score(y, pred)
            recall = recall_score(y, pred)
            f1 = f1_score(y, pred)
            print("%-26s: %f, %f, %f, %f, %f" % (m, accuracy, precision, recall, f1, auc))
            eval_df.iloc[i, :] = [m, accuracy, precision, recall, f1, auc]
        return eval_df
        print("Done.\n")

    def train_eval():
        models = get_models()
        P = train_predict(models)
        eval_df = score_models(P, ytest)
        return eval_df

    # regard similarity as model directly and evaluate it
    regard_similarity_as_model_directly_and_eval()

    # evaluate neighbor based similarity and plot the result
    eval_neighbor_based()
    plot_neighbor_based()

    # evaluate path based similarity and plot the result
    eval_path_based()
    plot_path_based()

    # evaluate the features combination of all simialrity features
    eval_feature_combination()
    eval_feature_combination2()


