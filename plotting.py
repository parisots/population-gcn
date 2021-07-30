import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np


def boxplots_acc(model):
  ''' TODO - Description
   Model: f_evaluate, m_evaluate, balanced, unbalanced, balanced_bootstrapping_StratBySex'''
  # Loading data
  results_folder = '/content/drive/My Drive/LOGML21/logml/results/'
  with open(results_folder + 'crossval_train_' + str(model) + '_both.pkl', 'rb') as f:
    data = pickle.load(f)

  # Graph config
  measures = ['acc', 'auc', 'acc_asd', 'acc_neurotypical']
  i = 1
  fig = plt.figure(figsize=(16, 4))
  fig.suptitle(str(model))
  for measure in measures:
    # print(i)
    fig.add_subplot(1, 4, i)
    temp = np.concatenate([[data[str(measure)]['female'], ['female']*10],
                          [data[str(measure)]['male'], ['male']*10],
                          [data[str(measure)]['overall'], ['overall']*10]], axis=1)
    data_plot = pd.DataFrame(data=temp.T, columns=['Score', 'Group'])
    data_plot['Score'] = data_plot['Score'].astype(float)
    ax = sns.boxplot(x="Group", y="Score", data=data_plot)
    plt.axhline(y=0.6, color='tab:gray', linestyle='--')
    plt.title(str(measure))
    plt.ylim([0, 1])
    i+=1
  plt.show()

models = ['f_evaluate', 'm_evaluate', 'balanced', 
          'unbalanced', 'balanced_bootstrapping_StratBySex']


def boxplots_bias(models):
  ''' models (list): List with models' names'''
  
  measure = 'bias'
  i = 1
  fig = plt.figure(figsize=(len(models)*4, 4))
  for model in models:
    # Loading data
    results_folder = '/content/drive/My Drive/LOGML21/logml/results/'
    with open(results_folder + 'crossval_train_' + str(model) + '_both.pkl', 'rb') as f:
      data = pickle.load(f)

    # Graph config
    fig.add_subplot(1, len(models), i)
    temp = np.concatenate([[data[str(measure)]['neurotypical'], ['neurotypical']*10],
                          [data[str(measure)]['asd'], ['asd']*10]], axis=1)
    data_plot = pd.DataFrame(data=temp.T, columns=['Score', 'Group'])
    data_plot['Score'] = data_plot['Score'].astype(float)
    ax = sns.boxplot(x="Group", y="Score", data=data_plot)
    plt.title(str(model))
    plt.ylim([0, .6])
    i+=1
  plt.show()