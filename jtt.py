import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gcn.utils import sample_mask

def identify_error_indices(pred_train, ind_train, y_data):
  mask_train = sample_mask(ind_train, y_data.shape[0])
  y_train = y_data[mask_train, 0]
  pred_disc = 1*(pred_train>0.5)[:, 0]
  errors = np.where(pred_disc != y_train)
  inds = ind_train[errors]
  if inds.shape[0] == 0:
    return None
  else:
    return inds

# due to k-fold, intersection is zero, so let's work with repeat counts instead
def intersections_analyse(up_indices, y, sex_labels, thresh=5, folds=10):
  indices_repeated = np.concatenate(up_indices)
  unique, counts = np.unique(indices_repeated, return_counts=True)

  _ = sns.histplot(counts, label='repeated counts', bins=folds-1)
  _ = plt.legend()
  plt.show()

  indices_max = unique[counts>=thresh]
  y_up = y[indices_max] # 1 asd, 2 neurotypical
  sex_up = sex_labels[indices_max] # 1 male, 2 female

  male_asd = sum((y_up == 1) * (sex_up == 1))[0]
  female_asd = sum((y_up == 1) * (sex_up == 2))[0]
  male_neur = sum((y_up == 2) * (sex_up == 1))[0]
  female_neur = sum((y_up == 2) * (sex_up == 2))[0]

  sns.barplot(['male asd', 'male neurot', 'female asd', 'female neurot'],
              [male_asd, female_asd, male_neur, female_neur])
  plt.title('Thresh ' + str(thresh))
  plt.show()

  male_asd = sum((y_up == 1) * (sex_up == 1))[0] / sum((y==1)*(sex_labels==1))[0]
  female_asd = sum((y_up == 1) * (sex_up == 2))[0] / sum((y==1)*(sex_labels==2))[0]
  male_neur = sum((y_up == 2) * (sex_up == 1))[0] / sum((y==2)*(sex_labels==1))[0]
  female_neur = sum((y_up == 2) * (sex_up == 2))[0] / sum((y==2)*(sex_labels==2))[0]

  sns.barplot(['male asd', 'male neurot', 'female asd', 'female neurot'],
              [male_asd, female_asd, male_neur, female_neur])
  plt.title('Thresh ' + str(thresh) + ' (fraction)')
  plt.show()