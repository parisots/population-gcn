# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, , Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from nilearn import datasets

import ABIDEParser as Reader
import os
import shutil


# Selected pipeline
pipeline = 'cpac'

# Input data variables
num_subjects = 871  # Number of subjects
# Input data variables
# root_folder = '/users/tomdavies/Documents/Southampton/code/logml/population-gcn/data/'
root_folder = '/content/drive/My Drive/LOGML21/logml/data/'
# root_folder = root_path +  'logml/data'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')

# Files to fetch
files = ['rois_ho']

filemapping = {'func_preproc': 'func_preproc.nii.gz',
               'rois_ho': 'rois_ho.1D'}

if not os.path.exists(data_folder): 
  os.makedirs(data_folder)
  print('new folder made')
# shutil.copyfile('./subject_IDs.txt', os.path.join(data_folder, 'subject_IDs.txt'))
shutil.copyfile('population-gcn/subject_IDs.txt', os.path.join(data_folder, 'subject_IDs.txt'))

# Download database files
abide = datasets.fetch_abide_pcp(data_dir=root_folder, n_subjects=num_subjects, pipeline=pipeline,
                                 band_pass_filtering=True, global_signal_regression=False, derivatives=files)


subject_IDs = Reader.get_ids(num_subjects)
subject_IDs = subject_IDs.tolist()


# Create a folder for each subject
for s, fname in zip(subject_IDs, Reader.fetch_filenames(subject_IDs, files[0])):
    subject_folder = os.path.join(data_folder, s)

    if not os.path.exists(subject_folder):
        os.mkdir(subject_folder)

    # Get the base filename for each subject
    base = fname.split(files[0])[0]

    # Move each subject file to the subject folder
    for fl in files:
        print(subject_folder)
        print(base)
        print(filemapping[fl])

        if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
            shutil.move(base + filemapping[fl], subject_folder)


time_series = Reader.get_timeseries(subject_IDs, 'ho')

# Compute and save connectivity matrices
for i in range(len(subject_IDs)):
        Reader.subject_connectivity(time_series[i], subject_IDs[i], 'ho', 'correlation')

