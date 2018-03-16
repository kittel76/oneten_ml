# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from search_keyword import dbio

DATA_DIR = "/data/www/oneten/data/search_data"
filename =  DATA_DIR + "/data.txt"

MAX_KEYWORD_LEN = 1000

# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with open(filename, "r") as f:
    data =f.read().split()

  return data

vocabulary = read_data(filename)

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 100000


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = []
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  # count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
# print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

conn = dbio.getConnection()
dataLen = len(count[:MAX_KEYWORD_LEN])
unitSize = 100

dataSize = int(dataLen / unitSize)
if dataLen % unitSize > 0:
  dataSize = dataSize + 1


dbio.truncate_search_sggt_prdnm_freq_d()
for i in range(0, dataSize):
  startIdx = unitSize * i
  endIdx = unitSize * (i + 1)
  if endIdx > (dataLen):
    endIdx = dataLen

  dbio.insert_user_item_rank_batch(count[startIdx:endIdx], conn)


conn.close()