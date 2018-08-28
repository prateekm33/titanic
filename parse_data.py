import csv
import numpy as np

def parse_row(row, _type="train"):
  if _type == "train":
    is_female = row[5] == "female"
    row[5] = int(is_female)
    try:
      row.append(int(is_female) + 1 / int(row[6]))
    except:
      row.append(int(is_female))
    try:
      row[11] = ord(row[11][0])
    except:
      row[11] = 0
    try:
      row[12] = ord(row[12])
    except:
      row[12] = 0
    
    row.append(float(row[10]) * float(row[2]))
    _row = row[2:3] + row[5:9] + row[10:]

  elif _type == "test":
    is_female = row[4] == "female"
    row[4] = int(is_female)
    try:
      row.append(int(is_female) + 1 / int(row[4]))
    except:
      row.append(int(is_female))
    try:
      row[10] = ord(row[10][0])
    except:
      row[10] = 0
    try:
      row[11] = ord(row[11])
    except:
      row[11] = 0
    
    row.append(float(row[9]) * float(row[1]))
    _row = row[1:2] + row[4:8] + row[9:]


  _row = [float(el or 0) for el in _row]
  return _row

def parse_data(file, _type="train"):
  with open(file, newline='') as csvfile:
    testreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    X = []
    Y = []
    for i, row in enumerate(testreader):
      if i == 0:
        continue
      if _type == "train":
        _row = parse_row(row, "train")
        X.append(_row)
        Y.append(int(row[1]))
      elif _type == "test":
        _row = parse_row(row, "test")
        X.append(_row)
    X = np.array(X)
    Y = np.array(Y, ndmin = 2)
    if _type == "train":
      return X, Y
    elif _type == "test":
      return X

parse_data('train.csv')