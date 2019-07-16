import pandas as pd

fullfilepath = './data/kddcup.traindata.corrected.csv'
tenpfilepath = './data/kddcup.traindata.corrected.csv'

fulldf = pd.read_csv(fullfilepath)
tenpdf = pd.read_csv(tenpfilepath)

fulllist = fulldf.values.tolist()
tenplist = tenpdf.values.tolist()

results = []

for value, index in enumerate(fulllist, 1):
    if value in tenplist:
        # results.append(index)
        print(index)
        
with open('results.txt', 'w') as f:
    for item in results:
        f.write("%s\n" % item)

print(results)