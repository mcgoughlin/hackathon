import pickle
import os

fp = '/Users/mcgoug01/Downloads/splits.pkl'
with open(fp, 'rb') as f:
    splits = pickle.load(f)
    print(splits)

for i in range(len(splits)):
    fold = splits[i]
    train = fold['train']
    val = fold['val']

    #whereever a filename ends in 'noised_0.7.nii.gz' replace it with 'noised.nii.gz'
    for j in range(len(train)):
        train[j] = train[j].replace('noised_0.7.npy', 'noised.npy')
    for j in range(len(val)):
        val[j] = val[j].replace('noised_0.7.npy', 'noised.npy')

    fold['train'] = train
    fold['val'] = val
    splits[i] = fold
    print(fold)


# save new splits file
with open('/Users/mcgoug01/Downloads/splits_noised.pkl', 'wb') as f:
    pickle.dump(splits, f)
