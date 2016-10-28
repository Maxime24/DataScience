
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[13]:

import xgboost as xgb


# In[16]:

get_ipython().magic('pinfo xgb.DMatrix')


# # Importation des données

# In[2]:

train_fname = '/Users/Anailove/Documents/DataChallenge/train.csv'
test_fname = '/Users/Anailove/Documents/DataChallenge/test.csv'
df = pd.read_csv(train_fname, sep=';',na_values='(MISSING)')
df_test = pd.read_csv(test_fname, sep=';',na_values='(MISSING)')
y_train = df.VARIABLE_CIBLE == 'GRANTED'
df['SOURCE_BEGIN_MONTH']=df['SOURCE_BEGIN_MONTH'].fillna('D0')
df.shape


# In[3]:

df.head(5)


# In[4]:

#S'occuper des données en string=DONE
#Enlever les features qui veulent rien dire=DONE
#Regarder les paramètres du classifieur=DONE
#Regarder les features une à une=DONE
#S'occuper des valeurs manquantes= DONE
#S'occuper des dates (les transformer)=TODO
#Changer de classifieur?
#Donner des poids plus importants à des features


# # Raffinement des features (numériques, catégorielles, dates)

# In[5]:

feature_number=['APP_NB', 'APP_NB_PAYS','APP_NB_TYPE','NB_CLASSES', 'NB_ROOT_CLASSES','NB_SECTORS', 'NB_FIELDS','INV_NB', 'INV_NB_PAYS', 'INV_NB_TYPE'
               ,'cited_n', 'cited_nmiss', 'cited_age_min','cited_age_median', 'cited_age_max', 'cited_age_mean', 'cited_age_std',
               'NB_BACKWARD_NPL', 'NB_BACKWARD_XY','NB_BACKWARD_I', 'NB_BACKWARD_AUTRE', 'NB_BACKWARD_PL', 'NB_BACKWARD',
               'pct_NB_IPC', 'pct_NB_IPC_LY', 'oecd_NB_ROOT_CLASSES','oecd_NB_BACKWARD_PL', 'oecd_NB_BACKWARD_NPL', 'IDX_ORIGIN','IDX_RADIC']

feature_date=["PRIORITY_MONTH","FILING_MONTH","PUBLICATION_MONTH", 'BEGIN_MONTH']

feature_names = ["COUNTRY","FISRT_APP_COUNTRY","FISRT_APP_TYPE","LANGUAGE_OF_FILLING","TECHNOLOGIE_SECTOR",
        "TECHNOLOGIE_FIELD","FISRT_INV_COUNTRY","FISRT_INV_TYPE",'VOIE_DEPOT','SOURCE_BEGIN_MONTH',"SOURCE_CITED_AGE",
                 'SOURCE_IDX_ORI','SOURCE_IDX_RAD',
                "MAIN_IPC",'FIRST_CLASSE']


# In[6]:

nb_classes_withoutFC=[94, 143, 5, 29, 5, 35, 147, 5, 2, 2, 2, 2, 2, 630]


# # Preprocessing sur les dates

# Changer les dates en nombres croissants avec la chronologie, un couple (mois,année) unique sera encodé de manière unique.
# Cela vient du fait notamment qu'il était plus "facile" d'obtenir un brevet il y a quelques décennies.

# In[7]:

def convert(s):
    if not pd.isnull(s):
        r = s.split('/')
        return int(r[0]) + 12 * int(r[1])
    else:
        return np.nan


for i in feature_date:
    df[i] = df[i].apply(convert)


# In[8]:

df['FILING_MONTH']


# In[9]:

for i in feature_date:
    df_test[i] = df_test[i].apply(convert)


# # Label-Encoding des variables catégorielles

# Les algorithmes de classification ne reconnaissant pas (encore?) les variables catégorielles de type string, il faut convertir en numérique. Ici pour une feature avec n occurences, label encoding transforme en un nombre de 0 à n-1

# In[10]:

outlier=['at','jp','us']
for i in outlier:
    dfna=df['FISRT_INV_COUNTRY']==i
    index=dfna[dfna==True].index
    df['FISRT_INV_COUNTRY'][index]=i.upper()
    
for i in outlier:
    dfna=df_test['FISRT_INV_COUNTRY']==i
    index=dfna[dfna==True].index
    df_test['FISRT_INV_COUNTRY'][index]=i.upper()


# In[11]:

from sklearn.preprocessing import LabelEncoder



X_pays = df.COUNTRY
X_pays_test = df_test.COUNTRY
le = LabelEncoder()
le.fit(X_pays)
tab_pays = le.classes_

for i in tab_pays[1:] : 
    count = 0 
    Y = np.where((X_pays == i) & (df.VARIABLE_CIBLE=='GRANTED'))
    Z = np.asarray(np.where(X_pays == i))             
    count = float(len(Y[0]))
    count_tot = float(len(Z[0]))
    X_pays[Z[0]] = count/count_tot
    
    Z_test = np.asarray(np.where(X_pays_test == i))
    X_pays_test[Z_test[0]] = count/count_tot 
    
# Ces 4 pays 'GT' , 'CY' , 'AM' , 'LK' ne sont pas présentes dans le train : on met le rapport = 0 
Z_GT = np.asarray(np.where(X_pays_test == 'GT'))
X_pays_test[Z_GT[0]] = 0 
Z_CY = np.asarray(np.where(X_pays_test == 'CY'))
X_pays_test[Z_CY[0]] = 0 
Z_AM = np.asarray(np.where(X_pays_test == 'AM'))
X_pays_test[Z_AM[0]] = 0 
Z_LK = np.asarray(np.where(X_pays_test == 'LK'))
X_pays_test[Z_LK[0]] = 0 
    
    

X_train = np.c_[X_pays,X_train]
X_test = np.c_[X_pays_test,X_test]



# In[ ]:

tab_pays


# In[ ]:

#Convertir les strings en non string pour les données en entier
from sklearn.preprocessing import LabelEncoder
classes=[]

for i in feature_names:
    enc=LabelEncoder()
    #index
    dfna=df[i].isnull()
    index1=dfna[dfna == False].index
    dfna=df_test[i].isnull()
    index2=dfna[dfna == False].index
    #fitting
    fit1=df[i][index1].values
    fit2=df_test[i][index2].values
    conc=np.concatenate((fit1,fit2))
    enc.fit(conc)
    classes.append(enc.classes_)
    #transform
    df[i][index1]=enc.transform(df[i][index1]) 
    df_test[i][index2]=enc.transform(df_test[i][index2]) 
    


# # Traitement des variables numériques et ajout des variables de date

# Il n'y a quasi rien à faire pour les variables numériques. On concatène ensuite toutes les features et le preprocessing est fini une fois les données scalées. Au lieu d'imputer, on va fillna 0 => pas amélioré le score

# In[ ]:

features =[]
for i in feature_number:
    features.append(i)

for i in feature_date:
    features.append(i)

X_inc=df[features].values
X_inc_test= df_test[features].values

from sklearn.preprocessing import Imputer

imputer = Imputer()
X_inc = imputer.fit_transform(X_inc)

imputer1 = Imputer()
X_inc_test = imputer1.fit_transform(X_inc_test)


# # Traiter les variables catégorielles

# D'abord FIRST_CLASSE qui est la plus diversifiée de toutes

# A cause des trop nombreuses occurences différentes, j'ai été obligé de garder seulement celles qui apparaissaient plus de 30 fois.

# In[ ]:

X_hot=[]
X_hot_test=[]
count=0


for cat in feature_names:
        
    counts = pd.value_counts(pd.concat([df[cat], df_test[cat]], axis = 0))    
   
    #Essayer 60/30/20 (ce qui fait beaucoup moins de features et un arbre pas trop profond pour ne pas overfitter)
    
    if cat == 'FIRST_CLASSE':
        columns_to_keep = counts[counts > 80].index
    elif cat == 'MAIN_IPC' :
        columns_to_keep = counts[counts > 50].index
    else:
        columns_to_keep = counts[counts > 20].index
  
    
    mask = ~df[cat].isin(columns_to_keep)
    df[cat][mask] = "-"    
        
    dummies = pd.get_dummies(df[cat]).values
    print(dummies.shape)
    
    if(count==0):
        X_hot=dummies
    else:
        X_hot=np.concatenate((X_hot,dummies),axis=1)
    
    mask = ~df_test[cat].isin(columns_to_keep)
    df_test[cat][mask] = "-"
    dummies_test = pd.get_dummies(df_test[cat]).values
    print(dummies_test.shape)
    
    if(count==0):
        X_hot_test=dummies_test
    else:
        X_hot_test=np.concatenate((X_hot_test,dummies_test),axis=1)
    
    count+=1
     


# In[ ]:

nb_classes_withoutFC=[94, 143, 5, 29, 5, 35, 147, 5, 2, 2, 2, 2, 2, 630]


# ## Pour XGBOOST

# In[ ]:

#Concaténer les features one hot encodées avec le reste
X_train=np.concatenate((X_inc,X_hot),axis=1)
X_test=np.concatenate((X_inc_test,X_hot_test),axis=1)


# In[ ]:

#Scale les données, essayer sans, pas nécessaire mais peut booster l'algo
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)


# In[ ]:

X_train.shape


# In[ ]:

X_test.shape


# ## Fin du preprocessing

# # Cross validation et classification 

# In[ ]:

from sklearn import cross_validation

X1, X2, y1, y2 = cross_validation.train_test_split(X_train,y_train, test_size=0.2,random_state=1)


# J'utilise xgboost blabla bla bla

# In[ ]:

import xgboost as xgb


# In[ ]:

dtrain = xgb.DMatrix(X1,label=y1)
dtest = xgb.DMatrix(X2,label=y2)
dtraintot = xgb.DMatrix(X_train,label=y_train)


# In[ ]:

param={}
param = {'max_depth':10,'eta':0.1, 'silent':1, 'objective':'binary:logistic',
         'lambda':3,'alpha':3,'min_child_weight':0.5}
param['nthread'] = 4
param['eval_metric'] = 'auc'

evallist  = [(dtest,'eval'), (dtrain,'train')]


#param['subsample']=0.8
#param['colsample_bytree']=0.8
#param['scale_pos_weight']=0.3


# In[ ]:




# In[ ]:

xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed = 0, show_stdv = False)


# numround optimal around 500 (0.7194)

# In[ ]:

numround=600
bst = xgb.train(param, dtrain, numround,evallist)
ypred = bst.predict(dtest)
#bst1 = xgb.train(param, dtraintot,numround)
#y_pred_train=bst1.predict(dtraintot)


# In[ ]:

from sklearn.metrics import roc_auc_score

#print('Score (optimiste) sur le train : %s' % roc_auc_score(y_train, y_pred_train))

print(roc_auc_score(y2,ypred))


# In[ ]:

dtestvrai= xgb.DMatrix(X_test)
ypredtest=bst.predict(dtestvrai)
np.savetxt('y_pred.txt', ypredtest, fmt='%s')


# In[ ]:

df.head(5)


# In[ ]:

X_train


# # Utilisation de hyperopt pour trouver les meilleurs paramètres

# In[ ]:

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials 
from sklearn.metrics import roc_auc_score


# In[ ]:

def objective(space):

    param={}
    param = {'max_depth':int(space['max_depth']),'eta':space['eta'], 'silent':0, 'objective':'binary:logistic',
         'lambda':space['lambda'],'alpha':space['alpha'],'min_child_weight':space['min_child_weight']}
    param['nthread'] = 6
    param['eval_metric'] = 'auc'
    param['subsample']=space['subsample']
    param['colsample_bytree']=space['colsample_bytree']
    #param['scale_pos_weight']=space['scale_pos_weight']
    
    numround = int(space['numround'])
    
    #evallist  = [(dtest,'eval'), (dtrain,'train')]
    
    bst = xgb.train(param, dtrain, numround)
    ypred = bst.predict(dtest)

    auc = roc_auc_score(y2,ypred)
    
    print("SCORE: %s" %auc)

    return{'loss':1-auc, 'status': STATUS_OK }


space ={
        'max_depth': hp.qnormal("x_max_depth", 9, 1.2, 1),
        'min_child_weight': 0.5,
        'eta': 0.1,
        'lambda': hp.qnormal ('x_lambda', 6, 2,1),
        'alpha': hp.normal ('x_alpha', 6, 2,1),
        'subsample':1,
        'colsample_bytree':1,
        #'scale_pos_weight': hp.normal('x_scale_pos_weight',0.5,0.5),
        'numround':hp.quniform('x_numround',400,600,1)
    }



# In[ ]:

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials
           )


# In[ ]:

print(best)


# In[ ]:

#Idées
#Treat missing values (là j'impute la moyenne et ça trompe l'algo) => ça n'a pas amélioré le score voire ça l'a même diminué
#Passer un array de weights (pas facile à faire mais à voir car il faut récupérer toutes les dimensions des données 
#c'est faisable mais grave chiant)
#Feature scaling is unecessary
get_ipython().magic('pinfo xgb.DMatrix')


# In[ ]:

get_ipython().magic('pinfo xgb.train')

