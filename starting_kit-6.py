
# coding: utf-8

# # Data Challenge : Prédiction de la valence d'un commentaire 
# # SD 207 
# ### Ducarouge Alexis, Bourgeon Maxime, Bourgey Florian

# # 1. Préliminaires : importation des données

# In[1]:

# on importe les librairies 
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:

train_fname = 'train.csv'
test_fname = 'test.csv'


# In[3]:

get_ipython().system('head -3 test.csv')


# In[4]:

# Pour le train 
# on crée la matrice X qui contiendra les phrases 
X = []
y = []
with open(train_fname) as f:
    for line in f:
        y.append(int(line[0]))
        X.append(line[5:-6])
y = np.array(y)

print('n_samples : %d' % len(X))


# In[5]:

print("Le taux de prédiction au niveau de la chance est : %s" % np.mean(y == 0))


# In[6]:

# Pour le test
# on crée la matrice X qui contiendra les phrases 
X_test = []
with open(test_fname) as f:
    for line in f:
        X_test.append(line[3:-6])

print('n_samples_test : %d' % len(X_test))


# In[7]:

X_test[:2]


# In[8]:

# on génère un dictionnaires d'insultes
li=[]
fs = open('insults.txt','r')
while 1:
    txt = fs.readline()
    txt=txt.lower()
    if txt =='':
        break
    else:
        li.append(txt[:-1])
fs.close()


# In[9]:

print (li) # dictionnaire en question 


# # 2. Prétraitement des données

# ## 2.1 Suppression des caractères spéciaux

# Prétraitement sur X 
# On modifie en remplaçant plusieurs caractères dans X, on s'intéresse en particulier aux caractères spéciaux. On les supprime dans leur grande majorité pour que nos méthodes ultérieures s'appliquent avec plus d'efficacité, c'est d'ailleurs ce que nos tests révèlent : en ne supprimant pas les caractères spéciaux, le score baisse en général.

# In[10]:

char_spec = ['\\xa0', '\\n', '\\', '\'', '..', ' /', '(', ')', '"', '-', '~', '!', '.', '#', '@', '^', '%', '/'
            , '?',';','x99', 'xc2', 'x80', 'x9d', 'xe2', 'xb7', 'u2019']
features_spec = np.empty((len(X), len(char_spec)))
for i in range(len(X)):
    for j in range(len(char_spec)):
        features_spec[i][j] = X[i].count(char_spec[j])


# In[40]:

features_spec_test = np.empty((len(X_test), len(char_spec)))
for i in range(len(X_test)):
    for j in range(len(char_spec)):
        features_spec_test[i][j] = X_test[i].count(char_spec[j])


# In[12]:

features_spec.shape


# In[13]:

features_spec_test.shape


# In[14]:

X_r = []
X_r = X
for i in range(len(X)) : 
    X_r[i] = X_r[i].replace('\\xa0',' ')
    X_r[i] = X_r[i].replace('\\n',' ')
    X_r[i] = X_r[i].replace('\\',' ')
    X_r[i] = X_r[i].replace('\'',' ')
    X_r[i] = X_r[i].replace('..',' ')
    X_r[i] = X_r[i].replace(' /',' ')
    X_r[i] = X_r[i].replace('(',' ')
    X_r[i] = X_r[i].replace(')',' ')
    X_r[i] = X_r[i].replace('"',' ')
    X_r[i] = X_r[i].replace('-',' ')
    X_r[i] = X_r[i].replace('~',' ')
    X_r[i] = X_r[i].replace('!',' ')
    X_r[i] = X_r[i].replace('.',' ')
    X_r[i] = X_r[i].replace(',',' ')
    X_r[i] = X_r[i].replace('#',' ')
    X_r[i] = X_r[i].replace('@',' ')
    X_r[i] = X_r[i].replace('^',' ')
    X_r[i] = X_r[i].replace('%',' ')
    X_r[i] = X_r[i].replace('/',' ')
    X_r[i] = X_r[i].replace('?',' ')
    X_r[i] = X_r[i].replace('  ',' ')
    X_r[i] = X_r[i].replace('  ',' ')
    X_r[i] = X_r[i].replace('  ',' ')
    X_r[i] = X_r[i].replace(';',' ')
    X_r[i] = X_r[i].lower()
    X_r[i] = X_r[i].replace('don t', 'dont ')
    X_r[i] = X_r[i].replace('you re', 'youre ')
    X_r[i] = X_r[i].replace('x99', '')
    X_r[i] = X_r[i].replace('xc2', '')
    X_r[i] = X_r[i].replace('x80', '')
    X_r[i] = X_r[i].replace('x9d', '')
    X_r[i] = X_r[i].replace('xe2', '')
    X_r[i] = X_r[i].replace('xb7', '')
    X_r[i] = X_r[i].replace('u2019', '')
    X_r[i] = X_r[i].replace('  ',' ')
    X_r[i] = X_r[i].replace('  ',' ')
    X_r[i] = X_r[i].replace('  ',' ')
    


# In[ ]:

X_r


# In[15]:

# même chose qu'avant mais pour X_test
X_rt = []
X_rt = X_test
for i in range(len(X_test)) : 
    X_rt[i] = X_rt[i].replace('\\xa0',' ')
    X_rt[i] = X_rt[i].replace('\\n',' ')
    X_rt[i] = X_rt[i].replace('\\',' ')
    X_rt[i] = X_rt[i].replace('\'',' ')
    X_rt[i] = X_rt[i].replace('..',' ')
    X_rt[i] = X_rt[i].replace(' /',' ')
    X_rt[i] = X_rt[i].replace('(',' ')
    X_rt[i] = X_rt[i].replace(')',' ')
    X_rt[i] = X_rt[i].replace('"',' ')
    X_rt[i] = X_rt[i].replace('-',' ')
    X_rt[i] = X_rt[i].replace('~',' ')
    X_rt[i] = X_rt[i].replace('!',' ')
    X_rt[i] = X_rt[i].replace('.',' ')
    X_rt[i] = X_rt[i].replace(',',' ')
    X_rt[i] = X_rt[i].replace('#',' ')
    X_rt[i] = X_rt[i].replace('@',' ')
    X_rt[i] = X_rt[i].replace('^',' ')
    X_rt[i] = X_rt[i].replace('%',' ')
    X_rt[i] = X_rt[i].replace('/',' ')
    X_rt[i] = X_rt[i].replace('?',' ')
    X_rt[i] = X_rt[i].replace('  ',' ')
    X_rt[i] = X_rt[i].replace(';',' ')
    X_rt[i] = X_rt[i].lower()
    X_rt[i] = X_rt[i].replace('don t', 'dont ')
    X_rt[i] = X_rt[i].replace('you re', 'youre ')
    X_rt[i] = X_rt[i].replace('x99', '')
    X_rt[i] = X_rt[i].replace('xc2', '')
    X_rt[i] = X_rt[i].replace('x80', '')
    X_rt[i] = X_rt[i].replace('x9d', '')
    X_rt[i] = X_rt[i].replace('xe2', '')
    X_rt[i] = X_rt[i].replace('xb7', '')
    X_rt[i] = X_rt[i].replace('u2019', '')
    X_rt[i] = X_rt[i].replace('  ',' ')
    X_rt[i] = X_rt[i].replace('  ',' ')
    X_rt[i] = X_rt[i].replace('  ',' ')


# In[ ]:

X_rt


# In[ ]:

import nltk
from nltk.stem import *
stemmer = PorterStemmer()


# In[ ]:

def stemming(X):
    X=[i.split(' ') for i in X]
    X_r_mots_clean=[]
    
    for j in X : 
        com=""
        for i in j :
            com=com+' '+stemmer.stem(i)
        X_r_mots_clean.append(com)
    
    return X_r_mots_clean
  
#X_r = clean(X_r)
#X_rt = clean(X_rt)


# In[ ]:

X_r = stemming(X_r)
X_rt = stemming(X_rt)


# In[ ]:

X_r


# ### Test naïf sur le train 

# Dans cette première partie, nous essayons un classifieur assez naïf qui labélise chaque phrase par un 1 si cette dernière comporte au moins 3 insultes répertoriées dans le dictionnaire définie plus haut. Le score obtenu sur le train, comme nous pouvions le prévoir, est assez faible. 

# In[ ]:

# prédiction sur le train 
y_pred = np.zeros(len(X_r)) # on crée une matrice vide
for i in range(len(X_r)) : 
    intersection = np.intersect1d(X_r[i].split(),li) # renvoie les insultes présentes dans chaque phrase
    if len(intersection) <= 3 : # si la phrase contient moins de 4 insultes présentes dans le fichier 'insults.txt'
        y_pred[i] = 0            # on suppose qu'elle ait pas injurieuse , y = 0 
    else : 
        y_pred[i] = 1    # sinon y = 1 


# In[ ]:

# score obtenu avec le fichier 'insults.txt' sur le train 
np.mean(y_pred == y)


# ## 2.2 Stop words

# Cette fonction à pour objectif de supprimer les 'stop words', c'est-à-dire les mots très fréquents en anglais qui auraient pu interférer avec les mots réellement porteurs de signification. Nous avons utilisé un dictionnaire importé.
# 
# Après quelques tests, nous nous sommes aperçu que cela avait plutôt tendance à diminuer les résultats. Ce nettoyage a certainement tendance à appauvrir certains structures dont le sens ambigüe ne peut se déduire qu'avec l'ensemble des mots, si insignifiant soient-ils dans d'autres contextes.
from stop_words import get_stop_words

stop_words = get_stop_words('en')
print(len(stop_words))
stop_words

def clean(X):
    X=[i.split(' ') for i in X]
    X_r_mots_clean=[]
    
    for j in X : 
        com=""
        for i in j :
            if (i not in stop_words):
                com=com+' '+i
        X_r_mots_clean.append(com)
    
    return X_r_mots_clean
  
X_r = clean(X_r)
X_rt = clean(X_rt)
# # 3. Définition de nos features : n_gram

# Nous avons décidé de baser notre approche sur les n-grams qui contituent nos features sur lesquelles se base l'apprentissage. Nous avons décidé de les utiliser car elles présentent plusieurs atouts : relativement aisés à coder, robustes aux fautes d'orthographe (pout leur version n-grams caractères) et leur grande expressivité (au niveau des structures grammaticales).  
# 
# Notre approche inclut 3 types de n-gram différents que nous avons tous utilisés dans nos tests : 
#  - n-gram caractères
#  - n-gram mots
#  - n-gram basé sur les mots du dictionnaire d'insultes
#  
# Notre implémentation utilise largement les tables des hachage pour garantir une compléxité acceptable, ces tables définissent nos dictionnaires de n-grams et les correspondances entre n-gram et colonne de la matrice d'apprentissage. 
#  
# Les paramètres principaux sont les suivant : nmin/nmax pour régler l'intervalle de la longueur de la séquence de caractères ou de mots consécutifs que l'algorithme va chercher et occmin (un pour chaque valeur possible de n-gram) pour régler le nombre minimal de commentaires dans lesquels le n-gram doit apparaître pour être retenu. L'occurence minimale permet en théorie de moins overfiter, en effet cela évite que des features soient non nulles uniquement pour un ou deux commentaires, et donc les discriminer sur cette unique base, qui plus est ne permet pas généraliser.

# ## 3.1 n-gram version dictionnaire d'insultes

# In[16]:

# fit pour le n-gram dictionnaire d'insultes
# le fit doit se faire sur une concaténation des ensembles de train et de test
def insults_fit(insults) :
    D={}
    T=insults
    for i in range(len(T)) :
        if (not (' ' in T[i])) :
            D[ " " + T[i] + " " ] = i
                        
    return D
  


# In[17]:

# transform pour le dictionnaire d'insultes
def insults_transform(T,D,nmin, nmax) :
    D_c={}
    D_cr={}
    cpt=0
    for v in D.keys():
        D_c[v]=cpt
        D_cr[cpt]=v
        cpt+=1
    gramT=np.zeros((len(T),cpt))
    for i in range(len(T)) :
        for j in range(len(T[i])) :
            for k in range(nmin,nmax+1) :
                if (j-k>=0) :
                    prov=T[i][j-k:j]
                    if (prov in D):
                        gramT[i,D_c[prov]]+=1
    return gramT, D_c, D_cr        


# ## 3.2 n-gram version caractères

# In[18]:

# fit pour le n-gram caractères
# le fit doit se faire sur une concaténation des ensembles de train et de test
def ngram_fit(T,nmin, nmax) :
    D={}
    D_der_comm={}
    for i in range(len(T)) :
        for j in range(len(T[i])) :
            for k in range(nmin,nmax+1) :
                if (j-k>=0) :
                    prov=T[i][j-k:j]
                    
                    if (not prov in D_der_comm):
                        if (not prov in D):
                            D[ prov ] = 1
                        else:
                            D[ prov ] += 1
                    
                    if (prov in D_der_comm and D_der_comm[ prov ]!=i):
                        if (not prov in D):
                            D[ prov ] = 1
                        else:
                            D[ prov ] += 1
                        
                    D_der_comm[ prov ] = i
                        
    return D


# In[19]:

# fit pour le n-gram caractères
def ngram_transform(T,D,nmin, nmax, occmin_tab) :
    D_c={}
    D_cr={}
    cpt=0
    for v in D.keys():
        if (D[v]>=occmin_tab[len(v)-nmin]):
            D_c[v]=cpt
            D_cr[cpt]=v
            cpt+=1
    gramT=np.zeros((len(T),cpt))
    for i in range(len(T)) :
        for j in range(len(T[i])) :
            for k in range(nmin,nmax+1) :
                if (j-k>=0) :
                    prov=T[i][j-k:j]
                    if ((prov in D) and D[prov]>=occmin_tab[len(prov)-nmin]):
                        gramT[i,D_c[prov]]+=1
    return gramT, D_c, D_cr


# ## 3.3 n-gram version mots

# In[20]:

# fit pour le n-gram mots
# le fit doit se faire sur une concaténation des ensembles de train et de test
def ngram_word_fit(T,nmin, nmax) :
    T=[i.split(' ') for i in T]
    D={}
    D_der_comm={}
    for i in range(len(T)) :
        for j in range(len(T[i])) :
            for k in range(nmin,nmax+1) :
                if (j-k>=0) :
                    prov=" ".join(T[i][j-k:j])
                    
                    if (not prov in D_der_comm):
                        if (not prov in D):
                            D[ prov ] = 1
                        else:
                            D[ prov ] += 1
                    
                    if (prov in D_der_comm and D_der_comm[ prov ]!=i):
                        if (not prov in D):
                            D[ prov ] = 1
                        else:
                            D[ prov ] += 1
                        
                    D_der_comm[ prov ] = i
                        
    return D


# In[21]:

# fit pour le n-gram mots
def ngram_word_transform(T,D,nmin, nmax, occmin_tab) :
    D_c={}
    D_cr={}
    T=[i.split(' ') for i in T]
    cpt=0
    for v in D.keys():
        if (D[v]>=occmin_tab[len(v.split(" "))-nmin]):
            D_c[v]=cpt
            D_cr[cpt]=v
            cpt+=1
    gramT=np.zeros((len(T),cpt))
    for i in range(len(T)) :
        for j in range(len(T[i])) :
            for k in range(nmin,nmax+1) :
                if (j-k>=0) :
                    prov=" ".join(T[i][j-k:j])
                    if ((prov in D) and D[prov]>=occmin_tab[len(prov.split(" "))-nmin]):
                        gramT[i,D_c[prov]]+=1
    return gramT, D_c, D_cr
             


# ## 3.4 Création des matrices n-gram d'apprentissage et de test

# On applique le n_gram (3 à 6) caractères sur la matrice X_r et X_rt transformées, avec une occurence minimale ici uniforme de 50 commentaires.

# In[22]:

Dc = ngram_fit(np.hstack((X_r, X_rt)),3, 6)
gramTc, D_cc, D_crc = ngram_transform(X_r, Dc, 3, 6, [50, 50, 50, 50, 50] )
gramT_testc, D_c_testc, D_cr_testc = ngram_transform(X_rt, Dc, 3, 6, [50, 50, 50, 50, 50] )


# In[23]:

# nombre de features :
gramTc.shape[1]


# On applique le n_gram mots (1, sac de mots) sur la matrice X_r et X_rt transformées, avec une occurence minimale d'ici 5 commentaires.

# In[24]:

Dw = ngram_word_fit(np.hstack((X_r, X_rt)), 1, 1)
gramTw, D_cw, D_crw = ngram_word_transform(X_r, Dw, 1, 1, [5] )
gramT_testw, D_c_testw, D_cr_testw = ngram_word_transform(X_rt, Dw, 1, 1, [5] )


# In[25]:

# nombre de features :
gramTw.shape[1]


# In[26]:

# Concaténation des diférents types de features
gramT=np.c_[gramTc,gramTw]


# In[27]:

gramT_test=np.c_[gramT_testc,gramT_testw]


# In[28]:

gramT = np.c_[gramT, features_spec]


# In[29]:

gramT_test = np.c_[gramT_test, features_spec_test]


# In[30]:

gramT.shape


# In[31]:

gramT_test.shape


# # 4. Divers fonctions outils

# ## 4.1 Predict

# In[32]:

#Fonction predict qui permet de prédire le score
def predict(X, fitted_values):
    y_pred_test = pred_values(fitted_values, X) 
    
    # dans la fonction, on renvoie 1-score, on change pour obtenir le score directement
    u = 1- np.array(y_pred_test)
    
    return u  


# ## 4.2 Bagging

# Nous avons décidé d'implémenter une fonction de baging.
# Après quelques essais infructueux sur des sets de prédicteurs assez limités, nous l'avons légèrement détourner de son emploi normal dans le sens ou nous nous en sommes surtout servis par la suite pour avoir une bonne idée de notre score sans avoir à trouver le nombre d'itérations optimal.
# 
# Nous nous sommes en effet rendu compte qu'en sauvegardant les prédicteurs à toutes les itérations et en les baggant sur de larges plages (1000 à 3000 itérations par exemple), nous obtenions un score de prédiction souvent égal, ou très peu inférieur à celui que nous trouvions en parcourant l'ensemble des prédicteurs (selon leur nombre d'itérations). Cela nous permettait d'avoir assez rapidement un idée de la performance de notre set d'hyperparamètres sans avoir à chercher le point de début de l'overfitting (en terme de nombre d'itérations).

# In[33]:

#Fonction pour le bagging 
def bagging(tab_theta,X,itmin,itmax):
    res = np.empty((itmax-itmin, len(X)))
    for i in range(itmin, itmax):
        res[i-itmin] = predict(X, tab_theta[i])
    
    res = np.mean(res, axis = 0)  
    
    return (res > 0.5).astype(np.int)        


# # 5. Classifieur : Régression Logistique

# Après avoir transformé la matrice n_gram, nous procédons à une régression logistique avec régularisation L2. Nous avons choisi la régression logistique car le problème d'optimisation est convexe et est relativement simple à mettre en place. La convergence est de plus assez rapide. 
# 
# Ce que nous avons essayé avec la régression logistique : 
# 
# -Pour améliorer la rapidité de la descente de gradient, nous avons codé la méthode de Newton avec la matrice Hessienne. Toutefois, cette méthode nous est apparu 'trop puissante et rapide' et nous convergions trop vite (ie overfit). 
# 
# ''' m = hessienne(theta_values, X, y)
#     M = np.linalg.solve(m, grad)
#     theta_values = theta_values - 0.2*M
#     grad = log_gradient(theta_values, X, y) ''' 
#     
#    
# -Nous avons également codé une régularisation L1 couplée avec une régularisation L2 (cf fonction proximal_vectorised) mais les résultats n'étaient pas concluants.
# 
# -Les résultats semblaient décroître lorsque nous utilisions un gamma trop élévé (C=1) 
# 

# In[34]:

# La matrice X (train) est ici gramT (voir code au-dessu)
# La matrice X_test (test) est ici gramT_test

X = np.copy(gramT)
X_test = np.copy(gramT_test)

# Rajouter une colonne de 1 à X, ça sera utile pour la régression logistique ensuite
X = np.vstack((np.ones(X.shape[0]), X.T))
X_test = np.vstack((np.ones(X_test.shape[0]), X_test.T))

# On prend la transposée
X = X.T
X_test = X_test.T

# On fixe maintenant n et p pour l'utilisation du classifieur
n,p = X.shape


# In[35]:

# Fonction utile pour la régularisation L1 
def proximal_vectorized(x,gamma,rho):
    x_abs = np.abs(x)
    x = np.sign(x) * (x_abs - gamma * rho) * (x_abs > gamma * rho)
    return x


# In[36]:

# Fonction prédicte pour la régression logistique, si la sigmoide donne un résultat > 0.5 alors pred_value = 1 
# sinon 0
def pred_values(theta, X):
    
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    
    return pred_value


# In[37]:

# Fonction hessienne qui permet la descente de gradient avec la méthode de Newton
def hessienne_sp(theta, X, y) :
    
    nu = logistic_func(theta, X)
    m = (np.ones(n) -  nu)*nu
    M = scipy.sparse.diags(np.ones(m),0, format="csr") 
    hess = ((X.scipy.sparse.csr_matrix.transpose()).scipy.sparse.csr_matrix.dot(M)).scipy.sparse.csr_matrix.dot(X) + 1./n*scipy.sparse.diags(np.ones(p),0, format="csr")
    
    return hess


# In[38]:

# Utilisation du classifieur : régression logistique
import math
import scipy

# Fonction sigmoïde vectorielle
def logistic_func(theta, x):
    
    return float(1) / (1 + math.e**(-x.dot(theta)))

# Gradient de la fonction objectif avec la régularisation L2 
def log_gradient(theta, x, y):
    
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    final_calc = first_calc.T.dot(x) - (20./n)*theta
    return final_calc

# Fonction objectif avec la régularisation L2 
def cost_func(theta, x, y, rho):
    
    log_func_v = logistic_func(theta,x)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2 + (rho/2)*np.linalg.norm(theta)**2
    return np.mean(final)

# Descente de gradient avec la régression logistique avec régularisation L2 
def grad_desc(theta_values, X, y, nbitermax, rho =5, gamma=1e-5):
    
    grad = log_gradient(theta_values, X, y) # calcul du gradient 
    nbiter=0 # on initialise le nombre d'itération à 0 
    
    # tableaux qui contiendront les valeurs de theta au fur et à mesure des itérations et le score
    tableau_score = []
    tableau_theta = []
    
    while nbiter < nbitermax:
        
        nbiter +=1
        
        theta_values = theta_values - gamma * log_gradient(theta_values, X, y) # descente de gradient avec gamma = 1e-5
        grad = log_gradient(theta_values, X, y) # calcul du gradient avec le nouveau theta_values
        predicted_y = pred_values(theta_values, X) 
            
        tableau_score.append(1-np.mean(y == predicted_y))
        tableau_theta.append(theta_values)
            
            
        if (nbiter%200==0) : 
            print('%s:  %s'%(nbiter,   np.linalg.norm(grad))) # on affiche la norme du gradient toutes les 200 itérations
            
    return theta_values, tableau_score, tableau_theta


# In[39]:

# On lance la descente de gradient avec pas constant (gamma = 1e-5) : 

shape = X.shape[1]
y_flip = np.logical_not(y) 
betas = np.zeros(shape)
fitted_values, tableau_score, tableau_theta = grad_desc(betas, X, y_flip, nbitermax = 3000) # on lance 3000 itérations
print(fitted_values)


# In[41]:

plt.figure(figsize=(20, 10))
plt.plot(range(3000),1-np.array(tableau_score))
plt.title('Score sur le train en fonction du nombre d iterations')


# On voit que le score augmente au fur et à mesure que le nombre d'itérations augmente. Nous avons utilisé cette courbe
# pour repérer le maximum du score obtenu avant overfitter (d'après notre expérience). Ce score optimale se situait environ vers 0.92 voire 0.93.

# In[42]:

# On commence par faire un bagging entre 1800 et 2500 itérations 
res = bagging(tableau_theta, X, 2000, 2600)
np.mean(res == y) # score optimiste 


# In[43]:

# Bagging sur le test
res_test = bagging(tableau_theta, X_test, 2000 , 2600)


# In[44]:

np.savetxt('y_pred_bagging.txt', res_test, fmt='%s')


# In[ ]:

# D'après nos tests, nous avons remarqué que le maximum du score avant d'overfitter aux environs de 2300 et 2500 

fitted_values = np.array(tableau_theta)[2400]
y_pred_test = predict(X_test, fitted_values)


# In[ ]:

np.savetxt('y_pred.txt', y_pred_test, fmt='%s')


# In[ ]:




# In[ ]:


# On mélange les données X 
ind = range(n)
np.random.shuffle(ind)
X_shu=X[ind]
y_shu=y[ind]
    
# On découpe jusqu'à 3600 le train et le test
X_train = X_shu[:3600]
X_test_global = X_shu[3600:]
y_train = y_shu[:3600]
y_test_global = y_shu[3600:]
    
shape = X_train.shape[1]
y_flip = np.logical_not(y_train) 
betas = np.zeros(shape)
fitted_values, tableau_score, tableau_theta = grad_desc(betas, X_train, y_flip, nbitermax = 2000) # on lance 2000 itérations


# In[ ]:




# In[ ]:

# Afin de valider nos choix de paramètre et de nombre de features (via le paramètre occmin), nous avons choisi de coder
# propre fonction de cross_validation qui divise le data_set d'apprentissage en un nombre cv de parties. Nous apprenons
# sur c-1 parties et nous testons sur la dernière et ceci cv fois. 

def cross_val_score(X, y, cv, nbiter):
    
    n=X.shape[0]
    ratio=n/cv
    
    #Shuffler les données
    #Faire attention de ne shuffler que les lignes pas les colonnes
    #index_shuffled = np.random.shuffle(np.arange(n))
    #X = X[index_shuffled]
    #y = y[index_shuffled]
    
    scores=[]
    
    vecs=[]
    
    for i in range(cv):
        
        X_test = X[i*ratio:(i+1)*ratio]
        y_test = y[i*ratio:(i+1)*ratio]
        
        X_ratio = np.vstack((X[:i*ratio],X[(i+1)*ratio:]))
        y_ratio = np.hstack((y[:i*ratio],y[(i+1)*ratio:]))
        
        
        vec = grad_desc(np.zeros(X.shape[1]), X_ratio, y_ratio, nbiter)
        predicted_y = pred_values(vec, X)
        
        #_, _, vec=(X_ratio, y_ratio, 5, 0, np.zeros(p-1), a, b, beta, )
        
        vecs.append(vec)
        scores.append(np.mean(y==predicted_y))

    return vecs, scores, np.mean(vecs,axis=0), np.mean(scores)

