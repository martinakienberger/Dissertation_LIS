import pandas as pd
import numpy as np
import scipy as sp
import math
from scipy import stats
from biofes import biplot
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

## Hilfsfunktionen

# Funktion, um Index (Name) einer Bezeichnung im Glossar zu finden:
def get_name_gloss(gloss, word):
    for i, val in enumerate(gloss[0]):
        if word.lower() in str(val).lower():            
            return i, gloss.index[i]

# Funktion, um Bezeichnung von Index im Glossar zu finden:        
def get_name_glossind(gloss,word):
    for i, val in enumerate(gloss.index):
        if word.lower() in str(val).lower():
            return i, gloss.loc[val][0]
        
# Funktion zur Hinzufügung neuer Spalten unter bestimmter Kondition:
def add_columns(data,lst,col_neu):
    for i in data.index:
        for el in data.loc[i,lst]:
            if el == 1:
                data.loc[i,col_neu] = 1
                pass
        

## Grafiken

# Funktion für Darstellung mehrerer Histogramme:
    
def graf_hist(D,l,sx,sy,nrow,ncol):
    fig = plt.figure(figsize=(sx,sy))
    for i, col in enumerate(l):
        plt.subplot(nrow,ncol,i+1)
        plt.hist(D[D[col]!=0][col],color = '#0063A6')
        plt.xticks([1,2,3,4,5])
        plt.title(col,size=13)
    plt.show()


# Funktion für Darstellung der Likert-Skalen:    
    
def graf_likert(D,sy):

    likert_colors = ['white', 'firebrick','lightcoral','gainsboro','cornflowerblue', 'darkblue']

    df = D.apply(pd.value_counts).T
    if 0 in df.columns:
        df = df.drop(0, axis=1).astype(int)
    
    middles = df[[1, 2]].sum(axis=1)+df[3]*.5
    longest = middles.max()
    complete_longest = df.sum(axis=1).max()
    df.insert(0, '', (middles - longest).abs())

    df.plot.barh(stacked=True, figsize=(8,sy), color=likert_colors, edgecolor='none', legend=False)
    z = plt.axvline(longest, linestyle='--', color='black', alpha=.5)
    z.set_zorder(-1)

    plt.show()
    
    
# Funktion für Darstellung der Likert-Skalen (Version für Pilotstudie):  
    
def graf_likert1(D,sy):

    likert_colors = ['white', 'firebrick','lightcoral','gainsboro','cornflowerblue', 'darkblue']

    df = D.apply(pd.value_counts).T
    if 0 in df.columns:
        df = df.drop(0, axis=1)
    
    middles = df[[1, 2]].sum(axis=1)+df[3]*.5
    longest = middles.max()
    complete_longest = df.sum(axis=1).max()
    df.insert(0, '', (middles - longest).abs())

    df.plot.barh(stacked=True, figsize=(8,sy), color=likert_colors, edgecolor='none', legend=False)
    z = plt.axvline(longest, linestyle='--', color='black', alpha=.5)
    z.set_zorder(-1)

    plt.show()

    
# Differenzen in Barcharts für Werte ohne 0:     
    
def graf_diff(data,item,var):
    
    df = pd.crosstab(data[data[var]!=0][item],data[data[var]!=0][var])
    return df.plot(kind='bar')


# Differenzen in Barcharts für Werte mit 0:

def graf_diff2(data,item,var):
    
    df = pd.crosstab(data[item],data[var])
    return df.plot(kind='bar')


# Differenzen in Barcharts für Werte ohne 0, normalisierte Darstellung:

def graf_diff_norm(data,item,var):
    
    df = pd.crosstab(data[data[var]!=0][item],data[data[var]!=0][var],normalize='index')

    for col in df.columns:
        print('n('+ str(col) + ')= ' + str(len(data[data[var] == col])))
    return df.plot(kind='bar',stacked=True)


# Differenzen in Likert-Graphiken für Werte ohne 0:

def graf_diff_likert(data,item,var):
    
    data = data[(data[var]!=0) & (data[item]!=0)]
    likert_colors = ['firebrick','lightcoral','gainsboro','cornflowerblue', 'darkblue']
    
    df = pd.crosstab(data[item],data[var])
    for col in df.columns:
        df[col] = df[col]*100/df[col].sum()
        print('n('+ str(col) + ')= ' + str(len(data[data[var] == col])))
    
    return df.T.plot(kind='barh', stacked='True',color=likert_colors,legend=None)


# Differenzen in Likert-Graphiken für Werte (in Gruppen) ohne 0:

def graf_diff_likert2(data,item,var):
    
    data = data[data[item]!=0]
    likert_colors = ['firebrick','lightcoral','gainsboro','cornflowerblue', 'darkblue']
    
    df = pd.crosstab(data[item],data[var])
    for col in df.columns:
        df[col] = df[col]*100/df[col].sum()
        print('n('+ str(col) + ')= ' + str(len(data[data[var] == col])))
    
    return df.T.plot(kind='barh', stacked='True',color=likert_colors,legend=None)


# Marker definieren:

def vector_to_shape(v):
    markers = ["p", "," , "o" , "v" , "x", "^", "D", "*", "1","<",">","d"]
    return [markers[el] for el in v]


# Farben definieren:

def vector_to_color(v):
    col = ['orange','b', 'g', 'r', 'm', 'c', 'y', 'k', 'saddlebrown','pink','grey','indigo']
    return [col[el] for el in v]


## Statistik        

# Deskriptive Statistik:

def df_describe(data,lst):
    df = pd.DataFrame(data[lst]).describe()
    df = df.T
    mode = pd.DataFrame(data[lst]).mode().T
    median = pd.DataFrame(data[lst]).median().T
    kurtosis = pd.DataFrame(data[lst]).kurtosis().T
    skew = pd.DataFrame(data[lst]).skew().T
    df['mode'] = mode
    df['median'] = median
    df['kurtosis'] = kurtosis
    df['skew'] = skew
    return df


def df_describe1(data,lst):
    df = pd.DataFrame(data[lst]).describe()
    df = df.T
    median = pd.DataFrame(data[lst]).median().T
    kurtosis = pd.DataFrame(data[lst]).kurtosis().T
    skew = pd.DataFrame(data[lst]).skew().T
    df['median'] = median
    df['kurtosis'] = kurtosis
    df['skew'] = skew
    return df


# Übersicht über prozentuelles Antwortverhalten:

def table_porc(D):
    df = D.apply(pd.value_counts).fillna(0).astype(int).T
    total_row = df.sum(axis=1)
    for i in df.index:
        df.loc[i] = df.loc[i]*100/total_row.loc[i]
    df['n'] = total_row
    return df


# Cronbach's Alpha:

def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=0, ddof=1)
    tscores = itemscores.sum(axis=1)
    nitems = itemscores.shape[1]
    calpha = nitems / float(nitems-1) * (1 - itemvars.sum() / float(tscores.var(ddof=1)))

    return calpha


# Spearman's Rho:

def df_spearman(data,lst):
    
    df = data[lst].corr(method = 'spearman')
    df = df.style.applymap(lambda x: 'background-color : salmon' if abs(x) > 0.5 
                           else 'background-color: lightsalmon' if abs(x) > 0.3 and abs(x) <= 0.5
                           else 'background-color: peachpuff' if abs(x) > 0.1 and abs(x) <= 0.3
                           else 'background-color : white')
    return df


def df_spearman_pv(data,lst):
    
    corr, pv = stats.spearmanr(data[lst])
    df = pd.DataFrame(pv)
    df.columns = data[lst].corr(method = 'spearman').columns
    df.index = data[lst].corr(method = 'spearman').index
    df = df.style.applymap(lambda x: 'background-color : salmon' if abs(x) < 0.01 
                           else 'background-color: lightsalmon' if abs(x) < 0.05 and abs(x) >= 0.01
                           else 'background-color : white')
    return df


# Mann-Whitney-U-Test:

def df_mannwhitneyu(data,var,lst,v1,v2):
    
    mannwhitneyu_test = []
    for col in lst:    
        mannwhitneyu_test.append(stats.mannwhitneyu(data[col][data[var]==v1],
                      data[col][data[var]==v2]).pvalue)
    
    df = pd.DataFrame(mannwhitneyu_test)
    df.index = [lst[ind] for ind in df.index]
    df_result = df[df[0] < 0.05]
    df_result.rename(columns={0:str(var)}, inplace=True)
    df_result = df_result.style.applymap(lambda x: 'background-color : salmon' if x < 0.01 else 'background-color: lightsalmon' if x >= 0.01 and x < 0.05 else 'background-color : white')

    return df_result


# Elbow Method zur Bestimmung der optimalen Cluster-Anzahl:

def elbow_method(data):
    Sum_of_squared_distances = []
    K = range(1,25)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()  

    
# Silhouette Coefficient zur Berechnung der optimalen Cluster-Anzahl:
    
def silhouette(data):

    Silhouette = []
    K = range(2,25)
    for k in K:
        model = AgglomerativeClustering(n_clusters=k).fit(data)
        labels = model.labels_
        sil_score = metrics.silhouette_score(data, labels, metric='euclidean')
        Silhouette.append(sil_score)

    plt.plot(K, Silhouette, 'bx-'); 
    

# Biplot:

def graf_biplot(data, dim, vcol, dim1 = 0, dim2 = 1, sx = 10, sy = 10):
    
    B = biplot.Classic(data, dim, method = 1) 
    R = B.RowCoord
    C = B.ColCoord
    
    fig = plt.figure(figsize=(sx,sy))
    ax = fig.add_subplot(111)
    
    color_v = vector_to_color(vcol)
    sh_v = vector_to_shape(vcol)
    
    for i in range(C.shape[0]):
        ax.arrow(0,0,C[i,dim1],C[i,dim2], alpha = 0.7)
        #ax.scatter(C[i,dim1],C[i,dim2]) # Hier würden Linien nicht angezeigt.
        ax.text(C[i,dim1],C[i,dim2],data.columns[i],fontsize=15, alpha = 0.7)

    for i in range(R.shape[0]):
        ax.scatter(R[i,dim1],R[i,dim2], s = 20, c = color_v[i], marker = sh_v[i])
    
    ax.legend(pd.Series(vcol.unique()).sort_values())
    

# Biplot mit TN_ID:

def graf_biplot_text(data, dim, vcol, dim1 = 0, dim2 = 1, sx = 10, sy = 10):
    
    B = biplot.Classic(data, dim, method = 1) 
    R = B.RowCoord
    C = B.ColCoord
    
    fig = plt.figure(figsize=(sx,sy))
    ax = fig.add_subplot(111)
    
    color_v = vector_to_color(vcol)
    sh_v = vector_to_shape(vcol)
    
    for i in range(C.shape[0]):
        ax.arrow(0,0,C[i,dim1],C[i,dim2], alpha = 0.7)
        #ax.scatter(C[i,dim1],C[i,dim2]) # Hier würden Linien nicht angezeigt.
        ax.text(C[i,dim1],C[i,dim2],data.columns[i],fontsize=15, alpha = 0.7)

    for i in range(R.shape[0]):
        ax.scatter(R[i,dim1],R[i,dim2], s = 20, c = color_v[i], marker = sh_v[i])
    
    ax.legend(pd.Series(vcol.unique()).sort_values())
    texts = [plt.text(R[i,dim1],R[i,dim2], data.index[i], ha='center', va='center') for i in range(len(R[:,0]))];

    
# Cluster-Analyse:    
    
def graf_cplot(data, dim, nclust, dim1 = 0, dim2 = 1, sx = 10, sy = 10):
    
    B = biplot.Classic(data, dim, method = 1) 
    R = B.RowCoord
    C = B.ColCoord
    
    ward = AgglomerativeClustering(n_clusters=nclust).fit(data)
    
    fig = plt.figure(figsize=(sx,sy))
    ax = fig.add_subplot(111)

    sh_v = vector_to_shape(ward.labels_)
    color_v = vector_to_color(ward.labels_)

    for i in range(C.shape[0]):
        ax.arrow(0,0,C[i,dim1],C[i,dim2], alpha = 0.7, color = 'black')
        #ax.scatter(C[i,dim1],C[i,dim2]) # Hier würden Linien nicht angezeigt.
        ax.text(C[i,dim1],C[i,dim2],data.columns[i],fontsize=15, alpha = 0.7)

    for i in range(R.shape[0]):
        ax.scatter(R[i,dim1],R[i,dim2], s = 20, c = color_v[i] , marker = sh_v[i]);

        
def graf_cplot2(data, dim, nclust, dim1 = 0, dim2 = 1, sx = 10, sy = 10):
    
    B = biplot.Classic(data, dim, method = 1) 
    R = B.RowCoord
    C = B.ColCoord
    
    K = KMeans(n_clusters=nclust,random_state=12).fit(data)
    
    fig = plt.figure(figsize=(sx,sy))
    ax = fig.add_subplot(111)

    sh_v = vector_to_shape(K.labels_)
    color_v = vector_to_color(K.labels_)

    for i in range(C.shape[0]):
        ax.arrow(0,0,C[i,dim1],C[i,dim2], alpha = 0.7, color = 'black')
        #ax.scatter(C[i,dim1],C[i,dim2]) # Hier würden Linien nicht angezeigt.
        ax.text(C[i,dim1],C[i,dim2],data.columns[i],fontsize=15, alpha = 0.7)

    for i in range(R.shape[0]):
        ax.scatter(R[i,dim1],R[i,dim2], s = 20, c = color_v[i] , marker = sh_v[i]);
       
    
# Cluster-Analyse mit TN_ID:         

def graf_cplot_text(data, dim, nclust, dim1 = 0, dim2 = 1, sx = 10, sy = 10):
    
    B = biplot.Classic(data, dim, method = 1) 
    R = B.RowCoord
    C = B.ColCoord
    
    ward = AgglomerativeClustering(n_clusters=nclust).fit(data)
    
    fig = plt.figure(figsize=(sx,sy))
    ax = fig.add_subplot(111)

    sh_v = vector_to_shape(ward.labels_)
    color_v = vector_to_color(ward.labels_)

    for i in range(C.shape[0]):
        ax.arrow(0,0,C[i,dim1],C[i,dim2], alpha = 0.7, color = 'black')
        #ax.scatter(C[i,dim1],C[i,dim2]) # Hier würden Linien nicht angezeigt.
        ax.text(C[i,dim1],C[i,dim2],data.columns[i],fontsize=15, alpha = 0.7)

    for i in range(R.shape[0]):
        ax.scatter(R[i,dim1],R[i,dim2], s = 20, c = color_v[i] , marker = sh_v[i])
    
    texts = [plt.text(R[i,dim1],R[i,dim2], data.index[i], ha='center', va='center') for i in range(len(R[:,0]))];
    
    
def graf_cplot2_text(data, dim, nclust, dim1 = 0, dim2 = 1, sx = 10, sy = 10):
    
    B = biplot.Classic(data, dim, method = 1) 
    R = B.RowCoord
    C = B.ColCoord
    
    K = KMeans(n_clusters=nclust,random_state=12).fit(data)
    
    fig = plt.figure(figsize=(sx,sy))
    ax = fig.add_subplot(111)

    sh_v = vector_to_shape(K.labels_)
    color_v = vector_to_color(K.labels_)

    for i in range(C.shape[0]):
        ax.arrow(0,0,C[i,dim1],C[i,dim2], alpha = 0.7, color = 'black')
        #ax.scatter(C[i,dim1],C[i,dim2]) # Hier würden Linien nicht angezeigt.
        ax.text(C[i,dim1],C[i,dim2],data.columns[i],fontsize=15, alpha = 0.7)

    for i in range(R.shape[0]):
        ax.scatter(R[i,dim1],R[i,dim2], s = 20, c = color_v[i] , marker = sh_v[i])
    
    texts = [plt.text(R[i,dim1],R[i,dim2], data.index[i], ha='center', va='center') for i in range(len(R[:,0]))];
    

# Cluster-Analyse mit 2 DF und TN-ID:    
    
def graf_cplot3_text(data,data1, dim, nclust, dim1 = 0, dim2 = 1, sx = 10, sy = 10):
    
    B = biplot.Classic(data, dim, method = 1) 
    R = B.RowCoord
    C = B.ColCoord
    
    K = AgglomerativeClustering(n_clusters=nclust, linkage='ward').fit(data1)
    
    fig = plt.figure(figsize=(sx,sy))
    ax = fig.add_subplot(111)

    sh_v = vector_to_shape(K.labels_)
    color_v = vector_to_color(K.labels_)

    for i in range(C.shape[0]):
        ax.arrow(0,0,C[i,dim1],C[i,dim2], alpha = 0.7, color = 'black')
        #ax.scatter(C[i,dim1],C[i,dim2]) # Hier würden Linien nicht angezeigt.
        ax.text(C[i,dim1],C[i,dim2],data.columns[i],fontsize=15, alpha = 0.7)

    for i in range(R.shape[0]):
        ax.scatter(R[i,dim1],R[i,dim2], s = 20, c = color_v[i] , marker = sh_v[i])
    
    texts = [plt.text(R[i,dim1],R[i,dim2], data.index[i], ha='center', va='center') for i in range(len(R[:,0]))];