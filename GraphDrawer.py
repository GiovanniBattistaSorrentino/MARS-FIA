import json
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def createClusters():
    #Recupero dataset dal sistema
    movie_dataset = pd.read_csv("final.txt", delimiter = "\t")

    #Dizionario che contiene i valori arbitrari dei generi
    genre_value_dic = {
        'Drama': 6,
        'Horror': 1,
        'Sci-Fi': 3,
        'Western': 23,
        'Adventure': 24,
        'Crime': 5,
        'Film-Noir': 22,
        'Action': 7,
        'Fantasy': 21,
        'Comedy': 19,
        'War': 8,
        'Romance': 17,
        'Thriller': 4,
        'Musical': 18,
        'History': 9,
        'Mystery': 2,
        'Documentary': 10,
        'Music': 15,
        'Sport': 12,
        'Adult': 13,
        'Family': 20,
        'Biography': 11,
        'Animation': 16,
        'News': 14
    }
    #Variabile che ospita la sommatoria del valore di genere di un singolo film
    single_value = 0

    #Calcolo del peso del genere per ogni singolo film
    for row in range(len(movie_dataset)):
        #Reperiamo per la riga i-esima il genere
        data_cell = movie_dataset.loc[row][3]
        #data_cell returna una stringa e con lo split prendiamo una lista dei generi
        data_value = data_cell.split(",")
       #Se un film possiede un singolo genere allora lo moltiplichiamo 3 volte
        if len(data_value) == 1:
            single_value = genre_value_dic[data_value[0]] * 3
        elif len(data_value) == 2:
            single_value = genre_value_dic[data_value[0]]*2 + genre_value_dic[data_value[1]]
        else:
            for elem in data_value:
                single_value += genre_value_dic[elem]
        #Nel nostro dataframe inseriamo al posto del genere il valore calcolato
        movie_dataset.at[row,"genere"] = single_value
        single_value = 0

    #Stampa del dataset con il peso del genere, si noti come tutti i dati siano discreti (tranne id)
    print("########## STAMPA DATASET CON PESO DEL GENERE #######")
    print(movie_dataset)


    #Rimozione colonna id dal dataset
    X = movie_dataset.drop('id', axis = 1)


############################### Pre-processing finito, inzia la fase di clustering


############################### DBSCAN: Preparazione al PCA
    # I dati vengono scalati per una migliore distribuzione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalizzazione dei dati in modo tale che la distribuzione dei dati
    # assomigli ad una distribuzione gaussiana
    X_normalized = normalize(X_scaled)

    # Conversione da un panda dataframe ad un numpyArray
    X_normalized = pd.DataFrame(X_normalized)

    #Stampa dei dati scalati e normalizzati
    print("###### STAMPA DATI SCALATI E NORMALIZZATI PER PCA ######")
    print(X_normalized)

    ####################################  PCA

    #Dichiarazione variabile pca, essa possiede il metodo per eseguire l'algoritmo
    pca = PCA(n_components = 2)
    #Esecuzione algoritmo
    X_principal = pca.fit_transform(X_normalized)
    #Conversione a dataFrame di pandas
    X_principal = pd.DataFrame(X_principal)
   #Nome degli headers delle colonne
    X_principal.columns = ['P1', 'P2']

   #Stampa dati ridotti da PCA
    print("####### STAMPA DATI OTTENUTI DAL PCA #######")
    print(X_principal)


    ########################################### Inizio DBSCAN

    # Passaggio dei dati all'algoritmo DBSCAN, la funzione accetta un array di dati con caratteristiche numeriche
    # Parametro eps: Distanza massima tra due campioni per essere considerati nello stesso vicinato
    # Parametro min_samples: numero minimo di punti per considerare un intorno di un punto denso
    db_default = DBSCAN(eps = 0.1, min_samples = 3).fit(X_principal)

    #La funzione ritorna un array di zeri dello stesso tipo e forma dell'array dato
    #In questo caso un array di booleani  in quanto dtype = bool
    core_samples_mask = np.zeros_like(db_default.labels_, dtype=bool)
    #Con questa istruzione in una matrice numpy vengono tracciati i dati del dataset che sono
    # punti densi, i valori che non vengono settati a true significa che sono punti di rumore
    core_samples_mask[db_default.core_sample_indices_] = True

    #Recupero dei labes per tutti i dati, ogni label indica a quale cluster appartiene
    #Labels è un array
    labels = db_default.labels_


    ################### Da qui recuperiamo a quale cluster un film appartiene
    cluster_dic_list = [{

    }]

# Con questo ciclo per ogni film presente nel dataset recuperiamo il cluster a cui appartiene
# I labels sono ordinati allo stesso modo del dataset quindi il labels[0] contiene il cluster del film movie_dataset.loc[0][0]
    for c_row in range(len(movie_dataset)):
      cluster_dic_list.append({"FilmId" : movie_dataset.loc[c_row][0],"ClusterNumber" : labels[c_row]})


    # Calcolo del numero dei cluster senza considerare i punti di rumore
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #Calcolo dei punti di rumore
    n_noise_ = list(labels).count(-1)

    print("############################\nRISULTATI DBSCAN\n########################")
    print('Numero di cluster stimati: %d' % n_clusters_)
    print('Numero di cluster stimati: %d' % n_noise_)
    print("Coefficiente di forma: %0.3f" % metrics.silhouette_score(X, labels))


    ######################## INIZIO RAPPRESENTAZIONE GRAFICA DBSCAN
    # Codice reperito dalla guida ufficiale della libreria sklearn per la rappresentazione del clustering
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('DBSCAN Numero di cluster stimati: %d' % n_clusters_)
    plt.show()
    ######################## FINE RAPPRESENTAZIONE GRAFICA DBSCAN
    ######################## FINE DBSCAN

    ###############################KMEANS######################################

    #Con questa istruzione rendiamo deterministica la posizione iniziale dei 12 centroidi
    #Attraverso una fase di testing dove abbiamo provato diversi valori di seme siamo arrivati alla
    #conclusione che con il valore 7 otteniamo il migliore coefficiente di forma
    np.random.seed(7)

    #Otteniamo una variabile k-means
    km = KMeans(n_clusters=12)




    #settiamo il titolo della finestra che ospita il grafico come la sua dimensione
    fig = plt.figure("Risultato K-MEANS", figsize=(4, 3))
    #Poichè abbiamo 3 caratteristiche per ogni dato ( Peso genere, anno e punteggio) otteniamo un grafico avente
    # tre dimensioni
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #esecuzione dell'algoritmo sul dataset senza l'id ( si veda il codice sopra )
    km.fit(X)
    #Ottenimento dei labes
    labels = km.labels_

    # Dizionario che conterrà l'id dei film ed il loro cluster di appartenenza
    kmeans_dic_list = []

    # Ci permette di recuperare tutti i film nel cluster del k-means
    for c_row in range(len(movie_dataset)):
        kmeans_dic_list.append({"FilmId": movie_dataset.loc[c_row][0], "ClusterNumber": labels[c_row]})

    #Con questa istruzione poniamo i punti nello spazio tridimensionale
    ax.scatter(X.iloc[:, 2], X.iloc[:, 0], X.iloc[:, 1], c=labels.astype(np.float), edgecolor='k')

#Istruzioni necessarie per la rappresentazione, reperite dalla guida ufficiale di sk learn
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Genere')
    ax.set_ylabel('Punteggio')
    ax.set_zlabel('Anno')
    title = "K-MEANS"
    ax.set_title(title)
    ax.dist = 12

    #Risultati k-means
    print("\n\n#################### RISULTATI K-MEANS ############################")
    print("Coefficiente di forma %0.3f" % metrics.silhouette_score(X, labels))

    #istruzione che visualizza il grafico tridimensionale
    plt.show()

    return kmeans_dic_list


#ritorna il numero di film presenti in un cluster
#clusters = lista dei film e cluster
#numCluster = cluster di cui si vuole conoscere il numero di film
def getSizeCluster(clusters, numCluster):
    r = 0
    for elem in clusters:
        if(elem['ClusterNumber'] == numCluster):
            r = r + 1
    return r

#ritorna il numero di cluster presenti
#clusters = lista dei film e cluster
def getNumOfClusters(clusters):
    r = 0
    for elem in clusters:
        if (elem['ClusterNumber'] > r):
            r = elem['ClusterNumber']
    return r+1

#ritorna un film random di un cluster
#clusters = lista dei film e cluster
#cluster = cluster di cui si vuole il film random
def getRandFilm(clusters, cluster):
    r = random.randint(0, getSizeCluster(clusters, cluster))
    for line in clusters:
        if(line['ClusterNumber'] == cluster):
            if (r == 0):
                return line['FilmId']
            r = r-1
            if (r == 0):
                return line['FilmId']

#ritorna un cluster random
#clusters = lista dei film e cluster
def getRandCluster(clusters):
    r = random.randint(0, getNumOfClusters(clusters)-1)
    return r




