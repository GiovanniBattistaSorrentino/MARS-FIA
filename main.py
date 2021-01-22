from GraphDrawer import *
import random

#ritorna un array contenente l'id dei film che sono stati trovati per nome
def ricercaFilmPerNome(nome):
    r = []
    file_titoli = open("titoli.txt", "r", encoding='UTF8')

    titoli = file_titoli.readlines()
    for line_titoli in titoli:
        arr = line_titoli.split('\t')
        if (nome in arr[1]):
            r.append(arr[0])
    file_titoli.close()
    return r

#ritorna un cluster che viene selezionato randomicamente in base ad una probabilità che viene calcolata in base
#ai film piaciuti all'utente, lista_decisioni = lista contenente i cluster dei film che sono piaciuti all'utente
def getClusteInBaseAlleDecisioni(lista_decisioni):
    while True:
        for x in range(getNumOfClusters(list)):
            y = float(getRateClusterInDecisioni(lista_decisioni, x) * 100)
            z = random.randint(1, 100)
            if(y >= z):
                return x

#ritorna la percentuale di film piaciuti all'utente del cluster dato in input
def getRateClusterInDecisioni(lista_decisioni, cluster):
    c = 0
    for line in lista_decisioni:
        if(int(line) == int(cluster)):
            c = c + 1
    return float(c / len(lista_decisioni))

#ritorna il cluster del film dato in input tramite l'id
def getClusterOfFilm(clusters, id):
    for line in clusters:
        if(line['FilmId'] == id):
            return line['ClusterNumber']

#ritorna il titolo del film dato in input tramite l'id
def getTitoloFilm(id):
    file_titoli = open("titoli.txt", "r", encoding='UTF8')

    titoli = file_titoli.readlines()
    for line_titoli in titoli:
        arr = line_titoli.split('\t')
        if(arr[0] == id):
            file_titoli.close()
            return arr[1].replace('\n', '')
    file_titoli.close()

#ritorna la stringa contenente i generi del film dato in input tramite l'id
def getGeneriFilm(id):
    file_final = open("final.txt", "r", encoding='UTF8')

    lines = file_final.readlines()
    for line_titoli in lines:
        arr = line_titoli.split('\t')
        if(arr[0] == id):
            file_final.close()
            return arr[3].replace('\n', '')
    file_final.close()

#ritorna l'anno del film dato in input tramite l'id
def getAnnoFilm(id):
    file_final = open("final.txt", "r", encoding='UTF8')

    lines = file_final.readlines()
    for line_titoli in lines:
        arr = line_titoli.split('\t')
        if(arr[0] == id):
            file_final.close()
            return arr[1]
    file_final.close()

#ritorna il voto imdb del film dato in input tramite l'id
def getRankFilm(id):
    file_final = open("final.txt", "r", encoding='UTF8')

    lines = file_final.readlines()
    for line_titoli in lines:
        arr = line_titoli.split('\t')
        if(arr[0] == id):
            file_final.close()
            return arr[2]
    file_final.close()

#################################################################################################
#inizio main

list = createClusters()
print("\nID E CLUSTER DEI FILM OTTENUTI DAL K-MEANS")
print(list)

#Costanti
numero_di_film_per_domande = 5
#le domande partono da 0. Quindi praticamente le domande sono 5
numero_di_domande = 4

domande = []
decisioni = []
for x in range(numero_di_domande):
    print("seleziona i film che ti piacciono")
    for y in range(numero_di_film_per_domande):
        clu = getRandCluster(list)
        film = getRandFilm(list, clu)
        domande.append({'filmid': film, 'cluster': clu, 'anno': getAnnoFilm(film), 'rank': getRankFilm(film), 'generi': getGeneriFilm(film), 'nome': getTitoloFilm(film)})
        print(str(y) + " per scegliere il film " + domande[y]['nome'] + " anno: " + domande[y]['anno'] + " voto: " + domande[y]['rank'] + " genere/generi: " + domande[y]['generi'])

    inp = input()
    if(int(inp) < 0) or (int(inp) >= numero_di_film_per_domande):
        print("input non valido")
    else:
        decisioni.append(getClusterOfFilm(list, domande[int(inp)]['filmid']))
    domande.clear()

numero_di_film_consigliati = 5
consigliati = []

inp = 9
while int(inp) != 0:
    print("digitare 0 per terminare, 1 per guardare film, 2 per vedere suggerimenti")
    inp = input()
    if(int(inp) == 1):
        print("digitare il nome del film che si vuole guardare")
        nome_film_cercato = input()
        ricercati = ricercaFilmPerNome(nome_film_cercato)
        if(len(ricercati) > 0):
            print("digitare")
            num = 0
            for y in ricercati:
                print(str(num) + " per vedere il film " + getTitoloFilm(y) + " anno: " + getAnnoFilm(y) + " voto: " + getRankFilm(y) + " genere/generi: " + getGeneriFilm(y))
                num = num + 1
            film_selezionato = input()
            print("ti è piaciuto? [y/n]")
            piaciuto = input()
            if(str(piaciuto) == "y"):
                decisioni.append(getClusterOfFilm(list, ricercati[int(film_selezionato)]))
        else:
            print("nessun film presente con il nome dato")
    elif(int(inp) == 2):
        for i in range(numero_di_film_consigliati):
            clu = getClusteInBaseAlleDecisioni(decisioni)
            filmid = getRandFilm(list, clu)
            consigliati.append(filmid)
        print("ecco i film consigliati:")
        for i in consigliati:
            print(str(getTitoloFilm(i)) + " " + str(getAnnoFilm(i)) + " " + str(getRankFilm(i) + " " + str(getGeneriFilm(i))))
        print('\n')
        consigliati.clear()










