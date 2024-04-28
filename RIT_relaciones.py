import pandas as pd
import numpy as np
import utils as ut # esta librería tiene funciones para poder obtener un procesamiento del <T,H>
import spacy
import mutual_info as mi
import time
from scipy.stats import wasserstein_distance
import sys

import conceptnet_lite
conceptnet_lite.connect("data/conceptnet.db")
from conceptnet_lite import Label, edges_for, edges_between
from nltk.corpus import wordnet as wn
## esta función revisa hiperonimia, sinonimia entre otras.
def encontrar_relaciones(relaciones):
    borrar=set()
    borrar_i=set()
    for r in relaciones:
        index=r[0]
        c=r[1]
        wt=str(index).split("{")[1].split(",")[0]
        wh=str(c).split("{")[1].split(",")[0]    
        try:
            bandera=0
            concepts_wt = Label.get(text=wt, language='en').concepts
            concepts_wh = Label.get(text=wh, language='en').concepts
            # SINONIMOS DE SINONIMOS
            sinonimos=[]
            for e in edges_for(Label.get(text=wt, language='en').concepts, same_language=True):
                if "synonym"==e.relation.name:# and word_h== e.start.text: #and word_t== e.end.text:
                    if wt == e.start.text:
                        #print(e.start.text, "-", e.end.text, "|", e.relation.name)
                        sinonimos.append(e.end.text)
                    elif wt == e.end.text:
                        #print(e.start.text, "-", e.end.text, "|", e.relation.name)
                        sinonimos.append(e.start.text)
            sinonimos2=[]
            for e in edges_for(Label.get(text=wh, language='en').concepts, same_language=True):
                if "synonym"==e.relation.name: #and word_t== e.end.text:
                    if wh == e.start.text:
                        #print(e.start.text, "-", e.end.text, "|", e.relation.name)
                        sinonimos2.append(e.end.text)
                    elif wh == e.end.text:
                        #print(e.start.text, "-", e.end.text, "|", e.relation.name)
                        sinonimos2.append(e.start.text)
            if len(set(sinonimos).intersection(set(sinonimos2)))>0:
                print("sinonimos de sinonimos",wt,wh)
                borrar.add(c)
                borrar_i.add(index)
                break
            else:
                for e in edges_between(concepts_wt, concepts_wh, two_way=False,relation="causes"):
                    print(wt," causes ",wh)
                    borrar.add(c)
                    borrar_i.add(index)
                    bandera=1
                    break
                for e in edges_between(concepts_wt, concepts_wh, two_way=False,relation="is_a"):
                    print(wt," is_a ",wh)
                    borrar.add(c)
                    borrar_i.add(index)
                    bandera=1
                    break
                if bandera==0:
                    for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="synonym"):
                        print(wt," synonym ",wh)
                        borrar.add(c)
                        borrar_i.add(index)
                        bandera=1
                        break
                if bandera==0:
                    for e in edges_between(concepts_wh,concepts_wt, two_way=False,relation="has_a"):
                        print(wh," has_a ",wt)
                        borrar.add(c)
                        borrar_i.add(index)
                        bandera=1
                        break
                if bandera==0:
                    for e in edges_between(concepts_wh,concepts_wt, two_way=False,relation="derived_from"):
                        print(wh," derivado_from ",wt)
                        borrar.add(c)
                        borrar_i.add(index)
                        bandera=1
                        break
                if bandera==0:
                    for e in edges_between(concepts_wt,concepts_wh, two_way=False,relation="used_for"):
                        print(wt," used_for ",wh)
                        borrar.add(c)
                        bandera=1
                        borrar_i.add(index)
                        break
                if bandera==0:
                    for e in edges_between(concepts_wt, concepts_wh,two_way=False,relation="HasProperty"):
                        bandera=1
                        print(wt," HasProperty ",wh)
                        borrar.add(c)
                        borrar_i.add(index)
                        break
        except:
            a=0
    return list(borrar),list(borrar_i)

def encontrar_relaciones_contrarias(relaciones):
    antonyms=[]
    borrar=[]
    borrar_i=[]
    for r in relaciones:
        index=r[0]
        c=r[1]
        wt=str(index).split("{")[1].split(",")[0]
        wh=str(c).split("{")[1].split(",")[0]
        try:
            bandera=0
            concepts_wt = Label.get(text=wt, language='en').concepts
            concepts_wh = Label.get(text=wh, language='en').concepts
            for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="antonym"):
                print(wt," antonym ",wh)
                borrar.append(c)
                borrar_i.append(index)
                antonyms.append(wt)
                bandera=1
                break
            if bandera==0:
                for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="distinct_from"):
                    print(wh," distinct from ",wt)
                    borrar.append(c)
                    bandera=1
                    borrar_i.append(index)
                    antonyms.append(wt)
                    break
            if bandera==0:
                for e in edges_between(concepts_wh, concepts_wt, two_way=False,relation="is_a"):
                    print(wh," is_a (ant) ",wt)
                    bandera=1
                    borrar.add(c)
                    borrar_i.add(index)
                    break
            if bandera==0:
                for e in edges_between(concepts_wt, concepts_wh, two_way=False,relation="has_a"):
                    print(wt," has_a (ant)",wh)
                    bandera=1
                    borrar.add(c)
                    borrar_i.add(index)
                    break
            if bandera==0:
                for e in edges_between(concepts_wt, concepts_wh, two_way=False,relation="derived_from"):
                    bandera=1
                    print(wt," derivado_from (ant)",wh)
                    borrar.add(c)
                    borrar_i.add(index)
                    break
            if bandera==0:
                for e in edges_between(concepts_wh, concepts_wt, two_way=False,relation="used_for"):
                    bandera=1
                    print(wh," used_for (ant)",wt)
                    borrar.add(c)
                    borrar_i.add(index)
                    break
            if bandera==0:
                for e in edges_between(concepts_wh, concepts_wt,two_way=False,relation="HasProperty"):
                    bandera=1
                    print(wh," HasProperty ",wt)
                    borrar.add(c)
                    borrar_i.add(index)
                    break
        except:
            a=0
    return borrar,borrar_i,antonyms

def encontrar_relaciones_cercanas(indexes,columnas):
    borrar=[]
    borrar_i=[]
    related=[]
    for index in indexes:
        for c in columnas:
            wt=str(index).split("{")[0].split(",")[0]
            wh=str(c).split("{")[0].split(",")[0]
            try:
                for e in edges_for(Label.get(text=wh, language='en').concepts, same_language=True):
                    if (wt == e.start.text or wt == e.end.text) and (e.relation.name in ["related_to","similar_to"]):
                        if "related_to" ==e.relation.name:
                            if wt== e.start.text:
                                if e.end.text==wh:
                                    print(wt," related_to ",wh)
                                    #borrar.append(c)
                                    #borrar_i.append(index)
                                    related.append(wt)
                            else:
                                if wt== e.end.text:
                                    print(wh," related_to ",wt)
                                    #borrar.append(c)
                                    #borrar_i.append(index)
                                    related.append(wt)
                        elif "similar_to" ==e.relation.name:
                            if wh== e.start.text:
                                if e.end.text==wt:
                                    print(wh," similar_to ",wt)
                                    #borrar.append(c)
                                    #borrar_i.append(index)
                                    related.append(wt)
                            else:
                                if e.start.text==wt:
                                    print(wt," similar_to ",wh)
                                    #borrar.append(c)
                                    #borrar_i.append(index)
                                    related.append(wt)
            except:
                a=0
    return borrar,borrar_i,related

def incompatibilidad_semantica(palabra1,pos1,palabra2,pos2):
    contextos_a = []
    if pos1=="VERB" and pos2=="VERB":
        bandera=1
        for a in wn.synsets(palabra1, pos=wn.VERB): #VERB
            contextos_a.append(a)
        contextos_b = []
        for a in wn.synsets(palabra2, pos=wn.VERB):
            contextos_b.append(a)
    else:
        bandera=0
        for a in wn.synsets(palabra1): #VERB
            contextos_a.append(a)
        contextos_b = []
        for a in wn.synsets(palabra2):
            contextos_b.append(a)
    #No hay nada que comparar
    if len(contextos_b)==0 or len(contextos_a)==0:
        return 0
    for b in contextos_b:
        for a in contextos_a:
            if type(a.shortest_path_distance(b))==type(1):
                if(a.shortest_path_distance(b)<=4):
                    print("encontró",a,b)
                    return 1
    if bandera==0:
        return 0
    else:
        print("por que no encontro",palabra1,pos1,palabra2,pos2)
        return 1

def obtener_distancia(texto_v,hipotesis_v,texto_t,texto_h,b_col,b_index):
    lista_l=[]
    for i in range(len(texto_t)):
        lista=[]
        for j in range(len(texto_h)):
            lista.append(np.linalg.norm(texto_v[i] - hipotesis_v[j]))#*wasserstein_distance(texto_2[i],hipotesis_2[j]))
        lista_l.append(lista)
    df_distEuc=pd.DataFrame(lista_l,index=texto_t,columns=texto_h)
    df_distEuc=df_distEuc.drop(b_col[1:],axis=1)
    df_distEuc=df_distEuc.drop(b_index[1:],axis=0)
    return df_distEuc

def wasserstein_mutual_inf(texto_v,hipotesis_v,texto_t,texto_h):  
    lista_l=[]
    lista_muinfor=[]   
    for i in range(len(texto_t)):
        lista=[]
        lista_mu=[]
        for j in range(len(texto_h)):
            lista.append(wasserstein_distance(texto_v[i],hipotesis_v[j]))
            lista_mu.append(mi.mutual_information_2d(np.array(texto_v[i]),np.array(hipotesis_v[j])))
        lista_l.append(lista)
        lista_muinfor.append(lista_mu)
    DFmearth=pd.DataFrame(lista_l,index=texto_t,columns=texto_h)
    DFmutual_inf=pd.DataFrame(lista_muinfor,index=texto_t,columns=texto_h)
    return DFmearth,DFmutual_inf

def entropia(X):
    """Devuelve el valor de entropia de una muestra de datos""" 
    probs = [np.mean(X == valor) for valor in set(X)]
    return round(sum(-p * np.log2(p) for p in probs), 3)



nlp = spacy.load("en_core_web_md") # modelo de nlp

ut.load_vectors_in_lang(nlp,"data/glove.840B.300d.txt") # carga de vectores en nlp.wv

#prueba=pd.read_csv("data/DEV/pruebaDEV.csv")
prueba=pd.read_csv("data/"+sys.argv[1])

textos = prueba["sentence1"].to_list()       # almacenamiento en listas
hipotesis = prueba["sentence2"].to_list()


sumas=[]
distancias=[]
entropias=[]
etiquetas=[]
mearts=[]
mutinf=[]
max_info=[]
list_antonimos=[]
listas_malign=[]
lista_anto=[]
lista_related=[]
lista_relatedT=[]
diferencias=[]
similitud_faltantes=[]
lista_incomp=[]
inicio = time.time()
for i in range(len(textos)):
    print(i)

    t_vectors=ut.get_matrix_rep(textos[i],nlp,pos_to_remove=['PUNCT'], normed=False,lemmatize=False)
    h_vectors=ut.get_matrix_rep(hipotesis[i],nlp,pos_to_remove=['PUNCT'], normed=False,lemmatize=False)
    t_vectors_n=ut.get_matrix_rep(textos[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)
    h_vectors_n=ut.get_matrix_rep(hipotesis[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)
    t_clean=ut.get_words_rep(textos[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)
    h_clean=ut.get_words_rep(hipotesis[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)


    # Obtencion de matriz de alineamiento, matriz de move earth y mutual information
    ma=np.dot(t_vectors_n,h_vectors_n.T)
    #print(len(t_vectors_n),len(h_vectors_n),len(t_clean),len(h_clean))
    m_earth,m_mi=wasserstein_mutual_inf(t_vectors_n,h_vectors_n,t_clean,h_clean)
    ma=pd.DataFrame(ma,index=t_clean,columns=h_clean)

    ###### BORRADO DE COSAS QUE NO OCUPO, SOLO NOS QUEDAMOS CON INFORMACIÓN DE TIPOS DE PALABRA: NOUN, VERB, ADJ Y ADV
    # TAMBIÉN OMITIMOS EL VERBO BE DEBIDO A QUE POR LO REGULAR SE UTILIZA COMO AUXILIAR Y ES UN VERBO COPULATIVO
    # sirve para construir la llamada predicación nominal del sujeto de una oración: 
    # #el sujeto se une con este verbo a un complemento obligatorio llamado atributo que por lo general determina 
    # alguna propiedad, estado o equivalencia del mismo, por ejemplo: "Este plato es bueno". "Juan está casado".
     
    b_col=[0]
    b_index=[0]
    
    col=ma.columns
    borrar=[]
    indexes=ma.index
    for c in col:
        if "{null," in str(c) or "{be,VERB" in str(c) or (",NUM"  not in str(c) and "not," not in str(c) and "PRON" not in str(c) and "NOUN" not in str(c) and "VERB" not in str(c) and "ADJ" not in str(c) and "ADV" not in str(c)):
            borrar.append(c)        
        elif str(c) in indexes:
            borrar.append(c)        
    borrar_i=[]
    for index in indexes:
        if "{null," in str(index) or "{be,VERB" in str(index) or (",NUM"  not in str(index) and "not," not in str(index) and "PRON" not in str(index) and "NOUN" not in str(index) and "VERB" not in str(index) and "ADJ" not in str(index) and "ADV" not in str(index)):
            borrar_i.append(index) 
        elif str(index) in col:
            borrar_i.append(index) 
    ma=ma.drop(borrar,axis=1)
    ma=ma.drop(borrar_i,axis=0)
    m_earth=m_earth.drop(borrar,axis=1)
    m_earth=m_earth.drop(borrar_i,axis=0)
    m_mi=m_mi.drop(borrar,axis=1)
    m_mi=m_mi.drop(borrar_i,axis=0)

    b_col.extend(borrar)
    b_index.extend(borrar_i)
    
    # ELIMINAMOS INFORMACIÓN DONDE SE CORRESPONDAN EN LEMMAS, TENGA UN PRODUCTO IGUAL A 1 Y SEAN IGUALES LOS INDICES Y COLUMNAS
    # SI EL VALOR ES IGUAL A 1 SIGNIFICA QUE ES LA MISMA PALABRA, O SON SINONIMOS
    borrar=[]
    borrar_i=[]
    col=ma.columns
    for index,strings in ma.iterrows():
        lema_i=str(index).split("{")[1].split(",")[0]
        for c in col:
            if index==c:
                borrar_i.append(index)
                borrar.append(c)
            # if strings[c]>=1:
            #     borrar_i.append(index)
            #     borrar.append(c)
            lema_c=str(c).split("{")[1].split(",")[0]
            if lema_i == lema_c:
                borrar_i.append(index)
                borrar.append(c)
    ma=ma.drop(borrar,axis=1)
    ma=ma.drop(borrar_i,axis=0)
    m_earth=m_earth.drop(borrar,axis=1)
    m_earth=m_earth.drop(borrar_i,axis=0)
    m_mi=m_mi.drop(borrar,axis=1)
    m_mi=m_mi.drop(borrar_i,axis=0)
    
    b_col.extend(borrar)
    b_index.extend(borrar_i)
    
    #primera vuelta ---------------------------------------------------------------------------------
    # #PARA REVISAR SI EXISTEN RELACIONES DE SIMILITUD SEMÁNTICA A TRAVÉS DEL USO DE CONCEPNET
    d=[]
    d1=[]
    pasada=0
    while len(ma.index)>0 and len(ma.columns)>0 and pasada<2:
        a=ma.idxmax().values
        b=ma.columns
        rel=[]
        for j in range(len(a)):
            rel.append((a[j],b[j]))
        borrar,borrar_i=encontrar_relaciones(rel[:])
        ma=ma.drop(borrar,axis=1)
        ma=ma.drop(borrar_i,axis=0)
        m_earth=m_earth.drop(borrar,axis=1)
        m_earth=m_earth.drop(borrar_i,axis=0)
        m_mi=m_mi.drop(borrar,axis=1)
        m_mi=m_mi.drop(borrar_i,axis=0)
        b_col.extend(borrar)
        b_index.extend(borrar_i)
        # relaciones contrarias
        #PARA REVISAR SI EXISTEN RELACIONES CONTRARIAS A TRAVÉS DEL USO DE CONCEPNET
        if len(ma.index)>0 and len(ma.columns)>0:
            a=ma.idxmax().values
            b=ma.columns
            rel=[]
            for j in range(len(a)):
                rel.append((a[j],b[j]))
            if pasada==0:
                borrar,borrar_i,d=encontrar_relaciones_contrarias(rel[:])
            else:
                borrar,borrar_i,d1=encontrar_relaciones_contrarias(rel[:])
            ma=ma.drop(borrar,axis=1)
            ma=ma.drop(borrar_i,axis=0)
            m_earth=m_earth.drop(borrar,axis=1)
            m_earth=m_earth.drop(borrar_i,axis=0)
            m_mi=m_mi.drop(borrar,axis=1)
            m_mi=m_mi.drop(borrar_i,axis=0)
            b_col.extend(borrar)
            b_index.extend(borrar_i)
        pasada+=1

    # ultima VUELTA PARA CHECAR RELACIONES CERCANAS --------------------------------------------------
    # relaciones cercanas quitar
    r_l2=[]
    if len(ma.index)>0 and len(ma.columns)>0:
        indexes=ma.index
        columnas=ma.columns
        borrar,borrar_i,r_l2=encontrar_relaciones_cercanas(indexes,columnas)
        ma=ma.drop(borrar,axis=1)
        ma=ma.drop(borrar_i,axis=0)
        m_earth=m_earth.drop(borrar,axis=1)
        m_earth=m_earth.drop(borrar_i,axis=0)
        m_mi=m_mi.drop(borrar,axis=1)
        m_mi=m_mi.drop(borrar_i,axis=0)
        b_col.extend(borrar)
        b_col.extend(borrar)
        b_index.extend(borrar_i)
    #print(ma.index,ma.columns)
        
    #   ALMACENAMIENTO DE TODA LA INFORMACIÓN PROCESADA
    #alamacenado de resultados
    sumas.append(ma.min().sum())# cambie a minimo antes era suma
    if len(ma.columns)==0:
        max_info.append(0)
    else:
        max_info.append(ma.max().sum())#/(ma.shape[1]))) modificación1
    entropias.append(entropia(ma.round(1).values.flatten()))
    m_distancia = obtener_distancia(t_vectors,h_vectors,t_clean,h_clean,b_col,b_index)
    
    m_earth=m_earth*m_distancia
    mearts.append(m_earth.max().sum())#suma de sumas lo cambie a max
    mutinf.append(m_mi.max().sum())#cambie de suma a max
    distancias.append(m_distancia.sum().sum()) #cambie de maximas a sumas
    
    if len(ma.columns)==0:
        diferencias.append(1)
    elif len(ma.columns)>0 and len(ma.index)==0:
        diferencias.append(0.5)
    elif len(ma.columns)>0 and len(ma.index)>0:
        diferencias.append(len(ma.columns)/len(ma.index))
    
    d2=[]   
    if d==[] and d1==[]:
        d2=[]
    else:
        d2=list(set(d+d1))
    r2=[]   
    if r_l2==[]:
        r2=[]
    else:
        r2=list(set(r_l2))
    list_antonimos.append(len(d2))
    listas_malign.append(ma)
    lista_anto.append(d2)
    lista_related.append(len(r2))
    lista_relatedT.append(r2)
    #print(ma)
    incomp=0
    if len(ma.index)>0 and len(ma.columns)>0:
        a=ma.idxmax().values
        b=ma.columns
        for j in range(len(a)):
            incomp+=incompatibilidad_semantica(str(a[j]).split("{")[0],str(a[j]).split("}")[0].split(",")[1],str(b[j]).split("{")[0],str(b[j]).split("}")[0].split(",")[1])
        lista_incomp.append(incomp)
    st=""
    sh=""
    for t__1 in ma.index:
        st=st+" "+str(t__1).split("{")[0]        
    for t__2 in ma.columns:
        sh=sh+" "+str(t__2).split("{")[0]
    doc1 = nlp(st)
    doc2 = nlp(sh)
    #print(doc1, "<->", doc2, doc1.similarity(doc2))
    #print(st,sh)
    if sh!="" and st!="":
        similitud_faltantes.append(doc1.similarity(doc2))
    elif st!="" and sh=="":
        similitud_faltantes.append(1)
    else:
        similitud_faltantes.append(0)
fin = time.time()
clases=prueba["gold_label"].values
temp1 =np.array([sumas,distancias,entropias,mutinf,mearts,max_info,list_antonimos,lista_related,similitud_faltantes,lista_incomp,clases])
df_resultados = pd.DataFrame(temp1.T,columns=["suma","distancias","entropias","mutual_info","m_earth","max_info_p","antonimos","relaciones","sim_faltantes","incompatibilidad","CLASS"])
df_resultados.to_csv("salida/resultados/"+sys.argv[1]+".csv",index=False)

print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")