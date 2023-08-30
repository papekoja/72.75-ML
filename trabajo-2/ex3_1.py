import pandas as pd
import numpy as np


def convert(name_file):
    #open csv file
    df = pd.read_csv(name_file, sep=',', header=None, engine='python')

    #convert to numpy array
    df = df.to_numpy()

    #convert in int
    for i in range (1, len(df)):
        df[i][0] = int(df[i][0])
        df[i][1] = int(df[i][1])
        df[i][2] = float(df[i][2])
        df[i][3] = int(df[i][3])

    
    return df


def discretize (df):
    #discretize the data
    #if GRE is less than 500 then 0
    gre_min = 500
    for i in range (1, len(df)):
        if df[i][1] < gre_min or df[i][1] == gre_min:
            df[i][1] = 0
        else:
            df[i][1] = 1

    #GPA is less than 3 or equal then 0
    gpa_min = 3
    for i in range (1, len(df)):
        if df[i][2] < gpa_min or df[i][2] == gpa_min:
            df[i][2] = 0
        else:
            df[i][2] = 1
    

    
    return df

#Calcular la probabilidad de que una persona que proviene de una escuela con rango 1 no haya sido admitida en la universidad.
def probabilidad(df):
    #frequency rank 1
    nb_entry = len(df)-1
    print(nb_entry)
    nb_rank = 0
    for i in range (1, len(df)):
        if df[i][3] == 1:
            nb_rank += 1

    p_rank1= nb_rank/nb_entry

    #frequency no admitted

    nb_no_admitted = 0
    for i in range (1, len(df)):
        if df[i][0] == 0:
            nb_no_admitted += 1

    p_admitted = nb_no_admitted/nb_entry
    print(p_admitted)

    #frequency rank 1 and admitted
    nb_rank1_admitted = 0
    for i in range (1, len(df)):
        if df[i][3] == 1 and df[i][0] == 0:
            nb_rank1_admitted += 1
    p_rank1_admitted = nb_rank1_admitted/nb_no_admitted

    #bayes
    p_admitted_rank1 = (p_rank1_admitted*p_admitted)/(p_rank1)

    print("La probabilidad de que una persona que proviene de una escuela con rango 1 no haya sido admitida en la universidad es de: ", p_admitted_rank1)

    return p_admitted_rank1






if __name__ == '__main__':
    

    df = convert('binary.csv')
    df = discretize(df)
    pb1 = probabilidad(df)
    pb2 = probabilitad2(df)
    #sin discretizar
    '''
    
    df2 = convert('binary.csv')
    pb3 = probabilitad2_sin_discretise(df)

    '''