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
        if df[i][1] < gre_min :
            df[i][1] = 0
        else:
            df[i][1] = 1

    #GPA is less than 3 or equal then 0
    gpa_min = 3
    for i in range (1, len(df)):
        if df[i][2] < gpa_min :
            df[i][2] = 0
        else:
            df[i][2] = 1
    

    
    return df


def probabilitad3(df):
    #frequency rank 1
    nb_entry = len(df)-1
    nb_rank = 0
    for i in range (1, len(df)):
        if df[i][3] == 1:
            nb_rank += 1

    #p(r=1,ad=0,gpa=1,gre=0)
    nb1 = 0
    for i in range (1, len(df)):
        if df[i][3] == 1 and df[i][0] == 0 and df[i][2] == 1 and df[i][1] == 0:
            nb1 += 1
    p1 = nb1/nb_entry

    #p(r=1,ad=0,gpa=0,gre=1)
    nb2 = 0
    for i in range (1, len(df)):
        if df[i][3] == 1 and df[i][0] == 0 and df[i][2] == 0 and df[i][1] == 1:
            nb2 += 1
    p2 = nb2/nb_entry

    #p(r=1,ad=0,gpa=0,gre=0)
    nb3 = 0
    for i in range (1, len(df)):
        if df[i][3] == 1 and df[i][0] == 0 and df[i][2] == 0 and df[i][1] == 0:
            nb3 += 1
    p3 = nb3/nb_entry

    #p(r=1,ad=0,gpa=1,gre=1)

    nb4 = 0
    for i in range (1, len(df)):
        if df[i][3] == 1 and df[i][0] == 0 and df[i][2] == 1 and df[i][1] == 1:
            nb4 += 1
    p4 = nb4/nb_entry

    ptotal = p1+p2+p3+p4

    #p(rk=1,ad=1,gpa=1,gre=0)
    nb1 = 0
    for i in range (1, len(df)):
        if df[i][3] == 1 and df[i][0] == 1 and df[i][2] == 1 and df[i][1] == 0:
            nb1 += 1

    p6 = nb1/nb_entry

    #p(rk=1,ad=1,gpa=0,gre=1)
    nb2 = 0
    for i in range (1, len(df)):
        if df[i][3] == 1 and df[i][0] == 1 and df[i][2] == 0 and df[i][1] == 1:
            nb2 += 1

    p7 = nb2/nb_entry

    #p(rk=1,ad=1,gpa=0,gre=0)
    nb3 = 0
    for i in range (1, len(df)):
        if df[i][3] == 1 and df[i][0] == 1 and df[i][2] == 0 and df[i][1] == 0:
            nb3 += 1
    
    p8 = nb3/nb_entry

    #p(rk=1,ad=1,gpa=1,gre=1)
    nb4 = 0
    for i in range (1, len(df)):
        if df[i][3] == 1 and df[i][0] == 1 and df[i][2] == 1 and df[i][1] == 1:
            nb4 += 1
    
    p9 = nb4/nb_entry

    ptotal1 =p1+p2+p3+p4
    ptotal2 = p6+p7+p8+p9
    pfinal = ptotal1/ptotal2

    return pfinal


if __name__ == '__main__':
    

    df = convert('binary.csv')
    print(df)

    df = discretize(df)
    print ( "la probabilidad de que un estudiante que proviene de una escuela con rango 1 no haya sido admitido en la universidad es de: ")
    print(probabilitad3(df))
    



    
    '''
    
    df2 = convert('binary.csv')
    pb3 = probabilitad2_sin_discretise(df)

    '''