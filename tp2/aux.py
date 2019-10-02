import pandas
import matplotlib.pyplot as plt

a=[[0, 64, 0, 'relu', 0.48934550971201024, 0.5], [0, 64, 0, 'sigmoid', 0.97260273900024785, 0.92857140302658081], [0, 64, 0.2, 'relu', 0.92922374438295985, 0.92857140302658081], [0, 64, 0.2, 'sigmoid', 0.96194824889370301, 0.8571428656578064], [0, 64, 0.5, 'relu', 0.9421613394216134, 0.92857140302658081], [0, 64, 0.5, 'sigmoid', 0.96423135391653403, 0.92857140302658081], [0, 128, 0, 'relu', 0.49391171998447841, 0.5], [0, 128, 0, 'sigmoid', 0.94444444390010984, 0.92857140302658081], [0, 128, 0.2, 'relu', 0.50228310524963715, 0.5], [0, 128, 0.2, 'sigmoid', 0.95357686380998907, 0.92857140302658081], [0, 128, 0.5, 'relu', 0.93074581367240106, 0.92857140302658081], [0, 128, 0.5, 'sigmoid', 0.95053272459604965, 0.92857140302658081], [0, 256, 0, 'relu', 0.49619482487122579, 0.5], [0, 256, 0, 'sigmoid', 0.95966514405231684, 0.92857140302658081], [0, 256, 0.2, 'relu', 0.51141552583993477, 0.5], [0, 256, 0.2, 'sigmoid', 0.95129375878715805, 0.96428573131561279], [0, 256, 0.5, 'relu', 0.49086757972723094, 0.5], [0, 256, 0.5, 'sigmoid', 0.94748858456560825, 0.92857140302658081], [1, 64, 0, 'relu', 0.7701674278831192, 0.78571426868438721], [1, 64, 0, 'sigmoid', 0.8112633175682985, 0.67857140302658081], [1, 64, 0.2, 'relu', 0.50684931579427273, 0.5], [1, 64, 0.2, 'sigmoid', 0.81963470274273487, 0.67857140302658081], [1, 64, 0.5, 'relu', 0.71308980249378784, 0.6428571343421936], [1, 64, 0.5, 'sigmoid', 0.77549467230133451, 0.71428573131561279], [1, 128, 0, 'relu', 0.51141552511415522, 0.5], [1, 128, 0, 'sigmoid', 0.85692541838780933, 0.6071428656578064], [1, 128, 0.2, 'relu', 0.48249619546001904, 0.5], [1, 128, 0.2, 'sigmoid', 0.78919330243832087, 0.75], [1, 128, 0.5, 'relu', 0.49923896499238962, 0.5], [1, 128, 0.5, 'sigmoid', 0.80060882746175366, 0.71428573131561279], [1, 256, 0, 'relu', 0.50304414007580267, 0.5], [1, 256, 0, 'sigmoid', 0.82420091342345403, 0.71428573131561279], [1, 256, 0.2, 'relu', 0.49162861491628612, 0.5], [1, 256, 0.2, 'sigmoid', 0.78234398764199498, 0.6071428656578064], [1, 256, 0.5, 'relu', 0.78386605811082788, 0.71428573131561279], [1, 256, 0.5, 'sigmoid', 0.7983257223482001, 0.67857140302658081]]


for e in a:
    print("%d %d %0.1f %s %0.2f %0.2f" %(e[0],e[1],e[2],e[3],e[4],e[5]))
    
from pandas.tools.plotting import scatter_matrix

df = pandas.DataFrame(a, columns=['augment', 'nhid', 'dropout','actFunc', 'trainacc', 'testacc'])

scatter_matrix(df)

plt.show()