# Arguments used for RC network

## single zone

-a 120 0 5040 7 12 22 10 10 -1 0

## cav

-a 300 0 2016 1 3 4 500 100 0 0

## integrated rc

-a 300 0 2016 5 7 11 500 100 3 0

# Single Zone log

![Initial Performance](_3_radiant_py3/single_zone_swarm_performance.png)

# Integrated RC network

![image-20220309191732435](C:\Users\wulic\AppData\Roaming\Typora\typora-user-images\image-20220309191732435.png)
$$
C_{cav}\frac{dT_{cav}}{dt} = \frac{T_{out} - T_{cav}}{R_{out, cav}} + \frac{T_{room} - T_{cav}}{R_{cav, room}} + \dot{Q}_{sol, cav}\\
C_{room}\frac{dT_{room}}{dt} = \frac{T_{out} - T_{room}}{R_{out, room}} + \frac{T_{sur} - T_{room}}{R_{room, sur}} + \frac{T_{cav} - T_{room}}{R_{cav, room}} + \dot{Q}_{sol, room} + \dot{Q}_{int, room}\\
C_{sur}\frac{dT_{sur}}{dt} = \frac{T_{room} - T_{sur}}{R_{room, sur}} + \frac{T_{so} - T_{sur}}{R_{sur, so}}+ \dot{Q}_{sol, sur} + \dot{Q}_{int, sur}\\
C_{so}\frac{dT_{so}}{dt} = \frac{T_{sur} - T_{so}}{R_{sur, so}} + \frac{T_{si} - T_{so}}{R_{si, so}}\\
C_{si}\frac{dT_{si}}{dt} = \frac{T_{so} - T_{si}}{R_{so, si}}
$$

$$
x^T = [T_{cav}, T_{room}, T_{sur}, T_{so}, T_{si}]\\
u^T = [T_{out}, \dot{Q}_{sol, cav}, \dot{Q}_{sol, room}, \dot{Q}_{int, room}, \dot{Q}_{sol, sur}, \dot{Q}_{int, sur}, \frac{dT_{so}}{dt}]\\
y = \dot{Q}_{rslab} = \frac{T_{sur} - T_{so}}{R_{sur, so}} + \frac{T_{si} - T_{so}}{R_{si, so}} - C_{so}\frac{dT_{so}}{dt} \\
$$

$$
\begin{bmatrix}
    \frac{dT_{cav}}{dt} \\\frac{dT_{room}}{dt} \\\frac{dT_{sur}}{dt} \\\frac{dT_{so}}{dt} \\\frac{dT_{si}}{dt}
\end{bmatrix}
=
\begin{bmatrix}
(\frac{-1}{R_{out, cav}C_{cav}} + \frac{-1}{R_{cav, room}C_{cav}}) ,   \frac{1}{R_{cav, room}C_{cav}} ,0 , 0 , 0 \\
\frac{1}{R_{cav, room}C_{room}} ,  ( \frac{-1}{R_{out, room}C_{room}} + \frac{-1}{R_{room, sur}C_{room}} + \frac{-1}{R_{cav, room}C_{room}} ), \frac{1}{R_{room, sur}C_{room}} , 0 , 0 \\
0 ,   \frac{1}{R_{room, sur}C_{sur}} , (\frac{-1}{R_{room, sur}C_{sur}} + \frac{-1}{R_{sur, so}C_{sur}}),\frac{1}{R_{sur, so}C_{sur}} , 0 \\
0 ,  0 ,  \frac{1}{R_{sur, so}C_{so}},(\frac{-1}{R_{sur, so}C_{so}} + \frac{-1}{R_{si, so}C_{so} }) , \frac{1}{R_{si, so}C_{so} } \\
0 , 0 , 0 , \frac{1} {R_{so, si}C_{si}} , \frac{-1}{R_{so, si}C_{si}}
\end{bmatrix}
\begin{bmatrix}
	T_{cav}\\T_{room} \\ T_{sur}\\T_{so}\\T_{si}
\end{bmatrix}
+
\begin{bmatrix}
\frac{1}{R_{out, cav}C_{cav}}  , 1 ,0 , 0 ,0 , 0 , 0\\
\frac{1}{R_{out, room}C_{room}}  , 0 , 1 , 1 ,0 , 0 , 0\\
0 , 0 , 0 , 0 ,1 , 1 , 0\\
0 , 0 , 0 , 0 ,0 , 0 , 0\\
0 , 0 , 0 , 0 ,0 , 0 , 0
\end{bmatrix}
\begin{bmatrix}
	T_{out}\\ \dot{Q}_{sol, cav}\\ \dot{Q}_{sol, room}\\ \dot{Q}_{int, room}\\ \dot{Q}_{sol, sur}\\ \dot{Q}_{int, sur}\\ \frac{dT_{so}}{dt}
\end{bmatrix}
$$

$$
y = \dot{Q}_{rslab} =
\begin{bmatrix}
0 & 0 & \frac{1}{R_{sur, so}} & (\frac{-1}{R_{sur, so}} + \frac{-1}{R_{si, so}}) & \frac{1}{R_{si, so}}
\end{bmatrix}
\begin{bmatrix}
	T_{cav}\\T_{room} \\ T_{sur}\\T_{so}\\T_{si}
\end{bmatrix}
+
\begin{bmatrix}
0&0&0&0&0&0& -C_{so}
\end{bmatrix}
\begin{bmatrix}
	T_{out}\\ \dot{Q}_{sol, cav}\\ \dot{Q}_{sol, room}\\ \dot{Q}_{int, room}\\ \dot{Q}_{sol, sur}\\ \dot{Q}_{int, sur}\\ \frac{dT_{so}}{dt}
\end{bmatrix}
$$

$$
A = 
\begin{bmatrix}
(\frac{-1}{R_{out, cav}C_{cav}} + \frac{-1}{R_{cav, room}C_{cav}}) &   \frac{1}{R_{cav, room}C_{cav}} & 0 & 0 & 0 \\
\frac{1}{R_{cav, room}C_{room}} &  ( \frac{-1}{R_{out, room}C_{room}} + \frac{-1}{R_{room, sur}C_{room}} + \frac{-1}{R_{cav, room}C_{room}} )& \frac{1}{R_{room, sur}C_{room}} & 0 & 0 \\
0 &   \frac{1}{R_{room, sur}C_{sur}} &  (\frac{-1}{R_{room, sur}C_{sur}} + \frac{-1}{R_{sur, so}C_{sur}})&\frac{1}{R_{sur, so}C_{sur}} & 0 \\
0 &  0 &  \frac{1}{R_{sur, so}C_{so}}&(\frac{-1}{R_{sur, so}C_{so}} + \frac{-1}{R_{si, so}C_{so} }) &  \frac{1}{R_{si, so}C_{so} } \\
0 & 0 & 0 &\frac{1} {R_{so, si}C_{si}}  & \frac{-1}{R_{so, si}C_{si}}
\end{bmatrix}
\\
B = 
\begin{bmatrix}
\frac{1}{R_{out, cav}C_{cav}}  & 1 & 0 & 0 &0 & 0 & 0\\
\frac{1}{R_{out, room}C_{room}}  & 0 & 1 & 1 &0 & 0 & 0\\
0 & 0 & 0 & 0 &1 & 1 & 0\\
0 & 0 & 0 & 0 &0 & 0 & 0\\
0 & 0 & 0 & 0 &0 & 0 & 0
\end{bmatrix}
\\
c = 
\begin{bmatrix}
0 & 0 & \frac{1}{R_{sur, so}} & (\frac{-1}{R_{sur, so}} + \frac{-1}{R_{si, so}}) & \frac{1}{R_{si, so}}
\end{bmatrix}
\\
d = 
\begin{bmatrix}
0&0&0&0&0&0& -C_{so}
\end{bmatrix}
$$

parameter initial values:

1. r out cav, 0.036 K/W
2. r cav room, 0.0036 K/W
3. r out room, 0.036 K/W
4. r room sur, 10 K/W
5. r sur so, 40 K/W
6. r si so, 300 K/W
7. c cav,  (air 75300 j/k)
8. c room, (air 376500 J/K)
9. c sur (concrete, 2E7 J/K)
10. c so (water, 2629 J/K)
11. c si (common insulation material 3360000 J/K)

## Initial guessing

| 3.60E-02 | 3.60E-03 | 3.60E-02 | 1.00E+01 | 4.00E+01 | 3.00E+02 |
| -------- | -------- | -------- | -------- | -------- | -------- |
| 9.00E-01 | 1.413    | 8.70E-01 | 1.14E+01 | 4.17E+01 | 3.00E+02 |
| 5.67     | 8.24     | 9.63     | 1.57E+01 | 3.92E+01 | 2.94E+02 |
| 0.019    | 0.034    | 0.046    | 1.89E-03 | 7.00E-04 | 4.00E-03 |

```matlab
3.60E-02	3.60E-03	3.60E-02	1.00E+01	4.00E+01	3.00E+02	7.53E+04	3.77E+05	2.00E+07	2.63E+03	3.36E+06		5.37E+03
9.00E-01	1.413	8.70E-01	1.14E+01	4.17E+01	3.00E+02	7.53E+04	3.77E+05	2.00E+07	2.63E+03	3.36E+06		
5.67	8.24	9.63	1.57E+01	3.92E+01	2.94E+02	7.53E+04	3.77E+05	2.00E+07	2.63E+03	3.36E+06		
0.019	0.034	0.046	1.89E-03	7.00E-04	4.00E-03	8.85E+05	4.12E+06	2.80E+07	2.70E+06	2.42E+16		Jaewan
```



# Cav RC

![image-20220311145414130](C:\Users\wulic\AppData\Roaming\Typora\typora-user-images\image-20220311145414130.png)
$$
C_{cav}\frac{dT_{cav}}{dt} = \frac{T_{out} - T_{cav}}{R_{out, cav}} + \frac{T_{room} - T_{cav}}{R_{cav, room}} + \alpha_{sol, cav}\dot{Q}_{sol, cav}\\
x^T = [T_{cav}, \alpha_{sol, cav}]\\
u^T = [T_{out},T_{room}, \dot{Q}_{sol}]\\
y = T_{cav}\\
$$

$$
\begin{bmatrix}
\frac{dT_{cav}}{dt} 
\end{bmatrix}
=
\begin{bmatrix}
(\frac{-1}{R_{out, cav}C_{cav}} + \frac{-1}{R_{cav, room}C_{cav}}) 
\end{bmatrix}
\begin{bmatrix}
T_{cav} 
\end{bmatrix}
+
\begin{bmatrix}
\frac{1}{R_{room, cav}C_{cav}}, \frac{1}{R_{out, cav}C_{cav}}, \alpha_{sol, cav}
\end{bmatrix}
\begin{bmatrix}
T_{room} \\
T_{out}\\
\dot{Q}_{sol, cav}
\end{bmatrix}
$$

$$
y = T_{cav} = [1] [T_{cav}]  + [0,0,0]
\begin{bmatrix}
T_{room} \\
T_{out}\\
\dot{Q}_{sol, cav}
\end{bmatrix}
$$

## initial guessing

p[0] = r out cav, K/W, 1 / 51.92 = 0.019

p[1] = r cav room, K/W, 1/29.07 = 0.034

p[2] = c cav, K/W, 1 / 1.13e-6 = 885000

p[3] = alpha sol cav, -, 1

```python
r out cav, r cav room, c cav, alpha sol cav
0.019,0.034,885000,1 #calculated， initial error, best_cost=1.75e+5
0.019,0.034,885000,2E-5 #trivial and errors
2.58782929e-02 3.73895712e-02 8.85000163e+05 1.10505928e-05 # 500 particle 50 iterations， best_cost=1.21e+5
1.35746337e-02 4.80479809e-02 8.85000197e+05 1.69230027e-05 # 500 particle 200 iterations， best_cost=7.3e+4
```

![](_3_radiant_py3/cav_no_solar.png)

![](_3_radiant_py3/cav_with_solar.png)

![](_3_radiant_py3/cav_with_solar_500_50.png)

![](_3_radiant_py3/cav_with_solar_500_50_2.png)

![](_3_radiant_py3/cav_with_solar_500_50_3.png)

## test multi-processes



| 前50个数据 | pos                                                          |      |      |
| ---------- | ------------------------------------------------------------ | ---- | ---- |
| 3进程并发  | 8.16505902e-04  1.94427052e-03  8.85000490e+05 -1.09815616e-01 | 16   |      |
| 单进程     | 2.02431304e-03 4.91892908e-03 8.85000045e+05 2.27941988e-01  | 29   |      |



2.34823223e-03 6.96492546e-03 8.84999875e+05 4.34410595e-01
