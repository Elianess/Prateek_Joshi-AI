#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y):
    #Определение для X и Y минимального и максимального значений, 
    #которые будут использоваться при построениии сетки
    
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    '''Определили минимальное и максимальное знаения для координат
    вдоль осей X и Y, которые будут использоваться в нашей сетке.
    По сути, эта сетка представляет собой набор значений для 
    вычисленийя функции, чтобы можно было визуализировать границы 
    классов.
        Определим шаг сетки и создадим ее, используя заданные
    минимальные и максимальные значения'''
    
    #Определение величины шага для построения сетки
    mesh_step_size = 0.01
    
    #Определение сетки для значений X и Y
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))
    
    '''Запустим классификатор для всех точек сетки'''
    
    #Выполнение классификатора на сетке данных
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    
    #Переформирование выходного массива
    output = output.reshape(x_vals.shape)
    
    '''Создадим график, выберем цветовую схему и разметим все точки'''
    #Создание графика
    plt.figure()
    
    #Выбор цветово схемы для графика
    plt.pcolormesh(x_vals, 
                   y_vals, 
                   output, 
                   cmap = plt.cm.gray)
    
    #Размещение тренировочных точек на графике
    plt.scatter(X[:, 0], 
                X[:, 1], 
                c = y, 
                s = 75, 
                edgecolors='black', 
                linewidths = 1, 
                cmap = plt.cm.Paired)
    
    '''Укажем границы графика, используя минимальные и максимальные значения,
    добавим деления и отобразим график'''
    
    #Определение границ графика
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    
    #Определение делений на осях X и Y
    plt.xticks(np.arange(int(X[:, 0].min() - 1), 
                         int(X[:, 0].max() + 1), 
                         1.0))
    
    plt.yticks(np.arange(int(X[:, 1].min() - 1), 
                         int(X[:, 1].max() + 1), 
                         1.0))
    
    plt.show()

