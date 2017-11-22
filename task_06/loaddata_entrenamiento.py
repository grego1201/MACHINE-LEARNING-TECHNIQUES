# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:05:59 2016

@author: FranciscoP.Romero
"""
import codecs
def load_data_entrenamiento(nombre):
    states = []
    names = []
    count = 0
    f = codecs.open(nombre, "r", "utf-8")
    for line in f:
            if count > 0: 
                # remove double quotes
                row = line.replace ('"', '').split(",")
                if row != []:
            	      states.append(map(str, row))
                else:
                   names = line.replace ('"', '').split(",")[1:]
            count += 1
            
 
    
    
   
    return states,names