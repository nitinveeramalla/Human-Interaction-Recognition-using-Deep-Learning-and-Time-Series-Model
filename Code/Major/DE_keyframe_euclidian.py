import random
import cv2
import os
import numpy as np
import pandas as pd




pathr="D:/Major/segmented_set/validation/" # Change PATh
patho="D:/Major/segmented_frameset/validation/" # Change PATh

filenames=os.listdir(pathr)

name=filenames[24]

cv2video = cv2.VideoCapture(pathr+name)
height = cv2video.get(cv2.CAP_PROP_FRAME_HEIGHT)
width  = cv2video.get(cv2.CAP_PROP_FRAME_WIDTH)
fc = cv2video.get(cv2.CAP_PROP_FRAME_COUNT )
fps = cv2video.get(cv2.CAP_PROP_FPS)
    
print("Video Name:",name)
print ("Video Dimension: height:{} width:{}".format( height, width))
print("FPS:",fps) 
print("FrameCount:",fc)
print("Video duration (sec):", fc / fps)

MAX_NUMBER_OF_FRAMES = int(fc)
    
fname,fmat=name.split('.')
f,s,lbl=fname.split('_')

TOTAL_KEY_FRAMES = 5

STOPPING_ITERATION = 50

NUMBER_OF_NP_CANDIDATES = 10

# Population matrix.
NP = []

# Mutation vector.
MV = []

# Trail vector.
TV = []

# Scale factor.
F = 0.9

# Cr probability value.
Cr = 0.6

def genetic_algo():
    # Calculate AED for a chromosome.
    def getAED( KF ):
        ED_sum = 0
        for i in range(1, TOTAL_KEY_FRAMES - 1):
            while True:
                try:
                    im1 = cv2.imread(pathr+name + str(KF[i]) + ".jpg",0)
                    im2 = cv2.imread(pathr+name + str(KF[i+1]) + ".jpg",0)
                    ED_sum += cv2.norm(im1, im2, cv2.NORM_L2)
                except:
                    print (i, KF, KF[i], KF[i+1])
                    continue
                break
        return ED_sum/(TOTAL_KEY_FRAMES - 1)
    
    # INITIALISATION : Generates population NP of 10 parent vectors (and AEDs).
    def initialize_NP():
        for i in range(NUMBER_OF_NP_CANDIDATES):
            NP.append(sorted(random.sample(range(1, MAX_NUMBER_OF_FRAMES+1), TOTAL_KEY_FRAMES)))
            NP[-1].append(getAED(NP[-1]))
            print (NP[-1])
    
    # MUTATION
    def mutation(num):
        R = random.sample(NP,3)
        global MV
        MV[:] = []
    
        for i in range(TOTAL_KEY_FRAMES):
            MV_value = int(NP[num][i] + F*(R[1][i] - R[2][i]))
            if(MV_value < 1):
                MV.append(1)
            elif(MV_value > MAX_NUMBER_OF_FRAMES):
                MV.append(MAX_NUMBER_OF_FRAMES)
            else:
                MV.append(MV_value)
        MV.sort()
        MV.append(getAED(MV))
    
    # CROSSOVER (uniform crossover with Cr = 0.6).
    def crossover(parent, mutant):
        print ("mutant: ", mutant)
        print ("parent: ", parent)
        for j in range(TOTAL_KEY_FRAMES) :
            if(random.uniform(0,1) < Cr) :
                TV.append(mutant[j])
            else:
                TV.append(parent[j])
        TV.sort()
        TV.append(getAED(TV))
        print ("TV    : ", TV)
    
    # SELECTION : Selects offspring / parent based on higher ED value.
    def selection(parent, trail_vector):
        if(trail_vector[-1] > parent[-1]):
            parent[:] = trail_vector
            print ("yes", parent)
        else:
            print ("no")
    
    # bestParent returns the parent with then maximum ED value.
    def bestParent(population):
        Max_AED_value = population[0][-1]
        Best_Parent_Index = population[0]
        for parent in population:
            if (parent[-1] > Max_AED_value):
                Max_AED_value = parent[-1]
                Best_Parent_Index = parent
        return Best_Parent_Index
    
    
    
    
    
    
    initialize_NP()
    for GENERATION in range(STOPPING_ITERATION):
        for i in range(NUMBER_OF_NP_CANDIDATES):
            print ("---------------------", "PARENT:", i+1 , "GENERATION:", GENERATION+1, "---------------------")
            mutation(i)
            crossover(NP[i], MV)
            selection(NP[i], TV)
            print (NP[i])
            TV[:] = []
            print ("")
        print ("")
    best_parent = bestParent(NP)
    
    best_parent.pop()
    
    print ("best solution is: ", best_parent)


    
    frc=0
    for i in best_parent:
        cv2video.set(cv2.CAP_PROP_POS_FRAMES, i)
        print('Position:', int(cv2video.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = cv2video.read()
        #cv2.imshow('frame', frame)
        #cv2.waitKey(0)
        st=fname
        st=fname+"_frame"+str(frc)+".jpg"
        cv2.imwrite(os.path.join(patho, st), frame)
        print('Frame name:',st)
        print('.....Saved at:',patho)
        frc+=1
        cv2.waitKey(0)
    cv2.destroyAllWindows()


genetic_algo()
    
    