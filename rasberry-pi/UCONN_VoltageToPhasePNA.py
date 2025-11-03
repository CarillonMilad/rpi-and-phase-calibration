import time
import spidev 
import RPi.GPIO as gpio 
import sys
import csv
import os
import threading
import VNA21Test
import numpy as np
from gpiozero import PWMOutputDevice

LowBandDACMap = [[(5,2,2),(6,2,2),(22,1,2),(24,1,2),(21,0,3),(20,0,3),(6,2,3),(5,2,3)],
[(22,2,2),(9,2,2),(19,1,2),(5,1,2),(7,0,3),(11,0,3),(9,2,3),(22,2,3)],
[(10,2,2),(12,2,2),(15,1,2),(17,1,2),(19,0,3),(18,0,3),(12,2,3),(10,2,3)],
[(5,3,2),(7,3,2),(9,1,2),(7,1,2),(1,0,3),(4,0,3),(7,3,3),(5,3,3)],
[(12,3,2),(4,3,2),(21,0,2),(15,0,2),(9,1,3),(7,1,3),(2,3,1),(23,3,1)],
[(10,3,2),(8,3,2),(7,0,2),(10,0,2),(15,1,3),(17,1,3),(21,3,1),(19,3,1)],
[(21,3,2),(14,3,2),(13,0,2),(19,0,2),(5,1,3),(19,1,3),(16,3,1),(8,3,1)],
[(2,3,2),(22,3,2),(5,0,2),(1,0,2),(24,1,3),(22,2,3),(6,3,1),(4,3,1)],
[(21,0,1),(15,0,1),(21,1,1),(20,1,1),(2,3,3),(23,3,3),(16,2,1),(7,3,1)],
[(7,0,1),(10,0,1),(5,1,1),(19,1,1),(21,3,3),(19,3,3),(19,2,1),(12,2,1)],
[(19,0,1),(13,0,1),(17,1,1),(15,1,1),(8,3,3),(16,3,3),(20,2,1),(9,2,1)],
[(1,0,1),(5,0,1),(7,1,1),(9,1,1),(4,3,3),(6,3,3),(7,2,1),(6,2,1)]
]

HighBandOneDACMap =[[(3,2,2),(4,2,2),(1,1,2),(2,1,2),(14,0,3),(15,0,3),(4,2,3),(3,2,3)], 
[(7,2,2),(8,2,2),(20,1,2),(21,1,2),(16,0,3),(17,0,3),(8,2,3),(7,2,3)] ,
[(1,3,2),(21,2,2),(23,1,2),(24,0,2),(22,0,3),(10,0,3),(1,3,3),(21,2,3)] ,
[(20,2,2),(24,2,2),(4,1,2),(6,1,2),(8,0,3),(12,0,3),(24,2,1),(20,2,3)] ,
[(23,2,2),(13,2,2),(14,1,2),(3,1,2),(9,0,3),(13,0,3),(13,2,3),(23,2,3)] ,
[(19,2,2),(11,2,2),(16,1,2),(18,1,2),(3,0,3),(23,0,3),(11,2,3),(19,2,3)] ,
[(18,2,2),(17,2,2),(10,1,2),(8,1,2),(6,0,3),(5,0,3),(17,2,3),(18,2,3)] ,
[(16,2,2),(15,2,2),(12,1,2),(11,1,2),(2,0,3),(13,1,3),(15,2,3),(16,2,3)] ,
[(14,2,2),(11,3,2),(14,0,2),(13,1,2),(12,1,3),(8,1,3),(26,3,1),(14,2,3)] ,
[(6,3,2),(13,3,2),(16,0,2),(20,0,2),(10,1,3),(11,1,3),(20,3,1),(22,3,1)],
[(9,3,2),(3,3,2),(22,0,2),(17,0,2),(16,2,3),(18,2,3),(24,3,1),(25,3,1)] ,
[(16,3,2),(17,3,2),(8,0,2),(11,0,2),(14,1,3),(3,1,3),(18,3,1),(14,3,1)],
[(18,3,2),(15,3,2),(12,0,2),(9,0,2),(6,1,3),(4,1,3),(15,3,1),(17,3,1)] ,
[(24,3,2),(19,3,2),(18,0,2),(3,0,2),(24,0,3),(23,1,3),(10,3,1),(3,3,1)] ,
[(20,3,2),(25,3,2),(23,0,2),(6,0,2),(21,1,3),(20,1,3),(9,3,1),(11,3,1)],
[(26,3,2),(23,3,2),(4,0,2),(2,0,2),(2,1,3),(1,1,3),(12,3,1),(13,3,1)]  ,
[(14,0,1),(13,1,1),(23,1,1),(24,0,1),(26,3,3),(3,2,1),(14,2,1),(15,2,1)],
[(16,0,1),(20,0,1),(24,1,1),(20,1,1),(20,3,3),(22,3,3),(5,3,1),(17,2,1)] ,
[(22,0,1),(17,0,1),(1,1,1),(2,1,1),(24,3,3),(25,3,3),(18,2,1),(11,2,1)],
[(8,0,1),(11,0,1),(6,1,1),(4,1,1),(18,3,3),(14,3,3),(10,2,1),(13,2,1)]  ,
[(9,0,1),(13,0,1),(3,1,1),(14,1,1),(17,3,3),(15,3,3),(23,2,1),(24,2,1)],
[(3,0,1),(18,0,1),(18,1,1),(16,1,1),(3,3,3),(10,3,3),(22,2,1),(21,2,1)] ,
[(6,0,1),(23,0,1),(8,1,1),(10,1,1),(11,3,3),(9,3,3),(1,3,1),(8,2,1)],
[(2,0,1),(4,0,1),(11,1,1),(12,1,1),(13,3,3),(12,3,3),(5,2,1),(4,2,1)]  
]
class wgsTest():
        def __init__(self,bus=0,device=0,speed=100000,mode=0, dacAmpLB = [0]*96, dacAmpHB = [0]*192, testMsgs=False):
                self.bus=bus
                self.device=device
                self.speed=speed
                self.mode=mode
                 
                self.testMsgs=testMsgs
                self.timerRate=1000.0  #timer rate in Hz
                gpio.setup(27, gpio.OUT)
                gpio.setup(22, gpio.OUT)
                gpio.setup(17, gpio.OUT)
                
                #current baord and dac
                self.board = 0
                self.DAC = 0		
                
                self.dacAmpHB = dacAmpHB
                self.dacAmpLB = dacAmpLB
                
                self.spi0=spidev.SpiDev()
                self.spi0.open(0, 0)
                self.spi0.max_speed_hz=self.speed
                self.spi0.mode=self.mode							
                
                self.spi1=spidev.SpiDev()
                self.spi1.open(0, 1)
                self.spi1.max_speed_hz=self.speed
                self.spi1.mode=self.mode							
                
                self.spi2=spidev.SpiDev()
                self.spi2.open(0, 2)
                self.spi2.max_speed_hz=self.speed
                self.spi2.mode=self.mode	

        def writeDAC(self, PTR, SID, BUFA, boardID):

            #Clear display
            S = 1 << 22
            R = 1 << 23
            W = 0 << 23
                                
            #data = 0x01
            data = (SID << 12) | (PTR)
            regAdd = (6 << 18)  # PTR 
            msg1 = (W | S | regAdd | data)
            #print("{:03b}".format(msg1))
                
            data = BUFA
            regAdd = (0x00)  # BUFA
            msg2 = W | S | (regAdd << 18) | (data << 4)
                
            data = 0x8555
            regAdd = (0x04)  # CON
            msg3 = W | S | (regAdd << 18) | (data)
                
            if(boardID==1):
                #print("Board ID = 1")	
                reply = self.spi0.xfer2([msg1 >> 16, msg1 >> 8, msg1])    
                reply = self.spi0.xfer2([msg2 >> 16, msg2 >> 8, msg2])        
                reply = self.spi0.xfer2([msg3 >> 16, msg3 >> 8, msg3])
            if(boardID==2):
                #print("Board ID = 2")
                reply = self.spi1.xfer2([msg1 >> 16, msg1 >> 8, msg1])    
                reply = self.spi1.xfer2([msg2 >> 16, msg2 >> 8, msg2])        
                reply = self.spi1.xfer2([msg3 >> 16, msg3 >> 8, msg3])
            if(boardID==3):
                #print("Board ID = 3")
                reply = self.spi2.xfer([msg1 >> 16, msg1 >> 8, msg1])    
                reply = self.spi2.xfer([msg2 >> 16, msg2 >> 8, msg2])        
                reply = self.spi2.xfer([msg3 >> 16, msg3 >> 8, msg3]) 
                
        def writeData(self):
            start = str(input("Enter the desired starting frequncy: "))
            stop = str(input("Enter the desired stopping frequncy: "))
            points = str(input("Enter the number of points you whish to collect: "))
            element = str(input("Enter an Identifier for the element that you will be testing (this will be added to the file name): "))
            #sweep from 0 to 10 volts taking data every 0.1 volt
            for V in range(0,100,1):
                v = V/10
                #translate analog voltage to percentage of maximum dac voltage fro digital processing
                voltage = int((4096*v)/21)
                
                #loop over all DAC pins
                gpio.output(27, gpio.HIGH)
                gpio.output(22, gpio.HIGH)
                gpio.output(17, gpio.HIGH)
                for idx in range(12):
                    mapRow = LowBandDACMap[idx] 
                    for jdx in range(8):
                        mapTuple= mapRow[jdx]
                        self.writeDAC(int(mapTuple[0] - 1), int(mapTuple[1]), voltage, int(mapTuple[2]))

                for idx in range(24):
                    mapRow = HighBandOneDACMap[idx] 
                    for jdx in range(8):
                        mapTuple= mapRow[jdx]
                        self.writeDAC(int(mapTuple[0] - 1), int(mapTuple[1]), voltage, int(mapTuple[2]))
                gpio.output(27, gpio.LOW)
                gpio.output(22, gpio.LOW)
                gpio.output(17, gpio.LOW)
                if (v > 4):
                        time.sleep(30)
                else:
                        time.sleep(10)
                signal = VNA21Test.sweep(start,stop,points)
                phases = np.round(np.degrees(np.angle(signal)), 3)
                phaseString = []
                for i, number in enumerate(phases):
                    pString = str(number)
                    phaseString.append(pString)

                    if((i+1)%10)==0 and (i+1) < len(phases):
                        phaseString.append("\n")
                    elif (i+1) < len(phases):
                        phaseString.append(", ")

                phaseString = "".join(phaseString)
                fileName  =f"WGS_{element}_{start}-{stop}GHz_at_{v}V_.txt"
                with open(fileName, "w") as f:
                    f.write(phaseString)
    
if(__name__=="__main__"):
    gpio.setmode(gpio.BCM)
    gpio.setup(5, gpio.OUT)
    gpio.output(5,gpio.HIGH)
    gpio.setup(0, gpio.OUT)
    gpio.output(0,gpio.HIGH)
    gpio.setup(6, gpio.OUT)
    gpio.output(6,gpio.HIGH)

    gpio.setup(19,gpio.OUT)
    gpio.output(19, gpio.HIGH)
    gpio.setup(26,gpio.OUT)
    gpio.output(26, gpio.HIGH)
    gpio.setup(13,gpio.OUT)
    gpio.output(13, gpio.HIGH)

    #start Trigger signal
    pwm_pin = PWMOutputDevice(12, active_high=True, initial_value=0,frequency=1000)
    pwm_pin.value = 0.5    
    
    obj = wgsTest()
    
    obj.writeData()
    
    #end Triggger signal
    pwm_pin.close()
