'''

        @name ad45335SPI.py

        @description:  module to support the spi interface to 3 ad45335 evms

        @author robert antonetti

        @revision history

        1/14/25 Origin

'''
import time
import spidev
import RPi.GPIO as gpio
import sys
import csv
import os

from threading import Timer

class ad45335SPI():

        '''
        class to support interface of the rpi to 3 ad45335 evms

        '''

        def __init__(self,bus=0,device=0,speed=100000,mode=1,testMsgs=False, dacAmp=[10]*96,refDACVal=100,refBoard=2, refDAC=28, dc=False):
                self.bus=bus
                self.device=device
                self.speed=speed
                self.mode=mode

                self.testMsgs=testMsgs
                self.dac1CSPin=2 #chip select for board 1
                self.dac2CSPin=3 #chip select for board 2
                self.dac3CSPin=4  #chip select for board 3
                self.timerRate=1000.0  #timer rate in Hz
                self.timerActive=True  #time control
                #set up of spi port
                gpio.setmode(gpio.BCM)
                gpio.setup(self.dac1CSPin,gpio.OUT)
                gpio.setup(self.dac2CSPin,gpio.OUT)
                gpio.setup(self.dac3CSPin,gpio.OUT)

                gpio.output(self.dac1CSPin,gpio.HIGH)
                gpio.output(self.dac2CSPin,gpio.HIGH)
                gpio.output(self.dac3CSPin,gpio.HIGH)

                #dac parameters
                self.bits=14
                self.gain=50
                self.reference=4.06
                #self.reference=1.35

                #Dac modulation amplitude
                self.dacAmp=dacAmp

                self.dacState=False #track dac state ref+amp or ref-amp

                #current baord and dac
                self.board=0
                self.DAC=0

                #reference board and dac
                self.refBoard=refBoard
                self.refDAC=refDAC
                self.refDACVal=refDACVal
                self.dacAmp[self.refBoard*32+self.refDAC]=self.refDACVal
                self.dc=dc
                self.refDACSet=False
                self.spi=spidev.SpiDev(0,0)
                self.spi.open(self.bus,self.device)
                self.spi.max_speed_hz=self.speed
                self.spi.mode=self.mode
                #create the dac update timer
                self.updateTimer=Timer(1.0/(self.timerRate),self.updateDACs)

        def getRefBrd(self):
                ''' @brief method to get the refBoard
                '''
                return(self.refBoard)

        def setRefBrd(self,board):
                ''' @brief method to set the refBoard
                '''
                self.refBoard=board

        def getdacAmps(self):
                ''' @brief method to get the dacAmp list
                '''
                return(self.dacAmp)

        def setdacAmps(self,amps):
                ''' @brief method to set the dacAmp list
                '''
                self.dacAmp=amps

        def getdacAmp(self,idx):
                ''' @brief method to get dacAmp index
                '''
                return(self.dacAmp[idx])

        def setdacAmp(self,idx,value):
                ''' @brief method to set dacAmp index
                '''
                self.dacAmp[idx]=value

        def getRef(self):
                ''' @brief method to get reference
                '''
                return self.reference

        def setRef(self,ref):
                ''' @brief method to set reference
                '''
                self.reference=ref

        def startUpdateTimer(self):
                '''@brief method to start tjee dac update process
                '''
                self.timerActive=True
                self.updateTimer.start()

        def stopUpdateTimer(self):
                '''@brief method to stop the dac update process
                '''
                self.timerActive=False

        def updateDACs(self):
                '''
                @brief method to update all of the dacs
                '''

                #test for the acive board to determine the chip select to activate
                #note three boards repeat the same logic shown for board 0 below
                #if(self.board==0):

                self.board=1


                while(self.board!=4):
                        self.DAC=0
                        #spin through all 32 dacs
                        while(self.DAC != 32):

                                #reference board and reference dac
                                if((self.refBoard==self.board) and (self.DAC==self.refDAC)):
                                        value=self.voltageCode(self.refDACVal)
                                elif(self.dc==False):
                                #toggle dac amp to ref+amp
                                        if(self.dacState):
                                                value=self.voltageCode(self.dacAmp[((self.board-1)*32)+self.DAC]+self.refDACVal)
                                                #print(str(self.board)+ " : "+str(self.DAC)+" : "+str(self.dacAmp[((self.board-1)*32)+self.DAC]+self.refDACVal))
                                                self.dacState=False

                                        #toggle dac amp to ref-amp
                                        else:
                                                value=self.voltageCode(self.refDACVal-self.dacAmp[((self.board-1)*32)+self.DAC])
                                                #print(str(self.board)+ " : "+str(self.DAC)+" : "+str(self.refDACVal-self.dacAmp[((self.board-1)*32)+self.DAC]))
                                                self.dacState=True
                                else:
                                        value=self.voltageCode(self.dacAmp[((self.board-1)*32)+self.DAC])
                                        #print('dac: '+str((self.board*32)+self.DAC)+' value: '+str(value)+' voltage: '+str(self.dacAmp[(self.board*32)+self.DAC]))

                               #compose the message to the dac addr[4:0],data[13:0] 19 bits total
                               #chip select falling edge resets clock counter
                               #dac accepts 19 cycles then ignores all following cycles until next falling edge of chip select

                                msg0=((self.DAC<<3) | ((value>>11) & 0x07))

                                #print('msg0'+str(msg0))

                                msg1=((value>>3) &0xFF)

                                #print('msg1: '+str(msg1))

                                msg2=((value & 0x07)<<5)

                                #print('msg2: '+str(msg2))


                                if(self.board==1):
                                        gpio.output(self.dac1CSPin,gpio.LOW)
                                else:
                                        gpio.output(self.dac1CSPin,gpio.HIGH)

                                if(self.board==2):
                                        gpio.output(self.dac2CSPin,gpio.LOW)
                                else:
                                        gpio.output(self.dac2CSPin,gpio.HIGH)
                                if(self.board==3):
                                        gpio.output(self.dac3CSPin,gpio.LOW)
                                else:
                                        gpio.output(self.dac3CSPin,gpio.HIGH)

                                #send spi data mssg0,msg1,msg2 all bytes

                                reply = self.spi.xfer2([msg0, msg1, msg2])

                                gpio.output(self.dac1CSPin,gpio.HIGH)
                                gpio.output(self.dac2CSPin,gpio.HIGH)
                                gpio.output(self.dac3CSPin,gpio.HIGH)

                                #next dac
                                self.DAC+=1

                        self.board+=1

                #continue the update process if and only if timerActive is True

                if(self.timerActive):
                        trigTimer=Timer(1.0/(self.timerRate),self.updateDACs)
                        trigTimer.start()


        def voltageCode(self,volts):
                '''
                @brief method to generate the 14 bit dac voltage code
                '''
                return(int((2**self.bits*volts)/(self.gain*self.reference)))

        def voltsFromCode(self,code):
                '''
                @brief method to return the voltage generated by a code
                '''
                if(code!=0):
                        rtn=((self.gain*self.reference*code)/(2**self.bits))
                else: rtn=0
                return rtn

        def testCS(self,cs):
                gpio.output(cs,gpio.HIGH)
                time.sleep(.2)
                gpio.output(cs,gpio.LOW)

        def readFile(self):
                file_path="/home/ldantes/Desktop/data.csv"

                if not os.path.exists(file_path):
                        print("Error File Does Not Exist!")

                voltageList=[]

                with open(file_path, "r", newline="") as fileObj:
                        fileContents=csv.reader(fileObj)

                        for row in fileContents:
                                #print(row)
                                for value in row:
                                        #print(value)
                                        voltageList.append(float(value))

                print(voltageList)

                #original map
                ''' DACMap=[[(6, 1), (2, 1) , (5, 1), (10, 1), (28, 2), (18, 2), (22, 2), (19, 2), (15, 2), (11, 2), (1, 2), (3, 2)],
                [(3, 1), (1, 1) , (0, 1), (4, 1), (27, 2), (30, 2), (26, 2), (25, 2), (16, 2), (13, 2), (4, 2), (0, 2)],
                [(8, 1), (9, 1) , (7, 1), (12, 1), (24, 2), (29, 2), (20, 2), (17, 2), (9, 2), (8, 2), (2, 2), (6, 2)],
                [(11, 1), (15, 1) , (13, 1), (16, 1), (31, 2), (23, 2), (21, 2), (14, 2), (12, 2), (7, 2), (10, 2), (5, 2)],
                [(17, 1), (20, 1) , (14, 1), (21, 1), (5, 3), (10, 3), (7, 3), (12, 3), (14, 3), (21, 3), (23, 3), (31, 3)],
                [(19, 1), (22, 1) , (25, 1), (26, 1), (6, 3), (2, 3), (8, 3), (9, 3), (17, 3), (20, 3), (29, 3), (24, 3)],
                [(29, 1), (24, 1) , (23, 1), (31, 1), (0, 3), (4, 3), (13, 3), (16, 3), (25, 3), (26, 3), (30, 3), (27, 3)],
                [(18, 1), (28, 1) , (30, 1), (27, 1), (3, 3), (1, 3), (11, 3), (15, 3), (19, 3), (22, 3), (18, 3), (28, 3)]]'''

                #Gen 1 - 96 Element
                DACMap=[[(6, 1), (2, 1) , (5, 1), (10, 1), (28, 3), (18, 3), (22, 3), (19, 3), (15, 3), (11, 3), (1, 3), (3, 3)],
                [(3, 1), (1, 1) , (0, 1), (4, 1), (27, 3), (30, 3), (26, 3), (25, 3), (16, 3), (13, 3), (4, 3), (0, 3)],
                [(8, 1), (9, 1) , (7, 1), (12, 1), (24, 3), (29, 3), (20, 3), (17, 3), (9, 3), (8, 3), (2, 3), (6, 3)],
                [(11, 1), (15, 1) , (13, 1), (16, 1), (31, 3), (23, 3), (21, 3), (14, 3), (12, 3), (7, 3), (10, 3), (5, 3)],
                [(17, 1), (20, 1) , (14, 1), (21, 1), (5, 2), (10, 2), (7, 2), (12, 2), (14, 2), (21, 2), (23, 2), (31, 2)],
                [(19, 1), (22, 1) , (25, 1), (26, 1), (6, 2), (2, 2), (8, 2), (9, 2), (17, 2), (20, 2), (29, 2), (24, 2)],
                [(29, 1), (24, 1) , (23, 1), (31, 1), (0, 2), (4, 2), (13, 2), (16, 2), (25, 2), (26, 2), (30, 2), (27, 2)],
                [(18, 1), (28, 1) , (30, 1), (27, 1), (3, 2), (1, 2), (11, 2), (15, 2), (19, 2), (22, 2), (18, 2), (28, 2)]]

                # Dual band map
                ''''DACMap = [[(6, 1), (2, 1) , (5, 1), (10, 1), (6, 2), (2, 2) , (5, 2), (10, 2), (28, 3), (18, 3), (27, 3), (30, 3)],
                [(3, 1), (1, 1) , (0, 1), (4, 1), (3, 2), (1, 2) , (0, 2), (4, 2), (24, 3), (29, 3), (31, 3), (23, 3)],
                [(8, 1), (9, 1) , (7, 1), (12, 1), (8, 2), (9, 2) , (7, 2), (12, 2), (22, 3), (19, 3), (26, 3), (25, 3)],
                [(11, 1), (15, 1) , (13, 1), (16, 1), (11, 2), (15, 2) , (13, 2), (16, 2), (20, 3), (17, 3), (21, 3), (14, 3)],
                [(17, 1), (20, 1) , (14, 1), (21, 1), (17, 2), (20, 2) , (14, 2), (21, 2), (15, 3), (11, 3), (16, 3), (13, 3)],
                [(19, 1), (22, 1) , (25, 1), (26, 1), (19, 2), (22, 2) , (25, 2), (26, 2), (9, 3), (8, 3), (12, 3), (7, 3)],
                [(29, 1), (24, 1) , (23, 1), (31, 1), (29, 2), (24, 2) , (23, 2), (31, 2), (1, 3), (3, 3), (4, 3), (0, 3)],
                [(28, 1), (18, 1) , (30, 1), (27, 1), (18, 2), (28, 2) , (30, 2), (27, 2), (2, 3), (6, 3), (10, 3), (5, 3)]]'''

                listIdx=0
                voltIdx=0
                for idx in range(8):
                        mapRow=DACMap[idx] #select row in the eleDACMap
                        for jdx in range(12):
                                mapTuple=mapRow[jdx]
                                listIdx=mapTuple[0]+(32*(mapTuple[1]-1))
                                print(str(mapTuple[1])+":"+str(mapTuple[0])+":"+str(voltageList[listIdx])+" : "+str(listIdx))
                                self.dacAmp[listIdx]=voltageList[voltIdx]
                                voltIdx+=1


                #self.dacAmp=[25]*96

                print("DAC_AMP: "+str(self.dacAmp))

#module standalone test code

if(__name__=="__main__"):

        obj=ad45335SPI()

        obj.readFile()

        obj.startUpdateTimer()

        count=0
        while(1):
                time.sleep(1)
                count+=1
                if(count%10==0):
                        print('running')