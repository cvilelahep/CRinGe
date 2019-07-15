import matplotlib.pyplot as plt

testEpoch = []
testLoss = []
testHuber = []
testKLD = []

trainEpoch = []
trainLoss = []
trainHuber = []
trainKLD = []

lines = [line.rstrip('\n').split() for line in open('trainLogFIP_5VarPar')]


for line in lines :
    if line[0] == 'TRAINING' :
        trainEpoch.append(float(line[4]))
        trainLoss.append(float(line[6]))
        trainHuber.append(float(line[8]))
        trainKLD.append(float(line[10]))
    elif line[0] == "VALIDATION" :
        testEpoch.append(float(line[4]))
        testLoss.append(float(line[6]))
        testHuber.append(float(line[8]))
        testKLD.append(float(line[10]))



plt.plot(trainEpoch, trainLoss, color = 'b', alpha = 0.5, label = "Train LOSS")
plt.plot(testEpoch, testLoss, color = 'r', alpha = 0.5, label = "Test LOSS")
plt.plot(trainEpoch, trainHuber, color = None, alpha = 0.5, label = "Train Huber")
plt.plot(testEpoch, testHuber, color = None, alpha = 0.5, label = "Test Huber")
plt.plot(trainEpoch, trainKLD, color = None, alpha = 0.5, label = "Train KLD")
plt.plot(testEpoch, testKLD, color = None, alpha = 0.5, label = "Test KLD")

plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Loss')

#plt.yscale('log')

plt.tight_layout()

plt.savefig("trainLog_FIP_5VarPar.png")


plt.show()
