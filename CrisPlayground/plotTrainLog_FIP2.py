import matplotlib.pyplot as plt

testEpoch = []
testLoss = []
testHuber = []
testMSE = []

trainEpoch = []
trainLoss = []
trainHuber = []
trainMSE = []

lines = [line.rstrip('\n').split() for line in open('trainLogFIP2')]


for line in lines :
    if line[0] == 'TRAINING' :
        trainEpoch.append(float(line[4]))
        trainLoss.append(float(line[6]))
        trainHuber.append(float(line[8]))
        trainMSE.append(float(line[10]))
    elif line[0] == "VALIDATION" :
        testEpoch.append(float(line[4]))
        testLoss.append(float(line[6]))
        testHuber.append(float(line[8]))
        testMSE.append(float(line[10]))



plt.plot(trainEpoch, trainLoss, color = 'b', alpha = 0.5, label = "Train LOSS")
plt.plot(testEpoch, testLoss, color = 'r', alpha = 0.5, label = "Test LOSS")
plt.plot(trainEpoch, trainHuber, color = None, alpha = 0.5, label = "Train Huber")
plt.plot(testEpoch, testHuber, color = None, alpha = 0.5, label = "Test Huber")
plt.plot(trainEpoch, trainMSE, color = None, alpha = 0.5, label = "Train MSE")
plt.plot(testEpoch, testMSE, color = None, alpha = 0.5, label = "Test MSE")

plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Loss')

#plt.yscale('log')

plt.tight_layout()

plt.savefig("trainLog_FIP2.png")


plt.show()
