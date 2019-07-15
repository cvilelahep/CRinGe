import matplotlib.pyplot as plt

testEpoch = []
testLoss = []

trainEpoch = []
trainLoss = []

lines = [line.rstrip('\n').split() for line in open('trainLog')]


for line in lines :
    if line[0] == 'TRAINING' :
        trainEpoch.append(float(line[4]))
        trainLoss.append(float(line[6]))
    elif line[0] == "VALIDATION" :
        testEpoch.append(float(line[4]))
        testLoss.append(float(line[6]))


plt.plot(trainEpoch, trainLoss, color = 'b', alpha = 0.5, label = "Train")
plt.plot(testEpoch, testLoss, color = 'r', alpha = 0.5, label = "Test")

plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Loss')

#plt.yscale('log')

plt.tight_layout()

plt.savefig("trainLog.png")


plt.show()
