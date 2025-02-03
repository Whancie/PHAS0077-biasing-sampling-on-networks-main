import openmm

temp = openmm.Platform.getPlatform(2)
result = openmm.Platform.getName(temp)
print(result)