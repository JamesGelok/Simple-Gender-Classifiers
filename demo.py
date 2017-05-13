from sklearn import tree
from sklearn import naive_bayes
from sklearn import neural_network

#  ╔╦╗┌─┐┌┬┐┌─┐
#   ║║├─┤ │ ├─┤
#  ═╩╝┴ ┴ ┴ ┴ ┴

#    [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], 
	 [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], 
	 [159, 55, 37], 
	 [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 
	 'male', 'male', 'female', 'female', 'female', 
     'male', 'male']

#  ╔╦╗┌─┐┌─┐┬┌─┐┬┌─┐┌┐┌  ╔╦╗┬─┐┌─┐┌─┐
#   ║║├┤ │  │└─┐││ ││││   ║ ├┬┘├┤ ├┤ 
#  ═╩╝└─┘└─┘┴└─┘┴└─┘┘└┘   ╩ ┴└─└─┘└─┘
#  ╔═╗┌─┐┬ ┬┌─┐┌─┐┬┌─┐┌┐┌  ╔╗╔┌─┐┬┬  ┬┌─┐  ╔╗ ┌─┐┬ ┬┌─┐┌─┐
#  ║ ╦├─┤│ │└─┐└─┐│├─┤│││  ║║║├─┤│└┐┌┘├┤   ╠╩╗├─┤└┬┘├┤ └─┐
#  ╚═╝┴ ┴└─┘└─┘└─┘┴┴ ┴┘└┘  ╝╚╝┴ ┴┴ └┘ └─┘  ╚═╝┴ ┴ ┴ └─┘└─┘
#  ╔╦╗┬ ┬┬ ┌┬┐┬   ╦  ┌─┐┬ ┬┌─┐┬─┐  ╔═╗┌─┐┬─┐┌─┐┌─┐┌─┐┌┬┐┬─┐┌─┐┌┐┌
#  ║║║│ ││  │ │───║  ├─┤└┬┘├┤ ├┬┘  ╠═╝├┤ ├┬┘│  ├┤ ├─┘ │ ├┬┘│ ││││
#  ╩ ╩└─┘┴─┘┴ ┴   ╩═╝┴ ┴ ┴ └─┘┴└─  ╩  └─┘┴└─└─┘└─┘┴   ┴ ┴└─└─┘┘└┘

#clf is short for classifier
clfDT   = tree.DecisionTreeClassifier()
clfGNB  = naive_bayes.GaussianNB()
clfMLP = neural_network.MLPClassifier()

# The .fit method trains the decision tree on our dataset
clfDT  = clfDT.fit(X,Y)
clfGNB = clfGNB.fit(X,Y)
clfMLP = clfMLP.fit(X,Y)

# Predition for some new data. This should be male.
prediction = clfDT.predict([[190,70,43]])
prediction = clfGNB.predict([[190,70,43]])
prediction = clfMLP.predict([[190,70,43]])

print("Using a Decision Tree classifier, predition: " + str(prediction))
print("Using a Gaussian Naive Bayes classifier, predition: " + str(prediction))
print("Using a Multi-layer Perceptron classifier, predition: " + str(prediction))

