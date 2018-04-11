import random as rnd
from dirichlet import helper

class query():
    concept = ''
    probabilites = [0, 0, 0]

class learner(object):
    def __init__(self, r):
        self.concepts = ["none of the other ones", "sure about one", "sure about two", "unsure"]
        self.data = [[],[],[],[]]
        self.requiredDatasets = r
    def generateDataset(self):
        # Generates a new random dataset for a 
        q = query()
        q.concept = 'unknown dataset'
        rand = [rnd.random(), rnd.random(), rnd.random()]
        rand.sort(reverse=True)
        q.probabilities = helper.normalize(rand)
        return q
    def learn(self):
        print("\n============Nest Dataset===============\n")
        dataset = self.generateDataset()
        self.printDataset(dataset)
        self.printConcepts()
        selection = input("What concept best describes the probability vector? (Select by number) ")
        if selection.isdigit() == False:
            print("Error not a number")
            return
        else:  
            selection = int(selection)
        # If the user wants to create a new label
        if selection == 0:
            new_concept = input("Please name the new concept for the given probability vector: ")
            self.concepts.append(new_concept)
            self.data.append(dataset.probabilities)
        else:
            self.data[selection].append(dataset.probabilities)
    def printDataset(self, dataset):
        # Print out the generated query
        out = "Please label this " + dataset.concept + " for this probability vector: " + str(dataset.probabilities)
        print(out)   
    def printConcepts(self):#
        print("The following concepts exist already: ")
        i = 0
        for c in self.concepts:
            if i == 0:
                out = str(i) + ": " + c
            else:
                out = str(i) + ": " + c + ", Number of datasets for this concept: " + str(len(self.data[i]))
            print(out)
            i = i + 1
    def learningDone(self):
        x = 0
        for d in self.data:
            if x == 0: # Skip the first concept as it serves only to create new ones
                x = 1
                continue
            if len(d) < self.requiredDatasets:
                return False     
        print("Learning Done, got {} of each dataset!".format(self.requiredDatasets))
        return True
    def getDistribution(self):
        return self.data[1:]
    def getConcepts(self):
        return self.concepts[1:]