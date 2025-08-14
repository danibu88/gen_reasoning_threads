from top2vec import Top2Vec

# Load the existing model
model = Top2Vec.load('/Users/danielburkhardt/PhD/Prototyping/PyCharmProjects/models-extracted/top2vec.model')

# Save the model again to a new file
model.save('/Users/danielburkhardt/PhD/Prototyping/PyCharmProjects/models-extracted/new_top2vec.model')