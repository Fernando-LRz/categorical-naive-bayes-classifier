import pandas

class NaiveBayes:

    def __init__(self, training_dataset, attributes, class_name) -> None:       
        # Inicializar los atributos
        self.training_dataset = training_dataset
        self.class_name = class_name
        self.attributes = attributes

        # Crear un diccionario para almacenar las tablas de frecuencia
        self.frequency_tables = {}

        # Crear un diccionario para almacenar las tablas de verosimilitud
        self.likelihood_tables = {}

        self.possible_classes = []

    def computeFrequencyTables(self) -> None:
        # Iterar sobre cada atributo y calcular su tabla de frecuencia
        for attribute in self.attributes:
            frecuency_table = pandas.crosstab(self.training_dataset[attribute], self.training_dataset[self.class_name]) # filas, columnas
            self.frequency_tables[attribute] = frecuency_table

    def computeLikelihoodTables(self) -> None:
        # Itera sobre cada atributo en el diccionario de frecuencias
        for attribute, frequency_table in self.frequency_tables.items():
            # Calcula las probabilidades condicionales dividiendo la tabla de frecuencia por el total de cada clase
            likelihood_table = frequency_table.div(frequency_table.sum(axis=0), axis=1)
            
            # Almacena la tabla de verosimilitud en el diccionario con el nombre del atributo como clave
            self.likelihood_tables[attribute] = likelihood_table


    def fit(self) -> None:
        # Calcular las tablas de frecuencia
        self.computeFrequencyTables()

        self.computeLikelihoodTables()