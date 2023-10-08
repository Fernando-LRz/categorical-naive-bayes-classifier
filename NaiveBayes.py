import pandas

class NaiveBayes:

    def __init__(self, training_dataset) -> None:       
        # Inicializar los atributos
        self.training_dataset = training_dataset
        self.class_name = training_dataset.columns[-1]
        self.attributes = training_dataset.columns[:-1]

        # Crear un diccionario para almacenar las tablas de frecuencia
        self.frequency_tables = {}

        # Crear un diccionario para almacenar las tablas de verosimilitud
        self.likelihood_tables = {}

    def computeFrequencyTables(self) -> None:
        # Iterar sobre cada atributo y calcular su tabla de frecuencia
        for attribute in self.attributes:
            frequency_table = pandas.crosstab(self.training_dataset[attribute], self.training_dataset[self.class_name]) # filas, columnas
            
            # Aplicar el suavizado, sumando 1 a todas las frecuencias
            smoothed_frequency_table = frequency_table + 1

            # Almacenar la tabla de frecuencia en el diccionario
            self.frequency_tables[attribute] = smoothed_frequency_table

    def computeLikelihoodTables(self) -> None:
        # Itera sobre cada tabla de frecuencia
        for attribute, frequency_table in self.frequency_tables.items():
            # Calcular la tabla de verosimilitud
            likelihood_table = frequency_table.div(frequency_table.sum(axis=0), axis=1)
            
            # Almacenar la tabla de verosimilitud en el diccionario
            self.likelihood_tables[attribute] = likelihood_table

    def evaluate(self, test_dataset) -> pandas.DataFrame:
        # Crear una lista para almacenar las predicciones de clase
        predictions = []

        # Crear una lista de clases únicas en el conjunto de datos de entrenamiento
        classes = self.training_dataset[self.class_name].unique()

        # Itera sobre cada instancia en el conjunto de prueba
        for index, test_instance in test_dataset.iterrows():
            # Crear un diccionario para almacenar las probabilidades a posteriori para cada clase
            posterior_probabilities = {}

            # Iterar sobre cada clase
            for class_ in classes:
                posterior_probability = 1.0  # Inicializar la probabilidad a posteriori para la clase

                # Iterar sobre cada atributo en la instancia de prueba, excluyendo la columna de clase
                for attribute, value in test_instance.iloc[:-1].items():
                    # Acceder a la tabla de verosimilitud correspondiente al atributo
                    likelihood_table = self.likelihood_tables[attribute]

                    # Obtener el valor en la tabla de verosimilitud para el valor de la instancia
                    value_in_likelihood_table = likelihood_table.at[value, class_]

                    # Multiplicar la probabilidad a posteriori de la clase por el valor obtenido de la tabla
                    posterior_probability *= value_in_likelihood_table
                
                # Calcular la probabilidad a priori para la clase
                prior_probability = len(self.training_dataset[self.training_dataset[self.class_name] == class_]) / len(self.training_dataset)

                # Calcular la probabilidad a posteriori final para la clase
                posterior_probabilities[class_] = prior_probability * posterior_probability

            # Seleccionar la clase con la probabilidad más alta
            predicted_class = max(posterior_probabilities, key=posterior_probabilities.get)
            predictions.append(predicted_class)

        # Copiar el conjunto de entrenamiento en un dataframe nuevo
        result = test_dataset.copy()

        # Agregar la lista de predicciones como una nueva columna al dataframe result
        result['Prediction'] = predictions

        # Agregar una columna con la comparación entre la predicción y la clase real
        result['Match'] = result[self.class_name] == result['Prediction']

        return result

    def fit(self) -> None:
        # Calcular las tablas de frecuencia
        self.computeFrequencyTables()

        # Calcular las tablas de verosimilitud
        self.computeLikelihoodTables()

    def getFrequencyTables(self) -> dict:
        return self.frequency_tables
    
    def getLikelihoodTables(self) -> dict:
        return self.likelihood_tables