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

    def evaluate(self, test_dataset) -> None:
        # Inicializa una lista para almacenar las predicciones de clase
        predictions = []

        # Inicializa una lista de clases únicas en tus datos de entrenamiento
        classes = self.training_dataset[self.class_name].unique()

        # Itera sobre cada instancia en el conjunto de prueba
        for index, test_instance in test_dataset.iterrows():
            # Inicializa un diccionario para almacenar las probabilidades a posteriori para cada clase
            posterior_probabilities = {}

            # Itera sobre cada clase
            for class_ in classes:
                posterior_probability = 1.0  # Inicializa la probabilidad a posteriori para la clase

                # Itera sobre cada atributo en la instancia de prueba, excluyendo la columna de clase
                for attribute, value in test_instance.iloc[:-1].items():
                    # Accede a la tabla de verosimilitud correspondiente al atributo
                    likelihood_table = self.likelihood_tables[attribute]

                    # Obtener Pr[ A | H ] para el valor del atributo
                    conditional_probability = likelihood_table.at[value, class_]
                    print(class_, attribute, conditional_probability)

                    # Multiplica la probabilidad condicional por la probabilidad a posteriori Pr[H]
                    posterior_probability *= conditional_probability

                # Calcula la probabilidad a priori Pr[H] para la clase
                prior_probability = len(self.training_dataset[self.training_dataset[self.class_name] == class_]) / len(self.training_dataset)

                # Calcula la probabilidad a posteriori final Pr[H|A] para la clase
                posterior_probabilities[class_] = prior_probability * posterior_probability

            # Ahora, "posterior_probabilities" contiene las probabilidades a posteriori para cada clase
            # Puedes seleccionar la clase con la probabilidad más alta como la predicción final
            predicted_class = max(posterior_probabilities, key=posterior_probabilities.get)
            predictions.append(predicted_class)

        # Ahora, "predictions" contiene las predicciones de clase para todas las instancias de prueba
        


    def fit(self) -> None:
        # Calcular las tablas de frecuencia
        self.computeFrequencyTables()

        # Calcular las tablas de verosimilitud
        self.computeLikelihoodTables()
