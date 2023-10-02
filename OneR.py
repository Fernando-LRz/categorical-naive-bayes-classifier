import pandas

class OneR:

    def __init__(self, training_dataset, attributes, class_name) -> None:       
        # Inicializar los atributos
        self.training_dataset = training_dataset
        self.class_name = class_name
        self.attributes = attributes

        # Crear un diccionario para almacenar las tablas de frecuencia
        self.frequency_tables = {}

        # Crear un DataFrame para almacenar las reglas
        self.rules = pandas.DataFrame(columns=['Attribute', 'Value', 'Class', 'Errors', 'Instances'])

        # Crear un DataFrame para almacenar los errores totales
        self.total_errors = pandas.DataFrame(columns=['Attribute', 'Errors', 'Instances'])

        # Crear un atributo para almacenar la regla que será seleccionada (Serie de Pandas)
        self.selected_rule = None

        # Crear un par de atributos para almacenar información de la regla seleccionada
        self.selected_rule_index = None
        self.selected_rule_name= None

        # Crear un atributo para almacenar los resultados de la predicción (DataFrame de Pandas)
        self.result = None

    def computeFrequencyTables(self) -> None:
        # Iterar sobre cada atributo y calcular su tabla de frecuencia
        for attribute in self.attributes:
            frecuency_table = pandas.crosstab(self.training_dataset[attribute], self.training_dataset[self.class_name]) # filas, columnas
            self.frequency_tables[attribute] = frecuency_table

    def calculateErrors(self) -> None:
        # Iterar sobre las tablas de frecuencia 
        for attribute, frecuency_table in self.frequency_tables.items():
            # Obtener los valores del atributo
            attribute_values = frecuency_table.index

            # Iterar sobre los nombres de los atributos
            for value in attribute_values:
                # Encontrar la clase más frecuente para el valor del atributo
                most_frequent_class = self.training_dataset[self.training_dataset[attribute] == value][self.class_name].mode().iloc[0]
                self.rules = self.rules._append({'Attribute': attribute, 'Value': value, 'Class': most_frequent_class}, ignore_index=True)
        
        # Iterar sobre las reglas
        for index, regla in self.rules.iterrows():
            # Obtener los datos de cada regla
            attribute = regla['Attribute']
            value = regla['Value']
            expected_class = regla['Class']

            # Obtener y asignar la cantidad de instancias cuyo valor del atributo coincide con el valor del atributo de la regla
            instances = len(self.training_dataset[self.training_dataset[attribute] == value])
            self.rules.at[index, 'Instances'] = instances

            # Obtener y asignar la cantidad de instancias que no coinciden con la regla
            errors = len(self.training_dataset[(self.training_dataset[attribute] == value) & (self.training_dataset[self.class_name] != expected_class)])
            self.rules.at[index, 'Errors'] = errors

    def calculateTotalErrors(self) -> None:
        # Iterar sobre los nombres de los atributos 
        for attribute in self.attributes:
            # Obtener las reglas del atributo
            attribute_rule = self.rules[self.rules['Attribute'] == attribute]
            
            # Sumar los errores de las reglas
            errors = attribute_rule['Errors'].sum()

            # Sumar las instancias 
            instances = attribute_rule['Instances'].sum()
            
            # Guardar los errores y las instancias de cada regla
            self.total_errors = self.total_errors._append({'Attribute': attribute, 'Errors': errors, 'Instances': instances}, ignore_index=True)
        
        # Calcular la división Errors/Instances, que representa el error total
        self.total_errors['Result'] = self.total_errors['Errors'] / self.total_errors['Instances']

    def selectModel(self) -> None:
        # Encontrar el índice del atributo cuya regla es la de menor error total
        self.selected_rule_index = self.total_errors['Result'].idxmin()  # Si es más de una regla, retornar la primera

        # Obtener el nombre del atributo cuya regla es la de menor error total
        self.selected_rule_name = self.total_errors.loc[self.selected_rule_index]['Attribute']

        # Obtener la regla
        self.selected_rule = self.rules[self.rules['Attribute'] == self.selected_rule_name]

    
    def evaluate(self, test_dataset) -> None: 
        # Crear una lista para almacenar los resultados
        result = []

        # Iterar sobre cada instancia en el conjunto de datos de prueba
        for instance_index, instance in test_dataset.iterrows():
            # Obtener de la instancia el valor del atributo que se evalúa en la regla y también su clase
            instance_attribute_value = instance[self.selected_rule_name]
            instance_class = instance[self.class_name]

            # Crear una bandera para saber si la regla acertó
            rule_fulfilled = False

            # Obtener la clase esperada según la regla
            expected_class = (self.selected_rule[self.selected_rule['Value'] == instance_attribute_value]['Class']).values[0]

            # Iterar sobre cada condición en la regla seleccionada
            for condition_index, condition in self.selected_rule.iterrows():
                # Obtener de la condicion el valor del atributo y también su clase
                condition_attribute_value = condition['Value']

                # Evaluar si la regla acierta
                if instance_attribute_value == condition_attribute_value and instance_class == expected_class:            
                    rule_fulfilled = True
                    break
    
            result.append([expected_class, rule_fulfilled])
        
        # Convertir la lista de resultados en un DataFrame
        evaluations = pandas.DataFrame(result, columns=['Clase Esperada', 'Acierto'])

        # Crear una copia del set de datos de prueba 
        self.result = test_dataset.copy()

        # Iterar sobre las columnas de evaluations y agregarlas como nuevas columnas en result
        for col in evaluations.columns:
            self.result[col] = evaluations[col].values
        
    def fit(self) -> None:
        # Calcular las tablas de frecuencia
        self.computeFrequencyTables()

        # Calcular los errores
        self.calculateErrors()

        # Calcular el error total de las reglas
        self.calculateTotalErrors()

        # Seleccionar la regla con el menor error total
        self.selectModel()
    
    def getFrequencyTables(self) -> dict:
        return self.frequency_tables
    
    def getRules(self) -> pandas.DataFrame:
        return self.rules

    def getTotalErrors(self) -> pandas.DataFrame:
        return self.total_errors
    
    def getModel(self) -> pandas.DataFrame:
        return self.selected_rule

    def getModelEvaluationResult(self) -> pandas.DataFrame :
        return self.result
    
    def showAdditionalInformation(self) -> None:
        print('Tablas de frecuencia')
        print()
        for key, value in self.frequency_tables.items():
            print(value)
            print()

        print('Reglas')
        print()
        print(self.rules)
        print()

        print('Errores totales')
        print()
        print(self.total_errors)
        print()