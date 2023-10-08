import pandas
from NaiveBayes import NaiveBayes

def main():
    # Archivo con el set de datos
    csv = '../golf-dataset-categorical.csv'

    # Leer el archivo CSV y almacenar los datos en un DataFrame
    dataset = pandas.read_csv(csv)

    # Eliminar los espacios en blanco del DataFrame
    dataset = dataset.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Obtener una lista con los nombres de los atributos (omitir la última columna)
    attributes = dataset.columns[:-1]

    # Obtener el nombre de la clase (última columna)
    class_name = dataset.columns[-1]

    # Definir el tamaño del set de datos de entrenamiento
    training_percentage = 0.7
    number_of_instances = round(len(dataset) * training_percentage)
    
    # Definir el set de datos de entrenamiento, seleccionando aleatoriamente las instancias
    training_dataset = dataset.sample(number_of_instances)

    # Definir el set de datos de prueba
    test_dataset = dataset.drop(training_dataset.index)

    # Crear una instancia de la clase NaiveBayes
    naiveBayes = NaiveBayes(training_dataset, attributes, class_name)

    naiveBayes.fit()

    frequency_tables = naiveBayes.getFrequencyTables()
    likelihood_tables = naiveBayes.getLikelihoodTables()

    print()
    print('Conjunto de entrenamiento')
    print()
    print(training_dataset)

    print()
    print('Conjunto de prueba')
    print()
    print(test_dataset)
   
    print()
    print('Tablas de frecuencia')
    print()
    for i, value in frequency_tables.items():
        print(value)
        print()

    print('Tablas de verosimilitud')
    print()
    for i, value in likelihood_tables.items():
        print(value)
        print()
    
    result = naiveBayes.evaluate(test_dataset)
    confusion_matrix = naiveBayes.computeConfusionMatrix(result)

    print('Evaluación del modelo')
    print()
    print(result)

    print()
    print('Tabla de confusión')
    print()
    print(confusion_matrix)

    print()

# Ejecutar el main
if __name__ == '__main__':
    main()   