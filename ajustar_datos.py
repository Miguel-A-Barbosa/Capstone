def calcular_IMC(peso, altura):
    altura = altura / 100
    return round(peso / (altura * altura), 3)

def fumar_gordo(imc, fumador):
    return fumador == 1 and imc > 30

def rango_edad(edad):
    if edad < 29:
        return 'Generación Z'
    elif 29 <= edad < 45:
        return 'Millenials'
    elif 45 <= edad < 61:
        return 'Generación X'
    else:
        return 'Boomers'

def rango_IMC(imc):
    if imc < 18.5:
        return 'Bajo Peso'
    elif 18.5 <= imc < 25:
        return 'Peso saludable'
    elif 25 <= imc < 30:
        return 'Sobrepeso'
    elif 30 <= imc < 35:
        return 'Obesidad Clase 1'
    elif 35 <= imc < 40:
        return 'Obesidad Clase 2'
    else:
        return 'Obesidad Clase 3'

def dummies_categoria(valor, categorias):
    return [1 if valor == cat else 0 for cat in categorias]

def transformador(datos):

    edad = int(datos[0])
    genero = 1 if datos[1] == 'Masculino' else 0
    hijos = int(datos[2])
    peso = float(datos[4])
    altura = float(datos[5])
    fumador = 1 if datos[6] == 'Si' else 0
    imc = calcular_IMC(peso, altura)
    fumador_obeso = fumar_gordo(imc, fumador)

    grupo_edad_val = rango_edad(edad)
    grupo_region_val = datos[3]
    grupo_imc_val = rango_IMC(imc)

    CATEGORIAS_REGION = ['Noroeste', 'Sureste', 'Suroeste']
    CATEGORIAS_GRUPO_EDAD = ['Millenials', 'Generación X', 'Boomers']
    CATEGORIAS_GRUPO_IMC = ['Peso saludable', 'Sobrepeso', 'Obesidad Clase 1', 'Obesidad Clase 2', 'Obesidad Clase 3']

    grupo_edad_dummies = dummies_categoria(grupo_edad_val, CATEGORIAS_GRUPO_EDAD)
    grupo_region_dummies = dummies_categoria(grupo_region_val, CATEGORIAS_REGION)
    grupo_imc_dummies = dummies_categoria(grupo_imc_val, CATEGORIAS_GRUPO_IMC)
    
    datos_transformados = [[
        edad,
        hijos,
        imc,
        fumador,
        fumador_obeso,
        *grupo_edad_dummies,
        genero,
        *grupo_region_dummies,
        *grupo_imc_dummies
    ]]

    return datos_transformados