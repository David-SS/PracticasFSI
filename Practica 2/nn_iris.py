import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
# DADO UN NUMERO X, Y UN TAMANO N, TE HACE LA CONVERSION DE X=1 A [1,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

# IMPORTA DATOS A UNA MATRIZ
data = np.genfromtxt('iris.data', delimiter=",")
# DESORDENAMOS LOS DATOS PARA QUE ENTREN ALTERNADOS Y LA RED APRENDA A DIFERENCIAR
np.random.shuffle(data)
# OBTENEMOS LAS CUATRO PRIMERAS COLUMNAS (DATOS ESPECIES)
x_data = data[:, 0:4].astype('f4')
# OBTENEMOS LA ULTIMA COLUMNA (TIPO DE ESPECIE), PERO YA CONVERTIDA A FORMATO BINARIO
y_data = one_hot(data[:, 4].astype(int), 3)

# SEPARAMOS EN CONJUNTOS DE ENTRENAMIENTO, VALIDACION Y TEST
x_data_trainset = x_data[0:105];
y_data_trainset = y_data[0:105];

x_data_validset = x_data[105:128];
y_data_validset = y_data[105:128];

x_data_testset = x_data[128:150];
y_data_testset = y_data[128:150];


print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print


# CONJUNTO DE PLACEHOLDERS Y VARIABLES (QUE LUEGO SERAN METODOS)

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# MATRIZ DE NUMEROS ALEATORIOS QUE REPRESENTAN: PESOS (4 FILAS 5 COLUMNAS) Y BIAS (VECTOR DE 5 ELEMENTOS)
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

# MATRIZ DE NUMEROS ALEATORIOS QUE REPRESENTAN: PESOS (5 FILAS 3 COLUMNAS) Y BIAS (VECTOR DE 3 ELEMENTOS)
W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)


# H = MULTIPLICACION DE MATRICES (VECTOR_ENTRADAS * MATRIZ_PESOS_NEURONACOLUMNA) + B1
# Y = LO MISMO QUE EN H PERO AHORA LAS ENTRADAS SON LAS SALIDAS DE H Y LA MATRIZ DE PESOS ES OTRA (TAMBIEN EL BIAS ES OTRO)
# FUNCIONES APLICADAS A LA SALIDA: SIGMOIDE Y SOFTMAX ; DIFERENCIA: SOFTMAX ASEGURA QUE LA SUMA TOTAL DA UNO (PROBABILIDAD)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

# NUESTRA FUNCION DE ERROR QUE MUESTRA EL ERROR ACTUAL (SUMA DE LA RESTA ESA AL CUADRADO)
# POR DEFECTO ESTA EN FUNCION DE TODAS LAS VARIABLES DEL MODELO (DEFINIDAS ANTES)
loss = tf.reduce_sum(tf.square(y_ - y))

# COGE DEL PAQUETE DE ENTRENAMIENTO LA FUNCION PARA OPTIMIZAR POR EL GRADIENTE (IR DISMINUYENDO EL ERROR)
# opt = f.train.GradientDescentOptimizer(0.01)
# MINIMIZA LA FUNCION DE ERROR
# mytrain = opt.minimize(loss)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

# INICIALIZAMOS LAS VARIABLES
init = tf.initialize_all_variables()

# AQUI CREAMOS LA SESION Y EJECUTAMOS EL PRIMER PROGRAMITA (LLEVA LAS VARIABLES A LA GPU)
# AHORA SI TENEMOS NUESTRO PROGRAMA EN GPU, ESPERANDO A SER "EJECUTADO"
sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

# NUESTRO LOTE PARA EL CONJUNTO DE ENTRENAMIENTO SERA DE 35 (PARA COGER TODOS LOS ELEMENTOS, YA QUE 35 ES DIVISOR DE 105)
batch_size = 35

# RECORREMOS NUESTRO CONJUNTO DE MUESTRAS
# EPOCAS (NUMERO DE "ENTRENAMIENTOS")
for epoch in xrange(100):
    # ITERACIONES PARA RECORRER TODOS LOS ELEMENTOS DE 35 EN 35 REALIZANDO EL ENTRENAMIENTO
    for jj in xrange(len(x_data_trainset) / batch_size):
        # CONJUNTOS DE 35 MUESTRAS: VALORES DE MUESTRA (XS) Y TIPO DE MUESTRA (YS)
        batch_xs = x_data_trainset[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_trainset[jj * batch_size: jj * batch_size + batch_size]
        # SE CALCULAN LAS DERIVADAS PARCIALES Y SE REALIZA EL PROCESO DE ENTRENAMIENTO
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # ERROR MOSTRADO (EL ERROR GLOBAL) IRA BAJANDO; LO CALCULAMOS EN FUNCION DEL CONJUNTO DE VALIDACION
    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: x_data_validset, y_: y_data_validset})
    # Y = MODELO DE LA RED (SIN ALTERACIONES) --> PARA MOSTRAR LOS VALORES REALES FRENTE A LOS "DESEADOS"
    result = sess.run(y, feed_dict={x: x_data_validset})
    for b, r in zip(y_data_validset, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"


# YA CON EL ENTRENAMIENTO REALIZADO Y CON LOS RESULTADOS DEL CONJUNTO DE VALIDACION, PASAMOS AL TEST

error = 0;
result = sess.run(y, feed_dict={x: x_data_testset})
print "PRUEBAS CONJUNTO TEST"
for b, r in zip(y_data_testset, result):
    print b, "-->", r
    if (np.argmax(b) != np.argmax(r)):
        error = error +1
print ""
print "Numero de errores en el conjunto test: ", error
print "Porcentaje de error en el conjunto test: ", '%.2f'%(float(error*100)/len(x_data_testset)), "%"