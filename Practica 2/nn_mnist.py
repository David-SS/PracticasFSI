import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]

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


# SE OBTIENE EL CONJUNTO DE DATOS DEL MNIST
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# ESTABLECEMOS LOS TRES CONJUNTOS (DATOS Y ETIQUETAS)
trainset_x, trainset_y = train_set
validset_x, validset_y = valid_set
testset_x, testset_y = test_set

# CONVERTIMOS LAS ETIQUETAS A FORMATO ONEHOT
trainset_y = one_hot(trainset_y.astype(int), 10)
validset_y = one_hot(validset_y.astype(int), 10)
testset_y = one_hot(testset_y.astype(int), 10)


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

#plt.imshow(trainset_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print trainset_y[57]

# -----------------------------------------------------------------------------


# CONJUNTO DE PLACEHOLDERS Y VARIABLES
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels


# CAPA DE ENTRADA (5 NEURONAS): MATRIZ DE NUMEROS ALEATORIOS QUE CONTIENEN PESOS (784 POR NEURONA) Y BIAS (1 POR NEURONA)
We = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
be = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

# POSIBLES CAPAS OCULTAS CON LA MISMA "DINAMICA":
W1 = tf.Variable(np.float32(np.random.rand(10, 100)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1)
W2 = tf.Variable(np.float32(np.random.rand(100, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

# CAPA DE SALIDA (10 NEURONAS): MATRIZ DE NUMEROS ALEATORIOS QUE CONTIENEN PESOS (5 POR NEURONA) Y BIAS (1 POR NEURONA)
Wf = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
bf = tf.Variable(np.float32(np.random.rand(10)) * 0.1)


# PARA LAS SUCESIVAS CAPAS HACEMOS: (ENTRADAS * PESOS) + BIAS
h1 = tf.nn.sigmoid(tf.matmul(x, We) + be)
# SI USAMOS LAS DOS CAPAS OCULTAS:
h2 = tf.matmul(h1, W1) + b1
h3 = tf.matmul(h2, W2) + b2
y = tf.nn.softmax(tf.matmul(h3, Wf) + bf)
# SI USAMOS SOLO CAPA DE ENTRADA Y DE SALIDA:
#y = tf.nn.softmax(tf.matmul(h1, Wf) + bf)


# FUNCION DE ERROR (EN FUNCION DE LAS VARIABLES DEL MODELO DEFINIDAS ANTES... PESOS,ETC...)
# MUESTRA EL ERROR ACTUAL: (RESULTADO ESPERADO - OBTENIDO) * (RESULTADO ESPERADO - OBTENIDO)
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

# TAMANO DE LOTES, ERRORES ACTUAL Y ANTERIOR, CONTADOR DE ITERACIONES
batch_size = 100
minEpocas = 0 # EPOCAS MINIMAS QUE DESEAMOS REALIZAR
contador = 0;
errorActual = sess.run(loss, feed_dict={x: validset_x, y_: validset_y});
errorAnterior = errorActual;

while (errorAnterior >= errorActual or minEpocas >= 0):
    # ITERACIONES PARA RECORRER TODOS LOS ELEMENTOS DE 1000 EN 1000 REALIZANDO EL ENTRENAMIENTO
    for jj in xrange(len(trainset_x) / batch_size):
        # CONJUNTOS DE 35 MUESTRAS: VALORES DE MUESTRA (XS) Y TIPO DE MUESTRA (YS)
        batch_xs = trainset_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = trainset_y[jj * batch_size: jj * batch_size + batch_size]
        # SE CALCULAN LAS DERIVADAS PARCIALES Y SE REALIZA EL PROCESO DE ENTRENAMIENTO
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # ERROR CALCULADO EN FUNCION DEL CONJUNTO DE VALIDACION
    errorAnterior = errorActual
    errorActual = sess.run(loss, feed_dict={x: validset_x, y_: validset_y})
    minEpocas = minEpocas - 1
    contador = contador + 1;
    print "Epoch #:", contador, "Error: ", errorActual

    # SI QUISIERAMOS VER LOS RESULTADOS
    #result = sess.run(y, feed_dict={x: validset_x})
    #for b, r in zip(validset_y, result):
    #    print b, "-->", r
    print "----------------------------------------------------------------------------------"


# PONEMOS EN MARCHA EL CONJUNTO DE TEST
indice = -1;
error = 0;
maxMostrar = 0; # IMAGENES ERRONEAS A MOSTRAR
result = sess.run(y, feed_dict={x: testset_x})

print "PRUEBAS CONJUNTO TEST"
for b, r in zip(testset_y, result):
    #print b, "-->", r
    indice = indice + 1
    if (np.argmax(b) != np.argmax(r)):
        error = error +1
        if (error <= maxMostrar):
            plt.imshow(trainset_x[indice].reshape((28, 28)), cmap=cm.Greys_r)
            plt.show()  # Let's see a sample
            print np.argmax(trainset_y[indice])


print ""
print "Numero de errores en el conjunto test: ", error
print "Porcentaje de error en el conjunto test: ", '%.2f'%(float(error*100)/len(testset_x)), "%"