import cv2
import datetime

# Cargar el clasificador de camiones
truck_cascade = cv2.CascadeClassifier('/home/clone/Desktop/clone/Vehicule_Detector/XMLFiles/truck.xml')

# Inicializar la captura de video
vc = cv2.VideoCapture(0)

if not vc.isOpened():
    print("Error al abrir la cámara")
    exit()

# Crear una ventana de visualización en pantalla completa
cv2.namedWindow("preview", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Fuente de texto para mostrar información
font = cv2.FONT_HERSHEY_SIMPLEX

# Diccionario para mantener un seguimiento de los camiones detectados
trucks_detected = 0

# Tiempo anterior para el seguimiento
previous_time = datetime.datetime.now()

while True:
    rval, frame = vc.read()
    if not rval:
        print("No se puede capturar el fotograma")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reducir el valor de scaleFactor puede ayudar a reducir falsos positivos
    scaleFactor = 1.1

    # Realizar la detección de camiones
    trucks = truck_cascade.detectMultiScale(gray, scaleFactor, 5, minSize=(70, 70))

    current_time = datetime.datetime.now()

    # Realizar seguimiento de camiones detectados
    if (current_time - previous_time).seconds >= 10:
        previous_time = current_time
        trucks_detected += len(trucks)

    # Mostrar información sobre camiones detectados
    cv2.putText(frame, f'Trucks: {trucks_detected}',
                (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Dibujar rectángulos alrededor de los camiones detectados
    for (x, y, w, h) in trucks:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Truck', (x, y - 10), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostrar la hora actual
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Mostrar el fotograma
    cv2.imshow("preview", frame)

    key = cv2.waitKey(20)
    if key == 27:
        break

# Liberar la captura de video y cerrar todas las ventanas
vc.release()
cv2.destroyAllWindows()
