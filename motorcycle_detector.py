import datetime
import cv2

motorcycle_cascade = cv2.CascadeClassifier('/home/clone/Desktop/clone/Vehicule_Detector/XMLFiles/motorcycle.xml')

vc = cv2.VideoCapture(0)

if not vc.isOpened():
    print("Error al abrir la cámara")
    exit()

cv2.namedWindow("preview", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

font = cv2.FONT_HERSHEY_SIMPLEX

vehicles_detected = 0
previous_time = datetime.datetime.now()

while True:
    rval, frame = vc.read()
    if not rval:
        print("No se puede capturar el fotograma")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    scaleFactor = 1.2  
    minNeighbors = 5    

    motorcycles = motorcycle_cascade.detectMultiScale(gray, scaleFactor, minNeighbors, minSize=(30, 30))

    current_time = datetime.datetime.now()
    if (current_time - previous_time).seconds >= 10:
        previous_time = current_time
        vehicles_detected = len(motorcycles)

    cv2.putText(frame, f'Motocicletas detectadas: {vehicles_detected}',
                (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in motorcycles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Motorcycle', (x, y - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, 70), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("preview", frame)

    key = cv2.waitKey(20)
    if key == 27:
        break

print("Total de motocicletas detectadas:", vehicles_detected)

vc.release()
cv2.destroyAllWindows()
