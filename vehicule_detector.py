import datetime
import cv2

car_cascade = cv2.CascadeClassifier('/home/clone/Desktop/clone/Vehicule_Detector/XMLFiles/car_classifer.xml')
motorcycle_cascade = cv2.CascadeClassifier('/home/clone/Desktop/clone/Vehicule_Detector/XMLFiles/motorcycle.xml')
truck_cascade = cv2.CascadeClassifier('/home/clone/Desktop/clone/Vehicule_Detector/XMLFiles/truck.xml')

vc = cv2.VideoCapture(0)

if not vc.isOpened():
    print("Error al abrir la cámara")
    exit()

cv2.namedWindow("preview", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

font = cv2.FONT_HERSHEY_SIMPLEX

vehicles_detected = {'car': 0, 'motorcycle': 0, 'truck': 0}
previous_time = datetime.datetime.now()

while True:
    rval, frame = vc.read()
    if not rval:
        print("No se puede capturar el fotograma")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    scaleFactor = 1.2  # Ajustar este valor según la precisión de la detección
    minNeighbors = 5    # Ajustar este valor según la precisión de la detección

    cars = car_cascade.detectMultiScale(gray, scaleFactor, minNeighbors, minSize=(50, 50))
    motorcycles = motorcycle_cascade.detectMultiScale(gray, scaleFactor, minNeighbors, minSize=(30, 30))
    trucks = truck_cascade.detectMultiScale(gray, scaleFactor, minNeighbors, minSize=(70, 70))

    current_time = datetime.datetime.now()
    if (current_time - previous_time).seconds >= 10:
        previous_time = current_time
        vehicles_detected['car'] = len(cars)
        vehicles_detected['motorcycle'] = len(motorcycles)
        vehicles_detected['truck'] = len(trucks)

    cv2.putText(frame, f'Cars: {vehicles_detected["car"]}, Motorcycles: {vehicles_detected["motorcycle"]}, '
                       f'Trucks: {vehicles_detected["truck"]}',
                (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Car', (x, y - 10), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in motorcycles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Motorcycle', (x, y - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in trucks:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Truck', (x, y - 10), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("preview", frame)

    key = cv2.waitKey(20)
    if key == 27:
        break

total_vehicles = sum(vehicles_detected.values())
print("Total de vehículos detectados:", total_vehicles)

vc.release()
cv2.destroyAllWindows()
