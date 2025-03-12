import cv2
import numpy as np
import pytesseract
import re
import time
from datetime import datetime
from pymongo import MongoClient

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def conectar_mongodb():
    """Conecta a la base de datos MongoDB y devuelve la instancia de la base de datos."""
    try:
        connection_string = "mongodb+srv://lgaxiola117:H4QJA8wJDY1BPCXo@quickpass.3zpme.mongodb.net/?retryWrites=true&w=majority"
        client = MongoClient(connection_string)
        db = client['users']
        client.admin.command('ping')  # Verificar la conexión
        print("Conexión exitosa a MongoDB Atlas")
        return db
    except Exception as e:
        print(f"Error al conectar con MongoDB Atlas: {e}")
        return None

def inicializar_base_datos():
    """Inicializa la base de datos con placas de ejemplo si está vacía."""
    db = conectar_mongodb()
    if db is None:
        return False
    
    return True

def verificar_placa(numero_placa):
    """Verifica si una placa está autorizada y registra el acceso."""
    db = conectar_mongodb()
    if db is None:
        return False, None
    
    placa_info = db.placas.find_one({"numero_placa": numero_placa})
    
    if placa_info and placa_info.get("autorizado", True):
        acceso_info = {
            "numero_placa": numero_placa,
            "fecha_acceso": datetime.now(),
            "acceso_permitido": True
        }
        db.accesos.insert_one(acceso_info)
        return True, placa_info.get("propietario")
    
    return False, None

def abrir_caseta():
    """Simula la apertura de la caseta."""
    print("¡Caseta abierta!")
    time.sleep(2)  # Simulación del tiempo de apertura
    print("Caseta cerrada")

def preprocesar_imagen(imagen): 
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro bilateral para reducir ruido pero mantener bordes
    gris = cv2.bilateralFilter(gris, d=15, sigmaColor=75, sigmaSpace=75)

    # Mejorar contraste usando CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gris = clahe.apply(gris)

    # Aplicar umbralización adaptativa gaussiana
    umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 10)

    # Refinar con Otsu
    _, umbral = cv2.threshold(umbral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Aplicar operaciones morfológicas para eliminar ruido
    kernel = np.ones((3, 3), np.uint8)
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel, iterations=2)
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel, iterations=1)

    # Dilatar para hacer las letras más gruesas y mejorar OCR
    umbral = cv2.dilate(umbral, np.ones((2, 2), np.uint8), iterations=1)

    return gris, umbral

def detectar_placa_alternativo(imagen):
    """Detecta la región de la placa en la imagen."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.bilateralFilter(gris, 11, 17, 17)
    gris = cv2.equalizeHist(gris)
    umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = [(contorno, cv2.contourArea(contorno), float(cv2.boundingRect(contorno)[2]) / cv2.boundingRect(contorno)[3]) for contorno in contornos if cv2.contourArea(contorno) > 1000]
    
    # Filtrar candidatos por relación de aspecto
    candidatos = [c for c in candidatos if 1.5 < c[2] < 3.5]
    candidatos.sort(key=lambda x: x[1], reverse=True)
    
    if not candidatos:
        return None, umbral
    
    for contorno, _, _ in candidatos[:5]:
        peri = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * peri, True)
        if len(aprox) >= 4 and len(aprox) <= 8:
            return aprox, umbral
    
    return candidatos[0][0], umbral
    
def extraer_texto_de_umbral(imagen_umbral):
    """Extrae el texto de la placa a partir de la imagen umbralizada."""
    config = '--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
    texto = pytesseract.image_to_string(imagen_umbral, config=config).strip()
    
    return texto  # Retorna el texto tal como lo encontró


def modo_prueba_imagen_directo(ruta_imagen):
    """Modo de prueba con imagen estática."""
    print("Iniciando modo de prueba con imagen estática...")
    
    if not inicializar_base_datos():
        print("No se pudo inicializar la base de datos. El programa terminará.")
        return
    
    ruta_imagen = ruta_imagen.strip('"').strip("'")
    print(f"Intentando cargar la imagen desde: {ruta_imagen}")
    
    frame = cv2.imread(ruta_imagen)
    if frame is None:
        print(f"No se pudo cargar la imagen {ruta_imagen}. Verifica la ruta.")
        return
    
    cv2.imshow("Imagen original", frame)
    cv2.waitKey(0)
    
    ubicacion_placa, umbral = detectar_placa_alternativo(frame)
    
    if ubicacion_placa is None:
        print("No se detectó placa con el método de contornos. Intentando OCR directo en toda la imagen...")
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        cv2.imshow("Umbralización para OCR directo", umbral)
        cv2.waitKey(0)
        
        texto = pytesseract.image_to_string(umbral, config='--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-').strip()
        patron_placa = r'[A-Z]{3}-\d{3}-[A-Z]'
        coincidencias = re.findall(patron_placa, texto)
        
        if coincidencias:
            texto_placa = coincidencias[0]
            print(f"Texto detectado directamente: {texto_placa}")
            autorizado, propietario = verificar_placa(texto_placa)
            if autorizado:
                print(f"Acceso permitido. Propietario: {propietario}")
                abrir_caseta()
            else:
                print("Acceso denegado. Placa no autorizada.")
            resultado = frame.copy()
            cv2.putText(resultado, texto_placa, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Resultado final (OCR directo)", resultado)
            cv2.waitKey(0)
            return
        else:
            print("No se pudo detectar un patrón de placa válido con OCR directo.")
            texto_limpio = re.sub(r'[^\w-]', '', texto)
            print(f"Texto detectado (sin formato): {texto_limpio}")
            if len(texto_limpio) >= 7:
                texto_formateado = f"{texto_limpio[:3]}-{texto_limpio[3:6]}-{texto_limpio[6]}"
                print(f"Texto formateado manualmente: {texto_formateado}")
                autorizado, propietario = verificar_placa(texto_formateado)
                if autorizado:
                    print(f"Acceso permitido. Propietario: {propietario}")
                    abrir_caseta()
                else:
                    print("Acceso denegado. Placa no autorizada.")
                resultado = frame.copy()
                cv2.putText(resultado, texto_formateado, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Resultado final (formato manual)", resultado)
                cv2.waitKey(0)
            return
    
    print("Placa detectada, extrayendo texto...")
    frame_con_placa = frame.copy()
    cv2.drawContours(frame_con_placa, [ubicacion_placa], -1, (0, 255, 0), 3)
    cv2.imshow("Placa detectada", frame_con_placa)
    cv2.waitKey(0)
    
    x, y, w, h = cv2.boundingRect(ubicacion_placa)
    placa_recortada = frame[y:y+h, x:x+w]
    
    cv2.imshow("Placa recortada", placa_recortada)
    cv2.waitKey(0)
    
    placa_gris = cv2.cvtColor(placa_recortada, cv2.COLOR_BGR2GRAY)
    placa_gris = cv2.equalizeHist(placa_gris)
    _, placa_umbral = cv2.threshold(placa_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("Placa Gris",placa_gris)
    cv2.imshow("Placa umbralizada", placa_umbral)
    cv2.waitKey(0)
    
    texto_placa = pytesseract.image_to_string(placa_umbral, config='--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-').strip()
    texto_placa1 = pytesseract.image_to_string(placa_gris, config='--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-').strip()
    texto_placa2 = pytesseract.image_to_string(placa_recortada, config='--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-').strip()

    patron_placa = r'[A-Z]{3}-\d{3}-[A-Z]'
    coincidencias = re.findall(patron_placa,texto_placa2)
    if coincidencias:
        texto_placa = coincidencias[0]
    else:   
        texto_limpio = re.sub(r'[^\w]', '', texto_placa)
        if len(texto_limpio) >= 7:
            texto_placa = f"{texto_limpio[:3]}-{texto_limpio[3:6]}-{texto_limpio[6]}"       
    
    print(f"Texto detectado: {texto_placa}")
    autorizado, propietario = verificar_placa(texto_placa)
    
    if autorizado:
        print(f"Acceso permitido. Propietario: {propietario}")
        abrir_caseta()
    else:
        print("Acceso denegado. Placa no autorizada.")
    
    cv2.putText(frame_con_placa, texto_placa, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame_con_placa, texto_placa1, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_con_placa, texto_placa2, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    cv2.imshow("Resultado final", frame_con_placa)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def modo_camara_vivo():
    """Modo de cámara en vivo para detección de placas."""
    print("Iniciando sistema de reconocimiento de placas con cámara en vivo...")
    
    if not inicializar_base_datos():
        print("No se pudo inicializar la base de datos. El programa terminará.")
        return
    
    camara = cv2.VideoCapture(0)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    if not camara.isOpened():
        print("Error: No se pudo abrir la cámara. Verifica la conexión.")
        return
    
    ultima_placa_detectada = ""
    tiempo_ultima_deteccion = 0
    
    print("Sistema listo. Presiona 'q' para salir.")
    
    while True:
        ret, frame = camara.read()
        if not ret:
            print("Error al capturar imagen. Verificando cámara...")
            time.sleep(1)
            continue
        
        gris, bordes = preprocesar_imagen(frame)
        ubicacion_placa = detectar_placa_alternativo(frame)
        
        if ubicacion_placa is not None:
            cv2.drawContours(frame, [ubicacion_placa], -1, (0, 255, 0), 3)
            texto_placa = extraer_texto_de_umbral(frame, ubicacion_placa)
            
            tiempo_actual = time.time()
            if texto_placa and (texto_placa != ultima_placa_detectada or tiempo_actual - tiempo_ultima_deteccion > 5):
                print(f"Placa detectada: {texto_placa}")
                autorizado, propietario = verificar_placa(texto_placa)
                
                if autorizado:
                    print(f"Acceso permitido. Propietario: {propietario}")
                    abrir_caseta()
                else:
                    print("Acceso denegado. Placa no autorizada.")
                
                ultima_placa_detectada = texto_placa
                tiempo_ultima_deteccion = tiempo_actual
            
            if texto_placa:
                cv2.putText(frame, texto_placa, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Detección de Placas", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camara.release()
    cv2.destroyAllWindows()

def main():
    """Función principal que inicia el sistema de reconocimiento de placas."""
    print("Sistema de Reconocimiento de Placas para Caseta Inteligente")
    print("=========================================================")
    print("Modo prueba con imagen estática")
    
    opcion = input("Seleccione una opción (1/2): ")
    
    if opcion == "1":
        ruta_imagen = input("Ingrese la ruta de la imagen de prueba: ")
        modo_prueba_imagen_directo(ruta_imagen)
    elif opcion == "2":
        modo_camara_vivo()
    else:
        print("Opción no válida")

if __name__ == "__main__":
    main()