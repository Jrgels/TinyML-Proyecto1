#!/usr/bin/env python3
"""Evaluación de 6 actividades UCI HAR sobre ESP32 simulado por MQTT.

Ejemplo:
    python mqtt/evaluacion_automatica.py --por-clase 10 --seed 42

Requiere: numpy, pandas, scikit-learn, joblib, paho-mqtt.
Usa los mismos tópicos y formato de mensajes de enviar_ventana.py.
"""

import argparse
import json
import queue
import threading
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from sklearn.metrics import accuracy_score, classification_report, f1_score

BROKER = "broker.hivemq.com"
PUERTO = 1883
BASE = "tinyml/har/practica-js-7319"  # Debe coincidir con sketch.ino
TOPIC_DATA = f"{BASE}/data"
TOPIC_ACK = f"{BASE}/ack"
TOPIC_RESULT = f"{BASE}/result"

CANALES = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]
ACTIVIDADES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}


def buscar_archivos():
    raiz = Path(__file__).resolve().parents[1]
    datos_opciones = [
        raiz / "humanActivityRecognitionUsingSmartPhones" / "UCI HAR Dataset",
        raiz / "UCI HAR Dataset",
    ]
    modelo_opciones = [
        raiz / "humanActivityRecognitionUsingSmartPhones" / "models" / "modelo_har.joblib",
        raiz / "models" / "modelo_har.joblib",
    ]
    datos = next((p for p in datos_opciones if (p / "test" / "y_test.txt").is_file()), None)
    modelo = next((p for p in modelo_opciones if p.is_file()), None)
    if datos is None:
        raise FileNotFoundError("No encuentro 'UCI HAR Dataset/test/y_test.txt': "
                                + "; ".join(map(str, datos_opciones)))
    if modelo is None:
        raise FileNotFoundError("No encuentro 'modelo_har.joblib': "
                                + "; ".join(map(str, modelo_opciones)))
    return raiz, datos, modelo


def cargar_prueba(datos):
    carpeta = datos / "test"
    x = np.stack([
        np.loadtxt(carpeta / "Inertial Signals" / f"{canal}_test.txt", dtype=np.float32)
        for canal in CANALES
    ], axis=-1)
    y = np.loadtxt(carpeta / "y_test.txt", dtype=np.int64)
    sujetos = np.loadtxt(carpeta / "subject_test.txt", dtype=np.int64)
    assert x.shape == (2947, 128, 6), f"Forma inesperada: {x.shape}"
    assert len(y) == len(sujetos) == len(x)
    return x, y, sujetos


def extraer_caracteristicas(ventana):
    """Mismo orden que el notebook: 6 medias, 6 std, 6 min, 6 max, 6 RMS."""
    v = np.asarray(ventana, dtype=np.float32)
    f = np.concatenate([
        np.mean(v, axis=0),
        np.std(v, axis=0),
        np.min(v, axis=0),
        np.max(v, axis=0),
        np.sqrt(np.mean(v ** 2, axis=0)),
    ])
    return f.reshape(1, -1)


class ConexionMQTT:
    """Canal con ACK por fragmento y almacenamiento de resultados por ventana."""

    def __init__(self):
        self.cola = queue.Queue()
        self.con_suscripcion = threading.Event()
        self.resultados_pendientes = {}
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"har-eval-{uuid.uuid4().hex[:10]}",
        )
        self.client.on_connect = self._on_connect
        self.client.on_subscribe = self._on_subscribe
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe([(TOPIC_ACK, 0), (TOPIC_RESULT, 0)])
        else:
            print(f"Error al conectar a HiveMQ: {reason_code}")

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties):
        self.con_suscripcion.set()

    def _on_message(self, client, userdata, mensaje):
        try:
            dato = json.loads(mensaje.payload.decode("utf-8"))
            self.cola.put((mensaje.topic, dato))
        except (UnicodeError, ValueError):
            print("Aviso: se recibió un mensaje MQTT no válido")

    def conectar(self):
        self.client.connect(BROKER, PUERTO, keepalive=30)
        self.client.loop_start()
        if not self.con_suscripcion.wait(15):
            raise TimeoutError("No se confirmó la suscripción MQTT; verifica la red")

    def cerrar(self):
        self.client.disconnect()
        self.client.loop_stop()

    def _evento(self, timeout):
        try:
            topic, dato = self.cola.get(timeout=timeout)
        except queue.Empty:
            return None, None
        if topic == TOPIC_RESULT and "window_id" in dato:
            self.resultados_pendientes[dato["window_id"]] = dato
        return topic, dato

    def esperar_ack(self, window_id, chunk, timeout):
        limite = time.monotonic() + timeout
        while time.monotonic() < limite:
            topic, dato = self._evento(max(0.001, limite - time.monotonic()))
            if (topic == TOPIC_ACK and dato.get("window_id") == window_id
                    and dato.get("chunk") == chunk):
                return True
        return False

    def esperar_resultado(self, window_id, ultimo_payload, timeout, reintentos):
        for intento in range(reintentos + 1):
            if window_id in self.resultados_pendientes:
                return self.resultados_pendientes.pop(window_id)
            if intento:
                # El firmware anterior vuelve a publicar el resultado al recibir
                # de nuevo el fragmento 15 de una ventana ya terminada.
                self.client.publish(TOPIC_DATA, ultimo_payload, qos=0)
            limite = time.monotonic() + timeout
            while time.monotonic() < limite:
                if window_id in self.resultados_pendientes:
                    return self.resultados_pendientes.pop(window_id)
                self._evento(max(0.001, limite - time.monotonic()))
        raise TimeoutError(f"Sin resultado para la ventana {window_id}")

    def enviar_ventana(self, window_id, ventana, timeout, reintentos):
        inicio = time.perf_counter()
        ultimo_payload = None
        for chunk in range(16):
            bloque = ventana[chunk * 8:(chunk + 1) * 8]
            mensaje = {
                "window_id": window_id,
                "chunk": chunk,
                "data": [
                    [float(format(float(valor), ".9g")) for valor in fila]
                    for fila in bloque
                ],
            }
            payload = json.dumps(mensaje, separators=(",", ":"))
            ultimo_payload = payload
            confirmado = False
            for intento in range(reintentos + 1):
                info = self.client.publish(TOPIC_DATA, payload, qos=0)
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    raise ConnectionError(f"MQTT publish falló: {info.rc}")
                if self.esperar_ack(window_id, chunk, timeout):
                    confirmado = True
                    break
                print(f"  Reintento {intento + 1}: fragmento {chunk + 1}/16")
            if not confirmado:
                raise TimeoutError(f"Sin ACK del fragmento {chunk + 1}/16")

        resultado = self.esperar_resultado(
            window_id, ultimo_payload, timeout=timeout, reintentos=reintentos
        )
        tiempo_ms = (time.perf_counter() - inicio) * 1000
        return resultado, tiempo_ms


def main():
    parser = argparse.ArgumentParser(description="Evaluación TinyML HAR por MQTT")
    parser.add_argument("--por-clase", type=int, default=10,
                        help="Ventanas aleatorias por actividad (por defecto: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducir la selección")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Segundos de espera por ACK y por resultado")
    parser.add_argument("--reintentos", type=int, default=3)
    args = parser.parse_args()
    if args.por_clase < 1 or args.timeout <= 0 or args.reintentos < 0:
        parser.error("--por-clase y --timeout deben ser positivos; reintentos >= 0")

    raiz, datos, ruta_modelo = buscar_archivos()
    x_test, y_test, sujetos = cargar_prueba(datos)
    artefacto = joblib.load(ruta_modelo)
    modelo = artefacto["modelo"]
    nombres_esperados = [
        f"{estadistica}_{canal}"
        for estadistica in ["mean", "std", "min", "max", "rms"]
        for canal in CANALES
    ]
    assert list(artefacto["features"]) == nombres_esperados, (
        "El orden de las características del modelo no coincide con el script"
    )

    rng = np.random.default_rng(args.seed)
    seleccion = []
    for clase in ACTIVIDADES:
        opciones = np.flatnonzero(y_test == clase)
        if args.por_clase > len(opciones):
            parser.error(f"La clase {clase} tiene solo {len(opciones)} ventanas")
        seleccion.extend(rng.choice(opciones, args.por_clase, replace=False).tolist())
    rng.shuffle(seleccion)

    carpeta_resultados = raiz / "resultados"
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    salida = carpeta_resultados / f"evaluacion_har_60_seed{args.seed}.csv"
    if args.por_clase != 10:
        salida = carpeta_resultados / f"evaluacion_har_{len(seleccion)}_seed{args.seed}.csv"

    conexion = ConexionMQTT()
    registros = []
    base_id = int(time.time()) % 1_000_000_000

    print(f"Dataset: {datos}")
    print(f"Modelo: {ruta_modelo}")
    print(f"Pruebas: {len(seleccion)} ({args.por_clase} por clase)")
    print("Inicia primero la simulación Wokwi y verifica 'Conectado a HiveMQ'.\n")

    try:
        conexion.conectar()
        for numero, indice in enumerate(seleccion, start=1):
            ventana = x_test[indice]
            real_id = int(y_test[indice])
            pred_python = int(modelo.predict(extraer_caracteristicas(ventana))[0])
            window_id = base_id + numero
            fila = {
                "numero": numero,
                "indice_test": indice,
                "sujeto": int(sujetos[indice]),
                "real_id": real_id,
                "actividad_real": ACTIVIDADES[real_id],
                "python_id": pred_python,
                "prediccion_python": ACTIVIDADES[pred_python],
                "esp32_id": None,
                "prediccion_esp32": None,
                "coincide_python_esp32": None,
                "acierta_esp32": None,
                "tiempo_total_ms": None,
                "features_us": None,
                "inference_us": None,
                "estado": "ERROR",
            }
            try:
                resultado, ms = conexion.enviar_ventana(
                    window_id, ventana, args.timeout, args.reintentos
                )
                clase_esp = int(resultado["class_id"])
                if clase_esp not in ACTIVIDADES:
                    raise ValueError(f"Clase desconocida: {clase_esp}")
                fila.update({
                    "esp32_id": clase_esp,
                    "prediccion_esp32": ACTIVIDADES[clase_esp],
                    "coincide_python_esp32": int(pred_python == clase_esp),
                    "acierta_esp32": int(real_id == clase_esp),
                    "tiempo_total_ms": round(ms, 2),
                    "features_us": resultado.get("features_us"),
                    "inference_us": resultado.get("inference_us"),
                    "estado": "OK",
                })
                print(f"[{numero:02d}/{len(seleccion)}] "
                      f"Real={ACTIVIDADES[real_id]:19s} "
                      f"Python={ACTIVIDADES[pred_python]:19s} "
                      f"ESP32={ACTIVIDADES[clase_esp]:19s} "
                      f"{'OK' if real_id == clase_esp else 'ERROR_CLASIFICACION'} "
                      f"{ms:.0f} ms")
            except (TimeoutError, ConnectionError, ValueError, KeyError) as exc:
                fila["estado"] = f"ERROR: {exc}"
                print(f"[{numero:02d}/{len(seleccion)}] {fila['estado']}")

            registros.append(fila)
            # Persistir después de cada prueba evita perder resultados parciales.
            pd.DataFrame(registros).to_csv(salida, index=False)
    finally:
        conexion.cerrar()

    df = pd.DataFrame(registros)
    validos = df.loc[df["estado"] == "OK"].copy()
    print(f"\nCSV guardado en: {salida}")
    print(f"Ventanas con resultado: {len(validos)}/{len(seleccion)}")
    if validos.empty:
        print("No hay predicciones recibidas para calcular métricas.")
        return

    real = validos["real_id"].astype(int)
    esp32 = validos["esp32_id"].astype(int)
    acierto_modelo_embebido = accuracy_score(real, esp32)
    f1 = f1_score(real, esp32, labels=list(ACTIVIDADES), average="macro",
                  zero_division=0)
    equivalencia = validos["coincide_python_esp32"].mean()
    print(f"Accuracy ESP32: {acierto_modelo_embebido:.2%}")
    print(f"F1-macro ESP32: {f1:.4f}")
    print(f"Coincidencia Python/ESP32: {equivalencia:.2%}")
    print(f"Tiempo total MQTT + procesamiento (mediana): "
          f"{validos['tiempo_total_ms'].median():.1f} ms")
    if validos["inference_us"].notna().any():
        print(f"Inferencia en ESP32 (mediana): "
              f"{pd.to_numeric(validos['inference_us']).median():.1f} us")
    print("\nReporte de clasificación ESP32:")
    print(classification_report(real, esp32,
                                labels=list(ACTIVIDADES),
                                target_names=list(ACTIVIDADES.values()),
                                zero_division=0))
    print("\nMatriz de confusión (filas=real, columnas=predicción):")
    print(pd.crosstab(pd.Categorical(real, categories=list(ACTIVIDADES)),
                      pd.Categorical(esp32, categories=list(ACTIVIDADES)),
                      rownames=["Real"], colnames=["ESP32"], dropna=False))
    if len(validos) != len(seleccion):
        print("\nAtención: métricas calculadas solo con ventanas completadas; "
              "revisar los errores en el CSV.")


if __name__ == "__main__":
    main()
