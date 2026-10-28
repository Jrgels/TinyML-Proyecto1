
import argparse
import json
import queue
import threading
import time
import uuid

from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt


BASE = "tinyml/har/practica-js-7319"
BROKER = "broker.hivemq.com"

TOPIC_DATA = f"{BASE}/data"
TOPIC_ACK = f"{BASE}/ack"
TOPIC_RESULT = f"{BASE}/result"

DATASET = (
    Path(__file__).resolve().parents[1]
    / "humanActivityRecognitionUsingSmartPhones"
    / "UCI HAR Dataset"
)

CANALES = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

ACTIVIDADES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def seleccionar_ventana(actividad):
    carpeta = DATASET / "test"
    etiquetas = np.loadtxt(
        carpeta / "y_test.txt", dtype=np.int64
    )

    indices = np.flatnonzero(etiquetas == actividad)
    indice = int(np.random.default_rng().choice(indices))

    columnas = []
    for canal in CANALES:
        ruta = (
            carpeta / "Inertial Signals" /
            f"{canal}_test.txt"
        )
        columna = np.loadtxt(ruta, dtype=np.float32)[indice]
        columnas.append(columna)

    ventana = np.stack(columnas, axis=1)
    return indice, ventana


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actividad", type=int, choices=range(1, 7),
        default=1
    )
    args = parser.parse_args()

    indice, ventana = seleccionar_ventana(args.actividad)

    mensajes = queue.Queue()
    conectado = threading.Event()

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(TOPIC_ACK)
            client.subscribe(TOPIC_RESULT)
            conectado.set()

    def on_message(client, userdata, message):
        try:
            contenido = json.loads(message.payload.decode())
            mensajes.put((message.topic, contenido))
        except (ValueError, UnicodeDecodeError):
            pass

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"har-python-{uuid.uuid4().hex[:8]}"
    )
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, 1883, 30)
    client.loop_start()

    try:
        if not conectado.wait(10):
            raise TimeoutError("No se pudo conectar a HiveMQ")

        # Permitir que se activen las suscripciones
        time.sleep(0.5)

        window_id = int(time.time_ns() // 1_000_000) % 2_000_000_000

        print("Ventana:", indice)
        print("Actividad real:", ACTIVIDADES[args.actividad])
        print("Enviando 16 fragmentos...\n")

        resultado = None

        for chunk in range(16):
            bloque = ventana[chunk * 8:(chunk + 1) * 8]

            contenido = {
                "window_id": window_id,
                "chunk": chunk,
                "data": [
                    [float(format(float(x), ".9g")) for x in fila]
                    for fila in bloque
                ]
            }

            payload = json.dumps(
                contenido, separators=(",", ":")
            )

            confirmado = False

            for intento in range(4):
                client.publish(TOPIC_DATA, payload, qos=0)

                limite = time.monotonic() + 5

                while time.monotonic() < limite:
                    restante = limite - time.monotonic()

                    try:
                        topic, respuesta = mensajes.get(
                            timeout=restante
                        )
                    except queue.Empty:
                        break

                    if respuesta.get("window_id") != window_id:
                        continue

                    if topic == TOPIC_RESULT:
                        resultado = respuesta

                    if (
                        topic == TOPIC_ACK
                        and respuesta.get("chunk") == chunk
                    ):
                        confirmado = True
                        break

                if confirmado:
                    break

                print(f"Reintentando fragmento {chunk + 1}")

            if not confirmado:
                raise TimeoutError(
                    f"Sin confirmacion del fragmento {chunk + 1}"
                )

            print(f"Fragmento {chunk + 1}/16 confirmado")

        if resultado is None:
            limite = time.monotonic() + 10

            while time.monotonic() < limite:
                try:
                    topic, respuesta = mensajes.get(
                        timeout=limite - time.monotonic()
                    )
                except queue.Empty:
                    break

                if (
                    topic == TOPIC_RESULT
                    and respuesta.get("window_id") == window_id
                ):
                    resultado = respuesta
                    break

        if resultado is None:
            print("No llego el resultado final")
        else:
            predicha = resultado["activity"]
            real = ACTIVIDADES[args.actividad]

            print("\n--- RESULTADO ---")
            print("Actividad real:", real)
            print("Prediccion ESP32:", predicha)
            print("Correcta:", predicha == real)

    finally:
        client.disconnect()
        client.loop_stop()


if __name__ == "__main__":
    main()
