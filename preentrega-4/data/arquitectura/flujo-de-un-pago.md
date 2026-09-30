# Flujo de un pago

## Alta de la orden

El comercio llama a `POST /v1/payments` con una clave de idempotencia. El gateway valida la firma HMAC y reenvía a `pagos-api`. Si la clave ya existe en Redis se devuelve la respuesta original, sin crear otra transacción.

`pagos-api` inserta la transacción en PostgreSQL en estado `pendiente` y publica un mensaje en `pagos.autorizar`. Recién ahí responde `202 Accepted` al comercio.

## Autorización

`pagos-worker` toma el mensaje y llama al procesador de tarjetas con un timeout de 8 segundos. Según la respuesta, la transacción pasa a `aprobada`, `rechazada` o, si el procesador no contesta, vuelve a la cola con backoff.

Si la base de datos no responde cuando el worker quiere actualizar el estado, el mensaje no se confirma (no hay `ack`) y RabbitMQ lo vuelve a entregar. Por eso la actualización de estado tiene que ser idempotente.

## Notificación

Cada cambio de estado genera un evento en `pagos.eventos`. El servicio de notificaciones lo consume y manda el webhook al comercio, reintentando durante 24 horas si el comercio no responde con 2xx.
