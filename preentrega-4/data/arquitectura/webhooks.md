# Webhooks a comercios

## Entrega

El servicio `notificaciones` consume `pagos.eventos` y hace un `POST` a la URL que el comercio registró en el panel. El cuerpo es el evento en JSON y el timeout de cada intento es de 5 segundos.

## Firma

Cada webhook lleva el encabezado `X-PayFlow-Signature` con un HMAC-SHA256 del cuerpo, calculado con el secreto de webhooks del comercio. El comercio tiene que recalcularlo y descartar el mensaje si no coincide. El secreto se puede regenerar desde el panel; durante una hora se firman los webhooks con los dos secretos.

## Reintentos

Si el comercio no responde 2xx, `notificaciones` reintenta con backoff exponencial durante 24 horas: a los 30 segundos, 2 minutos, 10 minutos y después cada hora. Pasadas las 24 horas el evento va a `pagos.eventos.dlq` y el comercio lo puede pedir de nuevo con `GET /v1/events/{id}`.

## Orden

Los webhooks no garantizan orden. Un comercio puede recibir `aprobada` antes que `pendiente`, así que cada evento trae `occurred_at` y el comercio tiene que quedarse con el más reciente.
