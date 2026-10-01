# Runbook: un comercio dice que no recibe webhooks

## Síntomas

El comercio reporta transacciones que en su sistema siguen `pendiente` aunque en PayFlow están `aprobada`. En el panel, los eventos del comercio figuran con intentos fallidos.

## Diagnóstico

1. Buscar los últimos intentos en los logs de `notificaciones` filtrando por el id del comercio. El código de respuesta dice casi todo.
2. Si el comercio responde 401 o 403, lo más probable es que esté validando `X-PayFlow-Signature` con un secreto viejo.
3. Si los intentos terminan en timeout, su endpoint tarda más de 5 segundos: el comercio tiene que responder 200 enseguida y procesar después.
4. Si no hay intentos, revisar que el consumidor de `pagos.eventos` tenga lag cercano a cero.

## Mitigación

- Pedirle al comercio que corrija su endpoint y reenviar los eventos desde el panel o con `POST /internal/events/replay?comercio={id}`.
- Los eventos que ya pasaron a `pagos.eventos.dlq` no se reintentan solos: hay que reenviarlos.
