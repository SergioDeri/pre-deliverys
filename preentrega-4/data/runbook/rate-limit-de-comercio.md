# Runbook: un comercio recibe 429

## Síntomas

Un comercio reporta que `POST /v1/payments` le devuelve `429 Too Many Requests` con el código `PF-4290`. Los demás comercios funcionan bien.

## Diagnóstico

1. El límite lo aplica el api-gateway por comercio: 50 requests por segundo con una ráfaga de 100. Ver en los logs del gateway cuántas respuestas 429 tuvo ese comercio en la última hora.
2. Si el comercio reintenta de inmediato cada 429, se retroalimenta: cada reintento cuenta contra el mismo límite.
3. Revisar si el comercio tiene un límite especial en el `ConfigMap` `gateway-limites`. Los comercios grandes suelen tener uno.

## Mitigación

- Pedirle al comercio que respete el encabezado `Retry-After` y que reintente con backoff.
- Si el volumen es legítimo, subir su límite en `gateway-limites` y recargar NGINX con `kubectl rollout restart deployment/api-gateway`. Nunca desactivar el rate limiting global.
