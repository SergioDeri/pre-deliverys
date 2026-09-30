# Runbook: la cola de pagos no avanza

## Síntomas

El lag de `pagos.autorizar` supera los 1.000 mensajes y sigue subiendo. Los comercios ven transacciones que quedan en `pendiente` durante minutos.

## Diagnóstico

1. Ver en la consola de RabbitMQ si la cola tiene consumidores conectados. Cero consumidores significa que los workers están caídos o no pueden conectarse al broker.
2. Si hay consumidores pero el `ack rate` es cercano a cero, los workers están bloqueados esperando algo: casi siempre la base de datos o el procesador de tarjetas.
3. Revisar los logs de `pagos-worker` buscando timeouts.

## Mitigación

- Workers caídos: `kubectl rollout restart deployment/pagos-worker`.
- Workers bloqueados por la base: seguir el runbook de base de datos, no escalar workers (más workers piden más conexiones).
- Procesador de tarjetas lento: activar el modo degradado desde el feature flag `procesador_timeout_corto`.
