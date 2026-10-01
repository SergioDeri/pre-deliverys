# Runbook: la base de datos no responde

## Síntomas

- `pagos-api` devuelve 503 y en los logs aparece `PoolTimeout` o `connection refused` contra PostgreSQL.
- El lag de `pagos.autorizar` crece porque los workers no pueden confirmar mensajes.
- La alerta `PostgresUnreachable` o `PgBouncerWaitingClients` está disparada.

## Diagnóstico

1. Verificar si el primario está vivo: `kubectl exec -it pgbouncer-0 -- psql -c 'select 1'`.
2. Mirar `SHOW POOLS;` en PgBouncer. Muchos `cl_waiting` con `sv_active` al máximo indican saturación de conexiones, no una caída.
3. Buscar consultas largas o bloqueos en `pg_stat_activity` (más de 30 segundos en estado `active`).

## Mitigación

- Si es saturación: cancelar las consultas largas con `pg_cancel_backend` y bajar temporalmente las réplicas de `pagos-worker` para liberar conexiones.
- Si el primario cayó: seguir el failover a la réplica documentado por el equipo de datos y actualizar el host en PgBouncer.
- No reiniciar `pagos-api` en masa: todas las réplicas intentarían reconectar a la vez y empeora el problema.

## Después del incidente

Los mensajes que no se confirmaron se vuelven a entregar solos. Revisar la DLQ `pagos.autorizar.dlq` por si algo llegó a los cinco intentos.
