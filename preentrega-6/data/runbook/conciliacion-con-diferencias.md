# Runbook: la conciliación terminó con diferencias

## Síntomas

El resumen de `#pagos-conciliacion` muestra más de 20 transacciones `con_diferencia`, o el job `conciliacion` falló y Finanzas no puede liberar fondos.

## Diagnóstico

1. Si el job falló con `PF-7002`, el archivo de Cobralia no llegó. Confirmar con Cobralia por el canal de soporte antes de hacer cualquier otra cosa.
2. Si terminó con diferencias, agruparlas por motivo en la tabla `conciliaciones`. Muchas diferencias de monto juntas suelen ser un cambio de comisión que Cobralia no avisó.
3. Transacciones que faltan en el archivo casi siempre son aprobaciones que llegaron después del corte de las 23:59 y aparecen en el archivo del día siguiente.

## Mitigación

- Volver a correr el job para una fecha puntual: `kubectl create job --from=cronjob/conciliacion conciliacion-manual -- --fecha 2026-09-25`.
- No marcar a mano transacciones como `conciliada`: Finanzas audita esa tabla y cada cambio tiene que venir del job.
